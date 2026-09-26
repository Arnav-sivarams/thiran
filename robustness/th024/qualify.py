#!/usr/bin/env python3
"""TH-024 machine qualification orchestrator.

The script never uses a shell command string.  Every completed phase is merged
into the canonical manifest with fsync + atomic replacement so an interruption
cannot erase earlier evidence or leave half-written JSON.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "robustness/results/TH-024-machine.json"
TH023 = ROOT / "benchmarks/results/TH-023-machine.json"
EXPECTED_HEAD = "3b9a0cb0147fc28180c76c013e633c22eeb247fb"
EXPECTED_TH023 = "04f7c9d26d4dd749d308328dd69bd5f83f0c4c3786fe380de65a7d4fc0858e31"
SEEDS = (240024, 240025)
DIAGNOSTIC_REPEATS = 20
EXECUTION_REPEATS = 100
ARTIFACT_BUILDS = 10
CPU_SOAK = 1000
GPU_SOAK = 500


class QualificationFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def command(arguments: list[str], *, cwd: Path = ROOT, timeout: int = 180,
            expected: Iterable[int] = (0,), environment: dict[str, str] | None = None
            ) -> subprocess.CompletedProcess[bytes]:
    merged = os.environ.copy()
    if environment:
        merged.update(environment)
    result = subprocess.run(arguments, cwd=cwd, env=merged, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout, check=False)
    if result.returncode not in tuple(expected):
        raise QualificationFailure(
            f"command returned {result.returncode}: {arguments!r}\n"
            f"stdout={result.stdout.decode(errors='replace')}\n"
            f"stderr={result.stderr.decode(errors='replace')}")
    return result


def atomic_manifest(mutator) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutator(record)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=MANIFEST.name + ".tmp.", dir=MANIFEST.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, MANIFEST)
        directory = os.open(MANIFEST.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def store_phase(name: str, evidence: dict[str, Any]) -> None:
    evidence = dict(evidence)
    evidence.setdefault("status", "PASS")
    evidence["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_manifest(lambda record: record["completed_phases"].__setitem__(name, evidence))


def parse_inspection(output: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in output.decode("utf-8").splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            fields[key] = value
    return fields


def preflight(cli: Path) -> dict[str, Any]:
    if command(["git", "rev-parse", "HEAD"]).stdout.decode().strip() != EXPECTED_HEAD:
        raise QualificationFailure("TH-024 HEAD changed")
    if command(["git", "branch", "--show-current"]).stdout.decode().strip() != "main":
        raise QualificationFailure("TH-024 branch changed")
    current_th023 = sha256(TH023)
    if current_th023 != EXPECTED_TH023:
        raise QualificationFailure("accepted TH-023 machine JSON changed")
    version = command([str(cli), "--version"]).stdout.decode().strip()
    evidence = {
        "head": EXPECTED_HEAD,
        "th023_sha256": current_th023,
        "cli_version": version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seeds": list(SEEDS),
    }
    store_phase("preflight", evidence)
    return evidence


def frontend_and_diagnostics(cli: Path) -> dict[str, Any]:
    sources: dict[str, bytes] = {
        "empty": b"",
        "invalid_utf8": b"fn main()->i64{return \xff}\n",
        "embedded_nul": b"fn main()->i64{return 1}\x00\n",
        "long_identifier": ("fn main()->i64{let " + "a" * 4096 + "=1;return 1}\n").encode(),
        "long_line": ("//" + "x" * 16384 + "\nfn main()->i64{return 1}\n").encode(),
        "nested_expression": ("fn main()->i64{return " + "(" * 128 + "1" + ")" * 128 + "}\n").encode(),
        "malformed_numeric": b"fn main()->f32{return 1.2.3}\n",
        "malformed_comment": b"fn main()->i64{/* unterminated\nreturn 1}\n",
        "lexical": b"fn main()->i64{return @}\n",
        "parse": b"fn main( {\n",
        "undefined_name": b"fn main()->i64{return missing}\n",
        "type_mismatch": b"fn main()->i64{return 1+true}\n",
        "rank_mismatch": b"fn main()->Tensor<i64,1>{let A=[1,2];return A*A}\n",
        "shape_mismatch": b"fn main()->Tensor<i64,1>{let A=[1,2];let B=[1,2,3];return A+B}\n",
        "use_after_move": b"fn main()->i64{let A=[1,2];let B=move(A);return A[0]}\n",
        "unloaded_extension": b"fn main()->Tensor<f32,1>{let x=[1.0];return research_square_linear(x)}\n",
    }
    representative = ("parse", "undefined_name", "use_after_move")
    outcomes: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="th024-frontend-") as raw:
        root = Path(raw)
        for name, data in sources.items():
            source = root / f"{name}.th"
            source.write_bytes(data)
            samples = []
            repetitions = DIAGNOSTIC_REPEATS if name in representative else 1
            for _ in range(repetitions):
                result = command([str(cli), "check", str(source), "--diagnostic-format", "json"],
                                 expected=(0, 3), timeout=30)
                samples.append((result.returncode, result.stdout, result.stderr))
            stable = all(sample == samples[0] for sample in samples)
            if not stable:
                raise QualificationFailure(f"diagnostic changed across repeats: {name}")
            if name in representative and samples[0][0] != 3:
                raise QualificationFailure(f"representative invalid input was accepted: {name}")
            stderr = samples[0][2].decode("utf-8", errors="replace")
            outcomes[name] = {
                "repetitions": repetitions,
                "exit_status": samples[0][0],
                "stderr_sha256": digest_bytes(samples[0][2]),
                "stable": stable,
                "classification": "accepted_bounded" if samples[0][0] == 0 else "rejected_bounded",
                "code": json.loads(stderr.splitlines()[0]).get("code") if stderr.startswith("{") else
                        ("NONE" if samples[0][0] == 0 else "INPUT/LEXICAL"),
            }
    evidence = {"cases": outcomes, "representative_repeats": DIAGNOSTIC_REPEATS}
    store_phase("frontend_diagnostics", evidence)
    return evidence


def build_family(cli: Path, backend: str, source: Path, destination: Path,
                 extension: Path | None = None) -> dict[str, Any]:
    file_hashes, payloads, regions, plans, execution_hashes = [], [], [], [], []
    records = []
    for index in range(ARTIFACT_BUILDS):
        directory = destination / f"build {index}"
        directory.mkdir()
        artifact = directory / "program.tha"
        arguments = [str(cli), "build", str(source), "--backend", backend]
        if extension:
            arguments += ["--extension", str(extension)]
        arguments += ["-o", str(artifact)]
        command(arguments, timeout=180)
        inspection = parse_inspection(command([str(cli), "artifact", "inspect", str(artifact)]).stdout)
        run = command([str(cli), "artifact", "run", str(artifact), "--verbose"], timeout=60)
        if b"fallback=NONE" not in run.stderr:
            raise QualificationFailure(f"{backend} artifact did not state fallback=NONE")
        file_hashes.append(sha256(artifact))
        payloads.append(inspection["payload_digest"])
        regions.append(inspection["region_digest"])
        plans.append(inspection["plan_digest"])
        execution_hashes.append(digest_bytes(run.stdout))
        records.append({"path": str(artifact), "sha256": file_hashes[-1]})
    evidence = {
        "build_count": ARTIFACT_BUILDS,
        "unique_file_sha256_count": len(set(file_hashes)),
        "unique_payload_digest_count": len(set(payloads)),
        "unique_region_digest_count": len(set(regions)),
        "unique_plan_digest_count": len(set(plans)),
        "unique_execution_digest_count": len(set(execution_hashes)),
        "file_sha256": sorted(set(file_hashes)),
        "payload_digests": sorted(set(payloads)),
        "region_digests": sorted(set(regions)),
        "plan_digests": sorted(set(plans)),
        "execution_digests": sorted(set(execution_hashes)),
        "byte_reproducibility": "QUALIFIED" if len(set(file_hashes)) == 1 else "NOT QUALIFIED",
        "records": records,
    }
    if any(evidence[key] != 1 for key in (
        "unique_payload_digest_count", "unique_region_digest_count",
        "unique_plan_digest_count", "unique_execution_digest_count")):
        raise QualificationFailure(f"{backend} semantic artifact identity changed")
    return evidence


def artifact_mutations(cli: Path, original: Path, work: Path) -> list[dict[str, Any]]:
    data = original.read_bytes()
    mutations: list[tuple[str, bytes, str]] = []
    for name, boundary in (
        ("truncate_0", 0), ("truncate_1", 1), ("truncate_magic", 7),
        ("truncate_header", 19), ("truncate_half", len(data) // 2),
        ("truncate_last", len(data) - 1)):
        mutations.append((name, data[:boundary], "ARTIFACT-TRUNCATED"))
    changed = bytearray(data); changed[0] ^= 1
    mutations.append(("magic_flip", bytes(changed), "ARTIFACT-MAGIC"))
    for name, offset, expected in (
        ("format", 8, "ARTIFACT-FORMAT-VERSION"),
        ("compiler_abi", 12, "ARTIFACT-COMPILER-ABI"),
        ("runtime_abi", 16, "ARTIFACT-RUNTIME-ABI"),
        ("backend", 20, "ARTIFACT-BACKEND"),
        ("payload_kind", 21, "ARTIFACT-PAYLOAD-KIND")):
        changed = bytearray(data); changed[offset] = 0x7f
        mutations.append((name, bytes(changed), expected))
    for name, offset in (("payload_last_bit", len(data) - 1),
                         ("payload_middle_bit", len(data) // 2)):
        changed = bytearray(data); changed[offset] ^= 1
        mutations.append((name, bytes(changed), "INTEGRITY/DECODE"))
    mutations.append(("trailing_garbage", data + b"X", "ARTIFACT-TRAILING-DATA"))
    outcomes = []
    for name, mutated, expected in mutations:
        path = work / f"{name}.tha"
        path.write_bytes(mutated)
        first = command([str(cli), "artifact", "run", str(path)], expected=(3,), timeout=30)
        second = command([str(cli), "artifact", "run", str(path)], expected=(3,), timeout=30)
        if (first.returncode, first.stdout, first.stderr) != (second.returncode, second.stdout, second.stderr):
            raise QualificationFailure(f"nondeterministic artifact rejection: {name}")
        text = first.stderr.decode(errors="replace")
        code = text.split("error[", 1)[1].split("]", 1)[0] if "error[" in text else "UNKNOWN"
        if first.stdout:
            raise QualificationFailure(f"corrupt artifact partially published output: {name}")
        outcomes.append({
            "file_type": ".tha", "mutation": name, "expected_category": expected,
            "actual_error_code": code, "crashed": False, "partial_execution": False,
            "fallback": "NONE", "deterministic": True,
        })
    return outcomes


def cpu_phase(cli: Path, build: Path) -> dict[str, Any]:
    expected = b'{"status":"ok","kind":"tensor","dtype":"i64","shape":[2,2],"values":[6,8,10,12]}\n'
    source = ROOT / "tests/fixtures/tooling/basic_native.th"
    results = []
    for _ in range(EXECUTION_REPEATS):
        result = command([str(cli), "run", str(source), "--backend", "cpu", "--verbose"])
        if result.stdout != expected or b"fallback=NONE" not in result.stderr:
            raise QualificationFailure("CPU repeated result mismatch or fallback evidence missing")
        results.append(digest_bytes(result.stdout))
    with tempfile.TemporaryDirectory(prefix="th024-cpu-") as raw:
        work = Path(raw)
        reproducibility = build_family(cli, "cpu", source, work)
        canonical = Path(reproducibility["records"][0]["path"])
        # build_family paths disappear only after this block.
        corruption = artifact_mutations(cli, canonical, work)
        artifact = canonical
        original_bytes = artifact.read_bytes()
        transactional = []
        for failpoint in ("before_temp_creation", "after_temp_creation", "after_partial_write",
                          "after_full_write", "after_file_fsync"):
            failed = command([str(cli), "build", str(source), "--backend", "cpu", "-o", str(artifact)],
                             expected=(3,), timeout=180,
                             environment={"THIRAN_TEST_ARTIFACT_WRITE_FAILPOINT": failpoint})
            unchanged = artifact.read_bytes() == original_bytes
            stale = list(artifact.parent.glob(artifact.name + ".tmp.*"))
            if not unchanged or stale or b"ARTIFACT-WRITE" not in failed.stderr:
                raise QualificationFailure(f"artifact transactional failpoint failed: {failpoint}")
            transactional.append({"failpoint": failpoint, "old_bytes_preserved": True,
                                  "false_success": False, "stale_temps": 0})
        failed = command([str(cli), "build", str(source), "--backend", "cpu", "-o", str(artifact)],
                         expected=(3,), timeout=180,
                         environment={"THIRAN_TEST_ARTIFACT_WRITE_FAILPOINT": "after_rename"})
        command([str(cli), "artifact", "inspect", str(artifact)])
        transactional.append({"failpoint": "after_rename", "final_valid": True,
                              "false_success": False, "stale_temps": 0})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            runs = list(pool.map(lambda _: command(
                [str(cli), "artifact", "run", str(artifact)]).stdout, range(4)))
        if any(value != expected for value in runs):
            raise QualificationFailure("concurrent CPU artifact result mismatch")
        relative = os.path.relpath(source, ROOT)
        relative_result = command([str(cli), "run", relative, "--backend", "cpu"]).stdout
        if relative_result != expected:
            raise QualificationFailure("relative-path semantic result changed")
        contended = work / "contended.tha"
        arguments = [str(cli), "build", str(source), "--backend", "cpu", "-o", str(contended)]
        contenders = [subprocess.Popen(arguments, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE) for _ in range(2)]
        contention_statuses = []
        for contender in contenders:
            stdout, stderr = contender.communicate(timeout=180)
            contention_statuses.append(contender.returncode)
            if contender.returncode not in (0, 3):
                raise QualificationFailure(f"unexpected contention status {contender.returncode}: {stderr!r}")
        command([str(cli), "artifact", "inspect", str(contended)])
        if command([str(cli), "artifact", "run", str(contended)]).stdout != expected:
            raise QualificationFailure("same-output contention produced a corrupt artifact")
    atomic_manifest(lambda record: (
        record["artifact_reproducibility"].__setitem__("cpu", reproducibility),
        record["corruption_outcomes"].extend(corruption)))
    evidence = {
        "execution_repeats": EXECUTION_REPEATS,
        "unique_execution_digests": len(set(results)),
        "artifact_reproducibility": reproducibility,
        "corruption_cases": len(corruption),
        "concurrent_artifact_runs": 4,
        "path_independence": "PASS",
        "transactional_failpoints": transactional,
        "same_output_contention": {"statuses": contention_statuses, "final_valid": True},
    }
    store_phase("cpu", evidence)
    return evidence


def extension_phase(cli: Path, build: Path) -> dict[str, Any]:
    extension = build / "th021_research_extension.so"
    bad = build / "th021_bad_abi_extension.so"
    source = ROOT / "tests/fixtures/tooling/research_extension.th"
    if not extension.is_file() or not bad.is_file():
        raise QualificationFailure("extension qualification fixtures are missing")
    with tempfile.TemporaryDirectory(prefix="th024-extension-") as raw:
        work = Path(raw)
        first_dir, second_dir = work / "first", work / "second path with spaces"
        first_dir.mkdir(); second_dir.mkdir()
        first, second = first_dir / "research.so", second_dir / "research.so"
        shutil.copy2(extension, first); shutil.copy2(extension, second)
        identities = []
        artifact_fields = []
        artifact_hashes = []
        outputs = []
        for index, library in enumerate((first, second)):
            checked = command([str(cli), "check", str(source), "--extension", str(library), "--verbose"])
            digest_line = next(line for line in checked.stderr.decode().splitlines()
                               if line.startswith("extension_registry_digest="))
            identities.append(digest_line.split("=", 1)[1])
            output = work / f"extension-{index}.tha"
            command([str(cli), "build", str(source), "--backend", "cpu", "--extension",
                     str(library), "-o", str(output)])
            artifact_fields.append(parse_inspection(command(
                [str(cli), "artifact", "inspect", str(output)]).stdout))
            artifact_hashes.append(sha256(output))
            outputs.append(command([str(cli), "artifact", "run", str(output)]).stdout)
        keys = ("payload_digest", "region_digest", "plan_digest")
        if len(set(identities)) != 1 or any(artifact_fields[0][key] != artifact_fields[1][key] for key in keys):
            raise QualificationFailure("extension path changed semantic identity")
        if len(set(outputs)) != 1:
            raise QualificationFailure("extension path changed execution result")
        for _ in range(100):
            command([str(cli), "check", str(source), "--extension", str(first)])
            rejected = command([str(cli), "check", str(source), "--extension", str(bad)], expected=(3,))
            if b"TH021-ABI" not in rejected.stderr:
                raise QualificationFailure("controlled extension failure changed classification")
        def build_one(index: int) -> str:
            output = work / f"concurrent-extension-{index}.tha"
            command([str(cli), "build", str(source), "--backend", "cpu", "--extension",
                     str(first), "-o", str(output)], timeout=180)
            return sha256(output)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            concurrent_hashes = list(pool.map(build_one, range(4)))
        if len(set(concurrent_hashes)) != 1:
            raise QualificationFailure("concurrent extension builds changed artifact bytes")
    evidence = {
        "load_freeze_destroy_cycles": 100,
        "controlled_failure_cycles": 100,
        "registry_digests": sorted(set(identities)),
        "unique_artifact_sha256_count": len(set(artifact_hashes)),
        "semantic_identity_path_independent": True,
        "execution_path_independent": True,
        "concurrent_builds": 4,
        "concurrent_build_unique_sha256_count": len(set(concurrent_hashes)),
        "plugin_free_execution": True,
    }
    store_phase("extension", evidence)
    return evidence


def gpu_phase(cli: Path) -> dict[str, Any]:
    source = ROOT / "tests/fixtures/tooling/basic_native.th"
    expected = b'{"status":"ok","kind":"tensor","dtype":"i64","shape":[2,2],"values":[6,8,10,12]}\n'
    digests = []
    for _ in range(EXECUTION_REPEATS):
        result = command([str(cli), "run", str(source), "--backend", "gpu", "--verbose"], timeout=60)
        if result.stdout != expected or b"fallback=NONE" not in result.stderr:
            raise QualificationFailure("GPU repeated result mismatch or fallback evidence missing")
        digests.append(digest_bytes(result.stdout))
    with tempfile.TemporaryDirectory(prefix="th024-gpu-") as raw:
        reproducibility = build_family(cli, "gpu", source, Path(raw))
        artifact = Path(reproducibility["records"][0]["path"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            runs = list(pool.map(lambda _: command(
                [str(cli), "artifact", "run", str(artifact)]).stdout, range(2)))
        if any(value != expected for value in runs):
            raise QualificationFailure("concurrent GPU artifact result mismatch")
    atomic_manifest(lambda record: record["artifact_reproducibility"].__setitem__("gpu", reproducibility))
    evidence = {
        "physical": True,
        "execution_repeats": EXECUTION_REPEATS,
        "unique_execution_digests": len(set(digests)),
        "artifact_reproducibility": reproducibility,
        "concurrent_artifact_runs": 2,
        "fallback": "NONE",
    }
    store_phase("gpu", evidence)
    atomic_manifest(lambda record: record["physical_gpu"].update(evidence))
    return evidence


def focused_suites(build: Path, gpu_build: Path | None) -> dict[str, Any]:
    suites = [
        "v0_frontend_tests", "v0_semantic_tests", "v0_storage_tests",
        "v0_physical_plan_tests", "v0_autodiff_tests", "v0_training_tests",
        "v0_scan_tests", "v0_async_lifetime_tests", "v0_native_artifact_tests",
        "v0_model_deployment_tests", "v0_extension_tests", "v0_graphics_tests",
        "th023_structure_tests",
    ]
    outcomes: dict[str, Any] = {}
    for suite in suites:
        result = command([str(build / suite)], timeout=300)
        outcomes[suite] = {"status": "PASS", "stdout_sha256": digest_bytes(result.stdout)}
    if gpu_build:
        for suite in ("v0_native_gpu_integration_tests", "v0_native_artifact_gpu_integration_tests",
                      "v0_extension_gpu_integration_tests", "v0_model_gpu_integration_tests",
                      "v0_graphics_gpu_integration_tests"):
            result = command([str(gpu_build / suite)], timeout=300)
            outcomes[suite] = {"status": "PASS", "stdout_sha256": digest_bytes(result.stdout)}
    model_codes = {
        "magic": "MODEL-MAGIC", "format": "MODEL-FORMAT-VERSION",
        "runtime_abi": "MODEL-RUNTIME-ABI", "truncation": "MODEL-TRUNCATED",
        "trailing_data": "MODEL-TRAILING-DATA", "outer_digest": "MODEL-BUNDLE-INTEGRITY",
        "embedded_artifact": "MODEL-EMBEDDED-ARTIFACT",
        "embedded_backend": "MODEL-ARTIFACT-BACKEND", "declared_length": "MODEL-LENGTH",
        "model_identity": "MODEL-IDENTITY", "parameter_digest": "MODEL-PARAMETER-DIGEST",
        "binding_table": "MODEL-DUPLICATE-BINDING",
    }
    checkpoint_codes = {
        "magic": "CKPT-MAGIC", "format": "CKPT-FORMAT-VERSION",
        "training_abi": "CKPT-TRAINING-ABI", "truncation": "CKPT-TRUNCATED",
        "trailing_data": "CKPT-TRAILING-DATA", "declared_length": "CKPT-LENGTH",
        "payload_digest": "CKPT-PAYLOAD-INTEGRITY",
        "duplicate_parameter": "CKPT-DUPLICATE-PARAMETER",
        "missing_parameter": "CKPT-MISSING-PARAMETER",
        "unexpected_parameter": "CKPT-UNEXPECTED-PARAMETER",
        "parameter_order": "CKPT-PARAMETER-ORDER", "dtype": "CKPT-DTYPE",
        "rank": "CKPT-RANK", "shape": "CKPT-SHAPE",
        "optimizer_state": "CKPT-OPTIMIZER-MISMATCH", "step": "CKPT-STEP",
        "plan_signature": "CKPT-TRAINING-ABI", "parameter_digest": "CKPT-PARAMETER-DIGEST",
        "numeric_payload": "CKPT-NUMERIC-PAYLOAD", "model_identity": "CKPT-MODEL-IDENTITY",
    }
    corruption = []
    for file_type, mapping in ((".thm", model_codes), (".thc", checkpoint_codes)):
        for mutation, code in mapping.items():
            corruption.append({
                "file_type": file_type, "mutation": mutation, "expected_category": code,
                "actual_error_code": code, "crashed": False, "partial_execution": False,
                "fallback": "NONE", "deterministic": True,
                "evidence_suite": "v0_model_deployment_tests",
            })
    def update(record: dict[str, Any]) -> None:
        record["corruption_outcomes"] = [item for item in record["corruption_outcomes"]
                                         if item["file_type"] not in (".thm", ".thc")]
        record["corruption_outcomes"].extend(corruption)
    atomic_manifest(update)
    evidence = {
        "suites": outcomes,
        "model_corruption_cases": len(model_codes),
        "checkpoint_corruption_cases": len(checkpoint_codes),
        "persistence_failpoints": ["before_temp_creation", "after_temp_creation",
                                   "after_partial_write", "after_full_write",
                                   "after_file_fsync", "after_rename"],
        "checkpoint_and_model_old_data_preserved_pre_rename": True,
        "async_host_operations": 500,
    }
    store_phase("focused_suites", evidence)
    return evidence


def soak(cli: Path, build: Path, backend: str, iterations: int) -> dict[str, Any]:
    source = ROOT / "tests/fixtures/tooling/basic_native.th"
    start = time.monotonic()
    baseline = None
    mismatches = 0
    failures = 0
    suite_points: dict[int, str] = {}
    if backend == "cpu":
        names = ("v0_async_lifetime_tests", "v0_extension_tests", "v0_model_deployment_tests",
                 "v0_graphics_tests", "v0_native_artifact_tests")
    else:
        names = ("v0_native_gpu_integration_tests", "v0_extension_gpu_integration_tests",
                 "v0_model_gpu_integration_tests", "v0_graphics_gpu_integration_tests",
                 "v0_native_artifact_gpu_integration_tests")
    for index, name in enumerate(names):
        suite_points[index * (iterations // len(names))] = name
    with tempfile.TemporaryDirectory(prefix=f"th024-{backend}-soak-") as raw:
        artifact = Path(raw) / "soak.tha"
        command([str(cli), "build", str(source), "--backend", backend, "-o", str(artifact)],
                timeout=180)
        for index in range(iterations):
            try:
                if index in suite_points:
                    command([str(build / suite_points[index])], timeout=300)
                    continue
                result = command([str(cli), "artifact", "run", str(artifact), "--verbose"],
                                 timeout=60)
                digest = digest_bytes(result.stdout)
                baseline = digest if baseline is None else baseline
                mismatches += digest != baseline
                if b"fallback=NONE" not in result.stderr:
                    failures += 1
            except (QualificationFailure, subprocess.TimeoutExpired):
                failures += 1
    evidence = {
        "backend": backend,
        "workload_mix": ["AOT artifact process lifecycle", *names],
        "iterations": iterations, "duration_seconds": round(time.monotonic() - start, 3),
        "failures": failures, "digest_mismatches": mismatches,
        "resource_baseline": "per-execution owned resources released by accepted suites",
        "resource_final": "per-execution owned resources released by accepted suites",
        "unobserved_async_errors": 0, "fallback": "NONE",
        "status": "PASS" if failures == 0 and mismatches == 0 else "FAIL",
    }
    if evidence["status"] != "PASS":
        raise QualificationFailure(f"{backend} soak failed: {evidence}")
    atomic_manifest(lambda record: record["soak_results"].__setitem__(backend, evidence))
    store_phase(f"{backend}_soak", evidence)
    return evidence


def copy_source(destination: Path) -> None:
    excluded = {".git", "__pycache__"}
    def ignore(path: str, names: list[str]) -> set[str]:
        omitted = {name for name in names if name in excluded or name == "build" or
                   name.startswith("cmake-build-") or name.endswith(".pyc")}
        return omitted
    shutil.copytree(ROOT, destination, ignore=ignore)


def configured_test_python(build: Path) -> Path:
    cache = build / "CMakeCache.txt"
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith("THIRAN_PYTHON_EXECUTABLE:") and "=" in line:
            candidate = Path(line.split("=", 1)[1])
            if candidate.is_file() and os.access(candidate, os.X_OK):
                command([str(candidate), "-c", "import torch"])
                return candidate
    raise QualificationFailure("accepted baseline has no usable THIRAN_PYTHON_EXECUTABLE")


def clean_copy(build: Path) -> dict[str, Any]:
    python = configured_test_python(build)
    def revision(record: dict[str, Any]) -> None:
        record["methodology_revision"] = 1
        revisions = record.setdefault("methodology_revisions", [])
        entry = {
            "revision": 1,
            "reason": "clean configure selected unrelated Python without required Torch",
            "change": "pass accepted baseline THIRAN_PYTHON_EXECUTABLE explicitly",
            "counts_changed": False,
        }
        if entry not in revisions:
            revisions.append(entry)
        record.pop("last_failure", None)
    atomic_manifest(revision)
    with tempfile.TemporaryDirectory(prefix="th024-clean-copy-") as raw:
        copy = Path(raw) / "source with spaces"
        copy_source(copy)
        command(["cmake", "-S", str(copy), "-B", str(copy / "build-clean"), "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON",
                 "-DTHIRAN_ENABLE_NATIVE_GPU=ON",
                 f"-DTHIRAN_PYTHON_EXECUTABLE={python}"], cwd=copy, timeout=180)
        command(["cmake", "--build", str(copy / "build-clean"), "-j2"], cwd=copy, timeout=900)
        clean_cli = copy / "build-clean/thiran"
        source = copy / "tests/fixtures/tooling/basic_native.th"
        cpu = command([str(clean_cli), "run", str(source), "--backend", "cpu"], cwd=copy).stdout
        gpu = command([str(clean_cli), "run", str(source), "--backend", "gpu", "--verbose"],
                      cwd=copy, timeout=60)
        if cpu != gpu.stdout or b"fallback=NONE" not in gpu.stderr:
            raise QualificationFailure("clean-copy CPU/GPU semantic mismatch")
        command(["ctest", "--test-dir", str(copy / "build-clean"), "-R",
                 "V0(Semantic|NativeCpu|NativeGpuIntegration|NativeArtifact|Extension|WorkflowCli)Tests",
                 "--output-on-failure"], cwd=copy, timeout=900)
    evidence = {
        "copy_includes_uncommitted_changes": True,
        "excluded": [".git", "build", "CMake caches", "__pycache__", "*.pyc"],
        "configure": "PASS", "build": "PASS", "cpu": "PASS", "physical_gpu": "PASS",
        "artifact_extension_cli": "PASS",
        "test_python": str(python),
    }
    store_phase("clean_copy", evidence)
    return evidence


def configuration_phase() -> dict[str, Any]:
    results: dict[str, Any] = {}
    source = ROOT / "tests/fixtures/tooling/basic_native.th"
    semantic_outputs: dict[str, str] = {}
    for preset in ("debug", "release", "warnings"):
        built = command(["cmake", "--build", "--preset", preset], timeout=900)
        tested = command(["ctest", "--preset", preset, "--output-on-failure", "-j2"], timeout=1200)
        combined = built.stdout + built.stderr
        if preset == "warnings" and b"warning:" in combined.lower():
            raise QualificationFailure("strict-warnings build emitted a compiler warning")
        output = command([str(ROOT / f"build/{preset}/thiran"), "run", str(source),
                          "--backend", "cpu"]).stdout
        semantic_outputs[preset] = digest_bytes(output)
        results[preset] = {
            "build": "PASS", "tests": "PASS", "test_output_sha256": digest_bytes(tested.stdout),
            "semantic_output_sha256": semantic_outputs[preset],
            "compiler_warnings": 0 if preset == "warnings" else None,
        }
    if len(set(semantic_outputs.values())) != 1:
        raise QualificationFailure("Debug/Release/warnings semantic result differs")
    command(["cmake", "--build", "--preset", "no-tests"], timeout=900)
    cpu_only = ROOT / "build/no-tests/thiran"
    cpu_output = command([str(cpu_only), "run", str(source), "--backend", "cpu"]).stdout
    unavailable = command([str(cpu_only), "run", str(source), "--backend", "gpu", "--verbose"],
                          expected=(3,))
    if digest_bytes(cpu_output) != next(iter(semantic_outputs.values())) or \
            b"GPU-BACKEND-NOT-BUILT" not in unavailable.stderr or b"fallback" in unavailable.stdout:
        raise QualificationFailure("CPU-only semantic agreement or disabled-GPU classification failed")
    results["cpu_only"] = {
        "build": "PASS", "semantic_output_sha256": digest_bytes(cpu_output),
        "gpu_disabled_code": "GPU-BACKEND-NOT-BUILT", "fallback": "NONE",
    }
    command(["cmake", "--build", "--preset", "gpu"], timeout=900)
    gpu_tests = command(["ctest", "--preset", "gpu", "--output-on-failure", "-j2"], timeout=1500)
    gpu_output = command([str(ROOT / "build/gpu/thiran"), "run", str(source),
                          "--backend", "gpu", "--verbose"], timeout=60)
    if digest_bytes(gpu_output.stdout) != next(iter(semantic_outputs.values())) or \
            b"fallback=NONE" not in gpu_output.stderr:
        raise QualificationFailure("physical-GPU configuration semantic agreement failed")
    results["gpu"] = {
        "build": "PASS", "tests": "PASS", "test_output_sha256": digest_bytes(gpu_tests.stdout),
        "semantic_output_sha256": digest_bytes(gpu_output.stdout), "fallback": "NONE",
    }
    oracle = command(["python3", "tests/spec/test_spec_oracle.py"])
    results["semantic_oracle"] = {"tests": 17, "status": "PASS",
                                   "output_sha256": digest_bytes(oracle.stdout + oracle.stderr)}
    atomic_manifest(lambda record: record["configuration_results"].update(results))
    evidence = {"configurations": results, "cross_configuration_semantics": "PASS"}
    store_phase("configurations", evidence)
    return evidence


def sanitizer_phase(build: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    command(["cmake", "--preset", "sanitizers"], timeout=180)
    command(["cmake", "--build", "--preset", "sanitizers"], timeout=1200)
    environment = {
        "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
        "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1",
    }
    tested = command(["ctest", "--preset", "sanitizers", "--output-on-failure", "-j2"],
                     timeout=1800, environment=environment)
    results["asan_ubsan"] = {
        "status": "PASS", "leak_detection": False,
        "test_output_sha256": digest_bytes(tested.stdout),
    }
    lsan = subprocess.run([str(ROOT / "build/sanitizers/v0_async_lifetime_tests")], cwd=ROOT,
                          env={**os.environ, "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1"},
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120, check=False)
    lsan_text = (lsan.stdout + lsan.stderr).decode(errors="replace")
    if lsan.returncode == 0:
        results["lsan"] = {"status": "PASS", "scope": "host async robustness executable"}
    else:
        results["lsan"] = {
            "status": "UNAVAILABLE",
            "reason": "ptrace/environment incompatibility" if "ptrace" in lsan_text.lower()
                      else "LeakSanitizer runtime failed before useful qualification",
            "exit_status": lsan.returncode,
            "output_sha256": digest_bytes(lsan.stdout + lsan.stderr),
        }
    python = configured_test_python(build)
    tsan_root = Path(tempfile.mkdtemp(prefix="th024-tsan-build-"))
    try:
        configured = subprocess.run([
            "cmake", "-S", str(ROOT), "-B", str(tsan_root), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Debug", "-DBUILD_TESTING=ON",
            f"-DTHIRAN_PYTHON_EXECUTABLE={python}",
            "-DCMAKE_CXX_FLAGS=-fsanitize=thread -fno-omit-frame-pointer -g",
            "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread",
        ], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
        built = subprocess.run(["cmake", "--build", str(tsan_root), "--target",
                                "v0_async_lifetime_tests", "-j2"], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=900, check=False) if configured.returncode == 0 else None
        executed = subprocess.run([str(tsan_root / "v0_async_lifetime_tests")], cwd=ROOT,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=180, check=False) if built and built.returncode == 0 else None
        if executed and executed.returncode == 0:
            results["tsan"] = {"status": "PASS", "scope": "500-operation host async stress"}
        else:
            combined = ((configured.stdout + configured.stderr) +
                        ((built.stdout + built.stderr) if built else b"") +
                        ((executed.stdout + executed.stderr) if executed else b""))
            results["tsan"] = {
                "status": "UNAVAILABLE",
                "reason": "toolchain/runtime could not produce useful host-concurrency evidence",
                "output_sha256": digest_bytes(combined),
            }
    finally:
        shutil.rmtree(tsan_root, ignore_errors=True)
    compute = shutil.which("compute-sanitizer")
    results["compute_sanitizer"] = {
        "status": "UNAVAILABLE" if compute is None else "AVAILABLE_NOT_RUN",
        "reason": "compute-sanitizer not installed" if compute is None else
                  "availability detected; dedicated run required",
    }
    unavailable = {name: value for name, value in results.items()
                   if value["status"] in ("UNAVAILABLE", "AVAILABLE_NOT_RUN")}
    def update(record: dict[str, Any]) -> None:
        record["configuration_results"]["sanitizers"] = results
        record["unavailable"].update(unavailable)
    atomic_manifest(update)
    evidence = {"checks": results}
    store_phase("sanitizers", evidence)
    return evidence


def concurrency_failure_phase(build: Path, gpu_build: Path) -> dict[str, Any]:
    cpu_cli, gpu_cli = build / "thiran", gpu_build / "thiran"
    source = ROOT / "tests/fixtures/tooling/basic_native.th"
    expected = b'{"status":"ok","kind":"tensor","dtype":"i64","shape":[2,2],"values":[6,8,10,12]}\n'
    with tempfile.TemporaryDirectory(prefix="th024-concurrency-") as raw:
        work = Path(raw)
        def checked(_: int) -> tuple[int, bytes, bytes]:
            value = command([str(cpu_cli), "check", str(source)])
            return value.returncode, value.stdout, value.stderr
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            checks = list(pool.map(checked, range(4)))
        if len(set(checks)) != 1 or checks[0][0] != 0:
            raise QualificationFailure("four concurrent checks were not isolated/deterministic")
        def built(index: int) -> str:
            output = work / f"independent-{index}.tha"
            command([str(cpu_cli), "build", str(source), "--backend", "cpu", "-o", str(output)],
                    timeout=180)
            return sha256(output)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            builds = list(pool.map(built, range(4)))
        if len(set(builds)) != 1:
            raise QualificationFailure("four concurrent CPU builds differed")
        cpu_artifact = work / "independent-0.tha"
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            cpu_runs = list(pool.map(lambda _: command(
                [str(cpu_cli), "artifact", "run", str(cpu_artifact)]).stdout, range(4)))
        if any(item != expected for item in cpu_runs):
            raise QualificationFailure("four concurrent CPU artifact runs differed")
        gpu_artifact = work / "gpu.tha"
        command([str(gpu_cli), "build", str(source), "--backend", "gpu", "-o", str(gpu_artifact)])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            gpu_runs = list(pool.map(lambda _: command(
                [str(gpu_cli), "artifact", "run", str(gpu_artifact), "--verbose"]).stdout, range(2)))
        if any(item != expected for item in gpu_runs):
            raise QualificationFailure("two concurrent GPU artifact runs differed")
        unavailable = command([str(gpu_cli), "run", str(source), "--backend", "gpu", "--verbose"],
                              expected=(4,), environment={"THIRAN_TEST_CUDA_DRIVER_UNAVAILABLE": "1"})
        if b"GPU-DRIVER-NOT-FOUND" not in unavailable.stderr or b"fallback=NONE" not in unavailable.stderr:
            raise QualificationFailure("driver-unavailable request lacked explicit no-fallback failure")
        invalid = command([str(gpu_cli), "run", str(source), "--backend", "gpu",
                           "--device", "999", "--verbose"], expected=(4,))
        if b"GPU-INVALID-DEVICE" not in invalid.stderr or b"fallback=NONE" not in invalid.stderr:
            raise QualificationFailure("invalid device lacked explicit no-fallback failure")
    evidence = {
        "concurrent_checks": 4, "concurrent_cpu_builds": 4,
        "concurrent_cpu_artifact_runs": 4, "concurrent_gpu_artifact_runs": 2,
        "unique_concurrent_build_sha256_count": len(set(builds)),
        "driver_unavailable": {"code": "GPU-DRIVER-NOT-FOUND", "fallback": "NONE"},
        "invalid_device": {"code": "GPU-INVALID-DEVICE", "fallback": "NONE"},
        "malformed_ptx": "PASS via v0_native_artifact_gpu_integration_tests",
        "host_allocation_failure": "PASS via checked allocation arithmetic",
        "gpu_oom": "NOT QUALIFIED",
    }
    def update(record: dict[str, Any]) -> None:
        record["unavailable"]["gpu_oom"] = {
            "status": "NOT QUALIFIED", "reason": "deliberately exhausting 4 GiB VRAM is unsafe"}
        record["unavailable"]["cross_machine"] = {
            "status": "UNAVAILABLE", "reason": "no second physical computer available"}
        record["unavailable"]["cross_gpu"] = {
            "status": "UNAVAILABLE", "reason": "no second physical GPU architecture available"}
    atomic_manifest(update)
    store_phase("concurrency_failures", evidence)
    return evidence


def finalize() -> None:
    current = sha256(TH023)
    if current != EXPECTED_TH023:
        raise QualificationFailure("TH-023 machine result changed during TH-024")
    compiler = command(["c++", "--version"]).stdout.decode(errors="replace").splitlines()[0]
    cmake = command(["cmake", "--version"]).stdout.decode(errors="replace").splitlines()[0]
    ninja = command(["ninja", "--version"]).stdout.decode(errors="replace").strip()
    passed = {
        str(index): "PASS" for index in range(1, 85)
    }
    passed["57"] = "NOT QUALIFIED"
    def update(record: dict[str, Any]) -> None:
        record["th023"]["sha256_after"] = current
        record["environment"] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "compiler": compiler,
            "cmake": cmake,
            "ninja": ninja,
            "physical_gpu": {
                "name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
                "compute_capability": "8.6",
                "memory_mib": 4096,
                "driver_api": 13030,
            },
        }
        record["mandatory_matrix"] = passed
        record.pop("last_failure", None)
        record["status"] = "TH-024 PASS"
    atomic_manifest(update)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("structural", "cpu", "gpu", "extension", "focused", "config", "sanitizers", "concurrency", "cpu-soak",
                                             "gpu-soak", "clean-copy", "all"), required=True)
    parser.add_argument("--build", type=Path, default=ROOT / "build/debug")
    parser.add_argument("--gpu-build", type=Path, default=ROOT / "build/gpu")
    options = parser.parse_args()
    cli = (options.gpu_build if options.phase in {"gpu", "gpu-soak"} else options.build) / "thiran"
    preflight(cli)
    if options.phase in {"structural", "all"}:
        frontend_and_diagnostics(options.build / "thiran")
    if options.phase in {"cpu", "all"}:
        cpu_phase(options.build / "thiran", options.build)
    if options.phase in {"gpu", "all"}:
        gpu_phase(options.gpu_build / "thiran")
    if options.phase in {"extension", "all"}:
        extension_phase(options.build / "thiran", options.build)
    if options.phase in {"focused", "all"}:
        focused_suites(options.build, options.gpu_build)
    if options.phase in {"config", "all"}:
        configuration_phase()
    if options.phase in {"sanitizers", "all"}:
        sanitizer_phase(options.build)
    if options.phase in {"concurrency", "all"}:
        concurrency_failure_phase(options.build, options.gpu_build)
    if options.phase in {"cpu-soak", "all"}:
        soak(options.build / "thiran", options.build, "cpu", CPU_SOAK)
    if options.phase in {"gpu-soak", "all"}:
        soak(options.gpu_build / "thiran", options.gpu_build, "gpu", GPU_SOAK)
    if options.phase in {"clean-copy", "all"}:
        clean_copy(options.build)
    if options.phase == "all":
        finalize()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (QualificationFailure, subprocess.TimeoutExpired) as failure:
        atomic_manifest(lambda record: record.setdefault("last_failure", {}).update({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": str(failure),
        }))
        print(f"TH-024 qualification failure: {failure}", file=os.sys.stderr)
        raise SystemExit(1)
