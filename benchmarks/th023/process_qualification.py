#!/usr/bin/env python3
"""Fresh-process/build qualification for the production TH-022 CLI."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

COMMIT = "7722a58f260ca9993ab335ffc9dbbc22512371a0"
SAMPLES = 10


def fnv(data: bytes) -> str:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"fnv1a64:{value:016x}"


def aggregates(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)

    def percentile(p: float) -> float:
        position = p * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    mean = statistics.fmean(samples)
    return {
        "median_ns": percentile(0.5),
        "mean_ns": mean,
        "stddev_ns": math.sqrt(statistics.fmean((sample - mean) ** 2 for sample in samples)),
        "p10_ns": percentile(0.1),
        "p90_ns": percentile(0.9),
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
    }


def invoke(command: list[str]) -> tuple[float, subprocess.CompletedProcess[bytes]]:
    begin = time.perf_counter_ns()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = float(time.perf_counter_ns() - begin)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}\n"
            f"stdout={result.stdout.decode(errors='replace')}\n"
            f"stderr={result.stderr.decode(errors='replace')}"
        )
    return elapsed, result


def make_record(name: str, implementation: str, backend: str, scope: str,
                samples: list[float], output_digest: str,
                counters: dict[str, int] | None = None,
                labels: dict[str, str] | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "benchmark_name": name,
        "implementation": implementation,
        "backend": backend,
        "size": 32 if name != "reference_model_deployment" else 1,
        "build_type": "Release",
        "warmup_count": 0,
        "sample_count": len(samples),
        "batch_count": 1,
        "timing_scope": scope,
        "samples_ns": samples,
        "input_digest": "fixture:benchmarks/th023/cli_workload.th",
        "output_digest": output_digest,
        "commit": COMMIT,
        "counters": counters or {},
        "labels": labels or {},
    }
    record.update(aggregates(samples))
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-python", type=Path)
    args = parser.parse_args()
    cli = str(args.cli.resolve())
    source = str(args.source.resolve())
    model_bundle = str(args.model_bundle.resolve())
    backend = args.backend
    records: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix=f"th023-process-{backend}-") as temporary_text:
        temporary = Path(temporary_text)
        _, checked = invoke([cli, "check", source])
        if checked.stdout or checked.stderr:
            raise RuntimeError("production check unexpectedly emitted output")

        build_samples: list[float] = []
        artifacts: list[Path] = []
        for sample in range(SAMPLES):
            artifact = temporary / f"program-{sample}.tha"
            elapsed, result = invoke([cli, "build", source, "--backend", backend,
                                      "-o", str(artifact)])
            if result.stdout or result.stderr:
                raise RuntimeError("production build unexpectedly emitted output")
            build_samples.append(elapsed)
            artifacts.append(artifact)
        artifact = artifacts[-1]
        _, inspected = invoke([cli, "artifact", "inspect", str(artifact)])
        inspection = {}
        for line in inspected.stdout.decode().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                inspection[key] = value
        payload_size = int(inspection["payload_size"])
        artifact_size = artifact.stat().st_size
        artifact_digest = fnv(artifact.read_bytes())
        records.append(make_record(
            "source_to_artifact_build", "thiran_cli_build", backend,
            "fresh_process_source_check_lower_codegen_artifact_write_host_wall",
            build_samples, artifact_digest,
            {"artifact_bytes": artifact_size, "payload_bytes": payload_size},
            {"payload_kind": inspection["payload_kind"],
             "planning_fusion": inspection["planning.fusion"],
             "planning_reuse": inspection["planning.reuse"]},
        ))

        source_samples: list[float] = []
        source_digest = None
        for _ in range(SAMPLES):
            elapsed, result = invoke([cli, "run", source, "--backend", backend])
            digest = fnv(result.stdout)
            if source_digest is None:
                source_digest = digest
            elif digest != source_digest:
                raise RuntimeError("source-run output was not deterministic")
            source_samples.append(elapsed)
        records.append(make_record(
            "source_run", "thiran_cli_run", backend,
            "fresh_process_parse_check_lower_build_load_execute_materialize_print_host_wall",
            source_samples, source_digest or "",
        ))

        artifact_samples: list[float] = []
        run_digest = None
        for _ in range(SAMPLES):
            elapsed, result = invoke([cli, "artifact", "run", str(artifact)])
            digest = fnv(result.stdout)
            if run_digest is None:
                run_digest = digest
            elif digest != run_digest:
                raise RuntimeError("artifact-run output was not deterministic")
            artifact_samples.append(elapsed)
        if run_digest != source_digest:
            raise RuntimeError("source-run and artifact-run outputs differ")
        records.append(make_record(
            "artifact_run", "thiran_cli_artifact_run", backend,
            "fresh_process_artifact_load_backend_load_execute_materialize_print_host_wall",
            artifact_samples, run_digest or "",
            {"artifact_bytes": artifact_size, "payload_bytes": payload_size},
        ))

        model_samples: list[float] = []
        model_digest = None
        for _ in range(SAMPLES):
            elapsed, result = invoke([cli, "model", "run", model_bundle,
                                      "--backend", backend, "--input", "2"])
            digest = fnv(result.stdout)
            if model_digest is None:
                model_digest = digest
            elif digest != model_digest:
                raise RuntimeError("model fresh-process output was not deterministic")
            model_samples.append(elapsed)
        records.append(make_record(
            "reference_model_deployment", "thiran_cli_model_run", backend,
            "deployment_fresh_process_bundle_load_backend_load_execute_materialize_print_host_wall",
            model_samples, model_digest or "",
            {"model_bundle_bytes": args.model_bundle.stat().st_size},
            {"workload_class": "deployment overhead workload"},
        ))

    if backend == "cpu" and args.test_python:
        # Do not resolve a virtual-environment interpreter symlink: Python uses
        # the invoked path to locate pyvenv.cfg and its installed packages.
        python = str(args.test_python.absolute())
        for name, statement in (("numpy", "import numpy"), ("pytorch", "import torch")):
            samples = []
            output_digest = None
            for _ in range(SAMPLES):
                elapsed, result = invoke([python, "-c", statement])
                current = fnv(result.stdout + result.stderr)
                output_digest = output_digest or current
                samples.append(elapsed)
            records.append(make_record(
                "framework_process_startup", f"python_import_{name}", "cpu",
                "fresh_python_process_import_host_wall", samples, output_digest or "",
                labels={"python": python},
            ))

    payload = {
        "schema_version": 1,
        "commit": COMMIT,
        "qualification_backend": backend,
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"TH-023 process {backend} PASS records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
