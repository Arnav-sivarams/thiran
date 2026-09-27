#!/usr/bin/env python3
"""Deterministic, inexpensive Thiran 0.1.0 release invariant checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


VERSION = "0.1.0"
BASE_COMMIT = "79f8fe5096d70100b9793b0d5018048f57a18c92"
TH023_SHA256 = "04f7c9d26d4dd749d308328dd69bd5f83f0c4c3786fe380de65a7d4fc0858e31"
TH024_SHA256 = "505b82dd929ebf5bf93dbbdf392ac4d5c4ffebeb56524bfbf6d16407daa7217f"
GRAPH_DIRECTORY = "docs/assets/benchmarks/v0.1"
GRAPH_FILES = (
    "cpu-elementwise-median.svg",
    "gpu-transfer-inclusive-median.svg",
    "fusion-effect.svg",
    "source-vs-artifact.svg",
    "extension-performance.svg",
    "deployment-overhead.svg",
)
GRAPH_SOURCE_COUNTS = {
    "cpu-elementwise-median.svg": 18,
    "gpu-transfer-inclusive-median.svg": 9,
    "fusion-effect.svg": 0,
    "source-vs-artifact.svg": 4,
    "extension-performance.svg": 4,
    "deployment-overhead.svg": 4,
}

ABI_SOURCES = {
    "artifact_format_version": ("include/artifact/v0/NativeArtifacts.hpp", "artifactFormatVersion"),
    "compiler_artifact_abi": ("include/artifact/v0/NativeArtifacts.hpp", "compilerArtifactAbiVersion"),
    "native_runtime_abi": ("include/artifact/v0/NativeArtifacts.hpp", "nativeRuntimeAbiVersion"),
    "extension_abi": ("include/extension/v0/Extension.hpp", "extensionAbiVersion"),
    "model_bundle_format_version": ("include/model/v0/ModelBundle.hpp", "modelBundleFormatVersion"),
    "model_runtime_abi": ("include/model/v0/ModelBundle.hpp", "modelRuntimeAbiVersion"),
    "training_checkpoint_format_version": ("include/training/v0/Checkpoint.hpp", "checkpointFormatVersion"),
    "training_checkpoint_abi": ("include/training/v0/Checkpoint.hpp", "trainingCheckpointAbiVersion"),
}

REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "docs/RELEASE_V0_1.md",
    "docs/ABI_V0_1.md",
    "docs/CLI.md",
    "release/THIRAN-0.1.0.json",
    "examples/basic_tensor.th",
    "examples/structured_scan.th",
    "examples/research_extension/square_linear.th",
    "examples/research_extension/square_linear_extension.cpp",
    "scripts/release/generate_benchmark_graphs.py",
    *(f"{GRAPH_DIRECTORY}/{name}" for name in GRAPH_FILES),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def read(root: pathlib.Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_constant(root: pathlib.Path, relative: str, name: str) -> int:
    match = re.search(
        rf"\b{name}\s*=\s*([0-9]+)\s*;", read(root, relative)
    )
    if not match:
        fail(f"cannot read {name} from {relative}")
    return int(match.group(1))


def tracked_hygiene(root: pathlib.Path) -> None:
    git = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if git.returncode != 0:
        return
    forbidden_names = {"CMakeCache.txt", "core", "core.dump"}
    forbidden_suffixes = {".o", ".obj", ".so", ".a", ".exe", ".tha", ".thm", ".thc", ".pyc"}
    bad: list[str] = []
    for raw in git.stdout.split(b"\0"):
        if not raw:
            continue
        path = pathlib.PurePosixPath(raw.decode("utf-8"))
        if path.name in forbidden_names or path.suffix in forbidden_suffixes or "__pycache__" in path.parts:
            bad.append(str(path))
    if bad:
        fail("tracked build/cache artifacts: " + ", ".join(sorted(bad)))


def validate_benchmark_graphs(root: pathlib.Path, readme: str) -> None:
    evidence_path = root / "benchmarks/results/TH-023-machine.json"
    if sha256(evidence_path) != TH023_SHA256:
        fail("benchmark graph source does not have the accepted TH-023 hash")
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        fail("TH-023 benchmark records are missing")

    index: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for record in records:
        key = tuple(str(record.get(field)) for field in (
            "benchmark_name", "implementation", "backend", "size", "timing_scope"
        ))
        index.setdefault(key, []).append(record)

    graph_root = root / GRAPH_DIRECTORY
    forbidden_path = re.compile(
        r"(?:/home/[A-Za-z0-9._-]+/|[A-Za-z]:[\\\\/]Users[\\\\/])",
        re.IGNORECASE,
    )
    generator_text = read(root, "scripts/release/generate_benchmark_graphs.py")
    if forbidden_path.search(generator_text):
        fail("benchmark graph generator contains an absolute personal path")

    expected_fusion: dict[str, float] = {}
    size_labels = ((1024, "small · 1,024"), (1048576, "medium · 1,048,576"),
                   (16777216, "large · 16,777,216"))
    for size, label in size_labels:
        def fusion_record(implementation: str) -> dict:
            key = ("elementwise_chain", implementation, "cpu", str(size),
                   "warm_in_process_host_wall")
            matches = index.get(key, [])
            if len(matches) != 1:
                fail(f"fusion source row is not unique: {key!r}")
            return matches[0]
        expected_fusion[label] = (
            float(fusion_record("thiran_unfused")["median_ns"])
            / float(fusion_record("thiran_fused")["median_ns"])
        )

    for name in GRAPH_FILES:
        relative = f"{GRAPH_DIRECTORY}/{name}"
        if f"]({relative})" not in readme:
            fail(f"README does not reference benchmark graph: {relative}")
        text = read(root, relative)
        if forbidden_path.search(text):
            fail(f"benchmark graph contains an absolute personal path: {relative}")
        try:
            document = ET.fromstring(text)
        except ET.ParseError as error:
            fail(f"invalid SVG {relative}: {error}")
        namespace = "{http://www.w3.org/2000/svg}"
        if document.tag != f"{namespace}svg" or document.get("viewBox") != "0 0 960 560":
            fail(f"benchmark graph has an unexpected SVG root: {relative}")
        if document.find(f"{namespace}title") is None or document.find(f"{namespace}desc") is None:
            fail(f"benchmark graph lacks accessible title/description: {relative}")

        source_markers = [node for node in document.iter(f"{namespace}circle")
                          if node.get("data-median-ns") is not None]
        if len(source_markers) != GRAPH_SOURCE_COUNTS[name]:
            fail(f"benchmark graph source-marker count is wrong: {relative}")
        for marker in source_markers:
            key = tuple(marker.get(attribute, "") for attribute in (
                "data-benchmark", "data-implementation", "data-backend", "data-size",
                "data-timing-scope"
            ))
            matches = index.get(key, [])
            if len(matches) != 1:
                fail(f"plotted TH-023 row is not unique in {relative}: {key!r}")
            median = matches[0].get("median_ns")
            if str(median) != marker.get("data-median-ns"):
                fail(f"plotted median differs from TH-023 in {relative}: {key!r}")
            rendered = f"{float(median) / 1_000_000.0:.6g}"
            if marker.get("data-rendered-ms") != rendered:
                fail(f"rendered precision differs from TH-023 in {relative}: {key!r}")

        derived = [node for node in document.iter(f"{namespace}circle")
                   if node.get("data-derived-value") is not None]
        if name == "fusion-effect.svg":
            if len(derived) != 3:
                fail("fusion graph does not contain exactly three derived values")
            for marker in derived:
                label = marker.get("data-derived-label", "")
                if label not in expected_fusion:
                    fail(f"unexpected fusion graph label: {label!r}")
                value = expected_fusion[label]
                if marker.get("data-derived-value") != f"{value:.12f}":
                    fail(f"fusion graph value differs from TH-023: {label}")
                if marker.get("data-rendered-value") != f"{value:.6g}":
                    fail(f"fusion graph displayed precision differs from TH-023: {label}")
        elif derived:
            fail(f"unexpected derived values in {relative}")

    generator = root / "scripts/release/generate_benchmark_graphs.py"
    with tempfile.TemporaryDirectory(prefix="thiran-v0.1-graphs-") as temporary:
        generated = pathlib.Path(temporary)
        completed = subprocess.run(
            [sys.executable, "-B", str(generator), "--input", str(evidence_path),
             "--output-dir", str(generated), "--quiet"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            fail(f"benchmark graph regeneration failed: {completed.stderr.strip()}")
        generated_names = tuple(sorted(path.name for path in generated.glob("*.svg")))
        if generated_names != tuple(sorted(GRAPH_FILES)):
            fail(f"benchmark graph generator emitted unexpected files: {generated_names!r}")
        for name in GRAPH_FILES:
            if (generated / name).read_bytes() != (graph_root / name).read_bytes():
                fail(f"benchmark graph is not deterministic/current: {name}")


def validate(root: pathlib.Path, cli: pathlib.Path | None) -> None:
    for relative in REQUIRED_PATHS:
        if not (root / relative).is_file():
            fail(f"required release path missing: {relative}")

    if (root / "VERSION").read_bytes() != b"0.1.0\n":
        fail("VERSION is not exactly 0.1.0 plus one newline")

    cmake = read(root, "CMakeLists.txt")
    if 'file(READ "${CMAKE_SOURCE_DIR}/VERSION" THIRAN_VERSION_FILE)' not in cmake:
        fail("CMake no longer reads the authoritative VERSION file")
    if 'project(Thiran VERSION "${THIRAN_VERSION_MAJOR}.${THIRAN_VERSION_MINOR}.${THIRAN_VERSION_PATCH}")' not in cmake:
        fail("CMake project version is not derived from VERSION")

    manifest = json.loads(read(root, "release/THIRAN-0.1.0.json"))
    if manifest.get("product_name") != "Thiran" or manifest.get("product_version") != VERSION:
        fail("release manifest product identity disagrees with VERSION")
    if manifest.get("language_generation") != "V0":
        fail("release manifest language generation is not V0")
    if manifest.get("release_base_commit") != BASE_COMMIT:
        fail("release manifest base commit is incorrect")
    if manifest.get("release_commit") is not None:
        fail("uncommitted TH-025 manifest must not fabricate a release commit")
    if manifest.get("release_qualification_status") != "TH-025 PASS":
        fail("release manifest is not frozen at TH-025 PASS")

    actual_abi = {
        key: source_constant(root, relative, constant)
        for key, (relative, constant) in ABI_SOURCES.items()
    }
    if manifest.get("abi") != actual_abi:
        fail(f"manifest ABI inventory disagrees with source: {actual_abi!r}")

    evidence = manifest.get("accepted_evidence", {})
    checks = (
        ("th023", TH023_SHA256),
        ("th024", TH024_SHA256),
    )
    for label, expected in checks:
        path = evidence.get(f"{label}_path")
        claimed = evidence.get(f"{label}_sha256")
        if not isinstance(path, str) or claimed != expected:
            fail(f"manifest {label.upper()} evidence metadata is incorrect")
        actual = sha256(root / path)
        if actual != expected:
            fail(f"protected {label.upper()} evidence hash changed: {actual}")

    qualification_path = root / "release/results/TH-025-machine.json"
    if qualification_path.exists():
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        if qualification.get("status") != "TH-025 PASS":
            fail("TH-025 qualification result is not PASS")
        if qualification.get("release", {}).get("product_version") != VERSION:
            fail("TH-025 qualification result version disagrees with VERSION")
        qualified_evidence = qualification.get("protected_evidence", {})
        if qualified_evidence.get("th023_sha256") != TH023_SHA256 or qualified_evidence.get("th024_sha256") != TH024_SHA256:
            fail("TH-025 qualification result evidence hashes disagree")

    release_doc = read(root, "docs/RELEASE_V0_1.md")
    readme = read(root, "README.md")
    cli_doc = read(root, "docs/CLI.md")
    validate_benchmark_graphs(root, readme)
    for label, text in (("release scope", release_doc), ("README", readme), ("CLI docs", cli_doc)):
        if "0.1.0" not in text:
            fail(f"{label} does not identify version 0.1.0")
    for required in ("SUPPORTED / QUALIFIED", "EXPERIMENTAL", "DEFERRED", "NOT SUPPORTED"):
        if required not in release_doc:
            fail(f"release scope omits status category {required}")
    normalized_release_doc = " ".join(release_doc.split()).lower()
    for limitation in (
        "scalar/non-vectorized",
        "Cross-machine",
        "RTX 3050 Ti",
        "no malicious-plugin sandbox",
        "no single-file portable executable",
        "Scan differentiation",
    ):
        if limitation.lower() not in normalized_release_doc:
            fail(f"release scope omits limitation marker: {limitation}")

    license_text = read(root, "LICENSE")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        fail("LICENSE is not recognizable as Apache-2.0")

    if cli is not None:
        completed = subprocess.run(
            [str(cli), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or completed.stdout != "Thiran 0.1.0\n" or completed.stderr:
            fail("CLI version is not exactly 'Thiran 0.1.0'")

    tracked_hygiene(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--cli", type=pathlib.Path)
    args = parser.parse_args()
    try:
        validate(args.source_root.resolve(), args.cli.resolve() if args.cli else None)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"TH025ReleaseStructure FAIL: {error}", file=sys.stderr)
        return 1
    print("TH025ReleaseStructure PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
