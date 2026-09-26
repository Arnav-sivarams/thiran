#!/usr/bin/env python3
"""Validate and merge the machine-local TH-023 raw evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

COMMIT = "7722a58f260ca9993ab335ffc9dbbc22512371a0"
REQUIRED_FIELDS = {
    "benchmark_name", "implementation", "backend", "size", "build_type",
    "warmup_count", "sample_count", "batch_count", "timing_scope", "samples_ns",
    "median_ns", "mean_ns", "stddev_ns", "p10_ns", "p90_ns", "min_ns", "max_ns",
    "output_digest", "commit", "counters", "labels",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-before", type=Path, required=True)
    parser.add_argument("--environment-after", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = json.loads(args.environment_before.read_text(encoding="utf-8"))
    after = json.loads(args.environment_after.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    sources = []
    for path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("commit") != COMMIT:
            raise RuntimeError(f"commit mismatch in {path}")
        sources.append(str(path))
        for record in payload["records"]:
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise RuntimeError(f"{path}: record missing {sorted(missing)}")
            if record["commit"] != COMMIT:
                raise RuntimeError(f"{path}: record commit mismatch")
            if record["sample_count"] != len(record["samples_ns"]):
                raise RuntimeError(f"{path}: sample count mismatch")
            if record["benchmark_name"] in {
                "elementwise_chain", "extension_chain", "multiple_consumer_materialization",
            } and record["sample_count"] != 30:
                raise RuntimeError(f"{path}: primary benchmark does not have 30 samples")
            if record["benchmark_name"] in {
                "source_to_artifact_build", "source_run", "artifact_run",
                "framework_process_startup",
            } and record["sample_count"] != 10:
                raise RuntimeError(f"{path}: process benchmark does not have 10 samples")
            record["machine_metadata"] = {
                "cpu_model": before["cpu_model"],
                "logical_cpu_count": before["logical_cpu_count"],
                "memory_total_bytes": before["memory_total_bytes"],
                "kernel": before["kernel"],
                "wsl2": before["wsl2"],
                "compiler": before["compiler"],
            }
            counters = record["counters"]
            if ("materialized_intermediates" in counters and
                    record["benchmark_name"] in {
                        "elementwise_chain", "extension_chain",
                        "multiple_consumer_materialization",
                    }):
                counters["logical_temporary_bytes_static_plan"] = counters.pop(
                    "logical_temporary_bytes", 0
                )
                counters["logical_temporary_bytes"] = (
                    counters["materialized_intermediates"] * record["size"] * 4
                )
                record["labels"]["logical_temporary_bytes_basis"] = (
                    "runtime concrete f32 size; static dynamic-shape plan byte count is unknown"
                )
            records.append(record)

    def present(name: str, backend: str | None = None) -> bool:
        return any(record["benchmark_name"] == name and
                   (backend is None or record["backend"] == backend) for record in records)

    for name, backend in (
        ("elementwise_chain", "cpu"), ("elementwise_chain", "gpu"),
        ("extension_chain", "cpu"), ("extension_chain", "gpu"),
        ("multiple_consumer_materialization", "cpu"),
        ("multiple_consumer_materialization", "gpu"),
        ("source_to_artifact_build", "cpu"), ("source_to_artifact_build", "gpu"),
        ("source_run", "cpu"), ("source_run", "gpu"),
        ("artifact_run", "cpu"), ("artifact_run", "gpu"),
        ("reference_model_deployment", "cpu"),
        ("reference_model_deployment", "gpu"),
    ):
        if not present(name, backend):
            raise RuntimeError(f"missing required record {name}/{backend}")

    output = {
        "schema_version": 1,
        "checkpoint": "TH-023",
        "commit": COMMIT,
        "methodology": "docs/spec/PERFORMANCE_QUALIFICATION_V0.md",
        "environment_before": before,
        "environment_after": after,
        "source_files": sources,
        "availability": {
            "system_numpy": "unavailable",
            "system_pytorch": "unavailable",
            "test_environment_numpy": "2.5.1",
            "test_environment_pytorch": "2.13.0+cu130",
            "pytorch_cuda_physical": True,
            "nvcc": "unavailable",
            "cuda_cpp_baseline": "unavailable",
            "matlab": "not qualified",
            "rust": "not qualified",
            "gpu_compute_oriented_thiran": "unavailable: production API has no timing-enabled kernel event",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent,
            prefix=f".{args.output.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(output, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, args.output)
        temporary_name = None
        directory_fd = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    print(f"TH-023 merged PASS records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
