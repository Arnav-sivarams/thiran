#!/usr/bin/env python3
"""Dependency-free completeness validator for future performance records."""

import argparse
import json
import math
import sys
from pathlib import Path

CATEGORIES = {"KERNEL", "MODEL_PREFILL", "MODEL_STEP", "END_TO_END_REQUEST", "RENDER_FRAME"}
FIELDS = {
    "benchmark_id", "git_revision", "compiler_version_build", "workload",
    "model_weight_identity", "quality_correctness_condition", "cpu_gpu_model",
    "driver_runtime", "precision_numeric_profile", "batch", "shapes",
    "context_state_size", "warm_cold_status", "aot_jit_state", "cache_state",
    "memory_allocation_policy", "host_device_transfer_boundary", "synchronization_point",
    "warmup_count", "measured_iteration_count", "raw_samples", "median", "p95",
    "p99", "peak_memory", "failure_count", "category",
}


class ManifestFailure(Exception):
    pass


def validate_document(document):
    if type(document) is not dict or set(document) != {"schema", "claims"} or document["schema"] != "thiran-benchmark-manifest-1":
        raise ManifestFailure("invalid benchmark manifest envelope")
    claims = document["claims"]
    if type(claims) is not list or not claims:
        raise ManifestFailure("no benchmark records")
    seen = set()
    for number, claim in enumerate(claims, 1):
        if type(claim) is not dict:
            raise ManifestFailure(f"record {number} is not an object")
        missing = FIELDS - set(claim)
        extra = set(claim) - FIELDS
        if missing or extra:
            raise ManifestFailure(f"record {number}: missing {sorted(missing)}, unexpected {sorted(extra)}")
        benchmark_id = claim["benchmark_id"]
        if type(benchmark_id) is not str or not benchmark_id or benchmark_id in seen:
            raise ManifestFailure(f"record {number}: invalid or duplicate benchmark ID")
        seen.add(benchmark_id)
        if claim["category"] not in CATEGORIES:
            raise ManifestFailure(f"{benchmark_id}: unknown category")
        for field in ("warmup_count", "measured_iteration_count", "failure_count"):
            entry = claim[field]
            if entry is not None and (type(entry) is not int or entry < 0):
                raise ManifestFailure(f"{benchmark_id}: {field} must be nonnegative integer or null")
        samples = claim["raw_samples"]
        if samples is not None:
            if type(samples) is not list or any(type(sample) not in (int, float) or not math.isfinite(sample) or sample < 0 for sample in samples):
                raise ManifestFailure(f"{benchmark_id}: raw samples must be nonnegative numbers or null")
            if claim["measured_iteration_count"] != len(samples):
                raise ManifestFailure(f"{benchmark_id}: sample count does not match iterations")
    return len(claims)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate mandatory metadata; this does not attest benchmark results.")
    parser.add_argument("manifest", help="JSON benchmark manifest")
    args = parser.parse_args(argv)
    try:
        count = validate_document(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    except (ManifestFailure, OSError, json.JSONDecodeError) as failure:
        print(f"manifest error: {failure}", file=sys.stderr)
        return 2
    print(f"VALID {count} benchmark metadata record(s); measurements not attested")
    return 0


if __name__ == "__main__":
    sys.exit(main())
