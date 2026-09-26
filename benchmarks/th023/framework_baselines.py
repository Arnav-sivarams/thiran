#!/usr/bin/env python3
"""Optional installed NumPy/PyTorch baselines for TH-023; no downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

COMMIT = "7722a58f260ca9993ab335ffc9dbbc22512371a0"
SIZES = (("small", 1 << 10, 256), ("medium", 1 << 20, 8), ("large", 1 << 24, 1))
SAMPLES = 30


def inputs(count: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(count, dtype=np.int64)
    x = (((index * 17) % 1009) - 504).astype(np.float32) / np.float32(37)
    y = ((((index * 29) + 11) % 1013) - 506).astype(np.float32) / np.float32(41)
    return x, y


def numpy_chain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = x + y
    b = -a
    c = b * y
    d = c - x
    return d * y


def torch_chain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    a = x + y
    b = -a
    c = b * y
    d = c - x
    return d * y


def digest(values: np.ndarray) -> str:
    # Match the qualification executable's FNV-1a over little-endian f32 bytes.
    data = np.asarray(values, dtype="<f4").tobytes(order="C")
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
    variance = statistics.fmean((sample - mean) ** 2 for sample in samples)
    return {
        "median_ns": percentile(0.5),
        "mean_ns": mean,
        "stddev_ns": math.sqrt(variance),
        "p10_ns": percentile(0.1),
        "p90_ns": percentile(0.9),
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
    }


def record(implementation: str, backend: str, label: str, count: int, batch: int,
           warmups: int, scope: str, samples: list[float], input_hash: str,
           output_hash: str, labels: dict[str, str] | None = None,
           counters: dict[str, int] | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "benchmark_name": "elementwise_chain",
        "implementation": implementation,
        "backend": backend,
        "size": count,
        "build_type": "installed-framework",
        "warmup_count": warmups,
        "sample_count": len(samples),
        "batch_count": batch,
        "timing_scope": scope,
        "samples_ns": samples,
        "input_digest": input_hash,
        "output_digest": output_hash,
        "commit": COMMIT,
        "labels": {"size_label": label, **(labels or {})},
        "counters": counters or {},
    }
    result.update(aggregates(samples))
    return result


def measure(function, batch: int, synchronize=None):
    if synchronize:
        synchronize()
    begin = time.perf_counter_ns()
    output = None
    for _ in range(batch):
        output = function()
    if synchronize:
        synchronize()
    elapsed = time.perf_counter_ns() - begin
    return elapsed / batch, output


def cpu_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    discovered_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    for label, count, batch in SIZES:
        x_numpy, y_numpy = inputs(count)
        expected = numpy_chain(x_numpy, y_numpy)
        x_torch = torch.from_numpy(x_numpy)
        y_torch = torch.from_numpy(y_numpy)
        actual = torch_chain(x_torch, y_torch).numpy()
        if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
            raise RuntimeError(f"PyTorch CPU output mismatch at {label}")
        input_hash = digest(np.concatenate((x_numpy, y_numpy)))
        output_hash = digest(expected)
        for _ in range(5):
            measure(lambda: numpy_chain(x_numpy, y_numpy), batch)
            measure(lambda: torch_chain(x_torch, y_torch), batch)
        numpy_samples: list[float] = []
        torch_samples: list[float] = []
        for block in range(SAMPLES // 2):
            for which in (0, 1, 1, 0):
                if which == 0:
                    elapsed, output = measure(lambda: numpy_chain(x_numpy, y_numpy), batch)
                    if digest(output) != output_hash:
                        raise RuntimeError("NumPy timed output digest mismatch")
                    numpy_samples.append(elapsed)
                else:
                    elapsed, output = measure(lambda: torch_chain(x_torch, y_torch), batch)
                    if digest(output.numpy()) != output_hash:
                        raise RuntimeError("PyTorch CPU timed output digest mismatch")
                    torch_samples.append(elapsed)
        records.append(record("numpy_eager", "cpu", label, count, batch, 5,
                              "warm_in_process_host_wall", numpy_samples,
                              input_hash, output_hash,
                              {"numpy_version": np.__version__, "dtype": "float32"}))
        records.append(record("pytorch_eager_single_thread", "cpu", label, count, batch, 5,
                              "warm_in_process_host_wall", torch_samples,
                              input_hash, output_hash,
                              {"torch_version": torch.__version__,
                               "torch_threads": "1",
                               "discovered_default_threads": str(discovered_threads),
                               "dtype": "torch.float32"}))
    return records


def gpu_records() -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("installed PyTorch cannot access CUDA")
    records: list[dict[str, object]] = []
    for label, count, batch in SIZES:
        x_numpy, y_numpy = inputs(count)
        expected = numpy_chain(x_numpy, y_numpy)
        input_hash = digest(np.concatenate((x_numpy, y_numpy)))
        output_hash = digest(expected)
        x_host = torch.from_numpy(x_numpy.copy())
        y_host = torch.from_numpy(y_numpy.copy())
        x_device = x_host.to("cuda")
        y_device = y_host.to("cuda")
        torch.cuda.synchronize()
        actual = torch_chain(x_device, y_device).cpu().numpy()
        if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
            raise RuntimeError(f"PyTorch CUDA output mismatch at {label}")

        def resident():
            return torch_chain(x_device, y_device)

        def inclusive():
            return torch_chain(x_host.to("cuda"), y_host.to("cuda")).cpu()

        for _ in range(10):
            measure(resident, batch, torch.cuda.synchronize)
            measure(inclusive, batch, torch.cuda.synchronize)
        resident_samples: list[float] = []
        inclusive_samples: list[float] = []
        for block in range(SAMPLES // 2):
            for which in (0, 1, 1, 0):
                if which == 0:
                    elapsed, output = measure(resident, batch, torch.cuda.synchronize)
                    if digest(output.cpu().numpy()) != output_hash:
                        raise RuntimeError("PyTorch CUDA resident digest mismatch")
                    resident_samples.append(elapsed)
                else:
                    elapsed, output = measure(inclusive, batch, torch.cuda.synchronize)
                    if digest(output.numpy()) != output_hash:
                        raise RuntimeError("PyTorch CUDA inclusive digest mismatch")
                    inclusive_samples.append(elapsed)
        common = {"torch_version": torch.__version__, "torch_cuda": str(torch.version.cuda),
                  "device": torch.cuda.get_device_name(0), "dtype": "torch.float32"}
        records.append(record("pytorch_eager_device_resident", "gpu", label, count, batch, 10,
                              "device_resident_synchronized_host_wall", resident_samples,
                              input_hash, output_hash, common,
                              {"h2d_count": 0, "d2h_count": 0}))
        records.append(record("pytorch_eager_transfer_inclusive", "gpu", label, count, batch, 10,
                              "transfer_inclusive_synchronized_host_wall", inclusive_samples,
                              input_hash, output_hash, common,
                              {"h2d_count": 2, "h2d_bytes": count * 8,
                               "d2h_count": 1, "d2h_bytes": count * 4}))
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = cpu_records() if args.backend == "cpu" else gpu_records()
    payload = {
        "schema_version": 1,
        "commit": COMMIT,
        "qualification_backend": args.backend,
        "python": tuple(map(int, __import__("sys").version_info[:3])),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"TH-023 framework {args.backend} PASS records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
