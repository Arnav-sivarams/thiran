#!/usr/bin/env python3
"""Capture the TH-023 environment without installing or modifying anything."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
from pathlib import Path


def command(arguments: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False, text=True, timeout=30)
        return {"command": arguments, "exit_status": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": arguments, "exit_status": None, "stdout": "", "stderr": str(error)}


def first_line(result: dict[str, object]) -> str | None:
    if result["exit_status"] == 0 and result["stdout"]:
        return str(result["stdout"]).splitlines()[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-python", type=Path, required=True)
    args = parser.parse_args()

    probes = {
        "uname": command(["uname", "-a"]),
        "lscpu": command(["lscpu"]),
        "memory": command(["free", "-b"]),
        "load_average": command(["cat", "/proc/loadavg"]),
        "taskset_available": command(["taskset", "--version"]),
        "taskset_cpu8": command(["taskset", "-c", "8", "true"]),
        "system_python": command(["python3", "--version"]),
        "system_numpy": command(["python3", "-c", "import numpy; print(numpy.__version__)"]),
        "system_torch": command(["python3", "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"]),
        "test_python": command([str(args.test_python), "--version"]),
        "test_numpy": command([str(args.test_python), "-c", "import numpy; print(numpy.__version__)"]),
        "test_torch": command([str(args.test_python), "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.get_num_threads())"]),
        "gxx": command(["g++", "--version"]),
        "clangxx": command(["clang++", "--version"]),
        "nvcc": command(["nvcc", "--version"]),
        "cmake": command(["cmake", "--version"]),
        "ninja": command(["ninja", "--version"]),
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,pstate", "--format=csv,noheader,nounits"]),
    }
    cpu_model = None
    for line in str(probes["lscpu"]["stdout"]).splitlines():
        if line.startswith("Model name:"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    memory_total = None
    memory_available = None
    for line in str(probes["memory"]["stdout"]).splitlines():
        if line.startswith("Mem:"):
            fields = line.split()
            memory_total = int(fields[1])
            memory_available = int(fields[6])
            break
    payload = {
        "phase": args.phase,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": "7722a58f260ca9993ab335ffc9dbbc22512371a0",
        "os": platform.platform(),
        "kernel": platform.release(),
        "wsl2": "microsoft-standard-WSL2" in platform.release(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "memory_total_bytes": memory_total,
        "memory_available_bytes": memory_available,
        "ac_power": "not exposed by WSL /sys/class/power_supply",
        "cmake_build_type": "Release",
        "thiran_cpu_artifact_flags": "-std=c++20 -fPIC -shared -Wl,--strip-debug",
        "cpp_baseline_flags": "-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -std=c++20",
        "affinity_policy": "taskset -c 8 for CPU warm microbenchmarks and framework baselines",
        "compiler": first_line(probes["gxx"]),
        "probes": probes,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"TH-023 environment {args.phase} captured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
