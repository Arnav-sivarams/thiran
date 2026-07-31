# Thiran

Thiran is a C++20 research compiler prototype for a small tensor-dataflow language with deterministic Region planning and a PyTorch execution path.

## Current status

Thiran parses and links single-file or explicitly imported `.th` modules into one Graph IR, verifies and prepares that Graph, classifies nodes, forms connected Regions, builds an immutable `RegionPlan`, and can emit a Region-controlled Python artifact. The artifact executes one Python function per Region and passes boundary tensors explicitly. All `AOT`, `JIT`, and `FALLBACK` Regions currently use PyTorch; the strategy labels do not select native implementations.

## What is implemented

- Source locations, explicit modules/imports/exports, top-level tensor functions, deterministic compile-time Graph inlining, Graph IR ownership, verification, optimization, and shape inference.
- Deterministic node strategy classification and connected Region formation.
- Immutable RegionPlan construction, verification, inspection, and text emission.
- Region-controlled Python source emission with explicit Region inputs and outputs.
- Import-safe and directly executable tensor-bundle artifacts.
- Restricted `torch.load(..., weights_only=True)`, input shape validation, and atomic output writes.
- Deterministic diagnostics and numerical equivalence tests against the accepted whole-Graph PyTorch path.

## What is not implemented

Native AOT and JIT compilation, specialization and artifact caches, adaptive dispatch, working Triton execution, distributed execution, and demonstrated performance improvements are not implemented. Scheduler, partition, communication, BackendIR, Executor, and Triton structures exist, but they do not make Region strategies execute through distinct native backends.

## Architecture overview

```text
source
  -> Graph IR
  -> preparation and shape analysis
  -> strategy classification
  -> connected Regions
  -> immutable RegionPlan
  -> Region-controlled Python artifact
  -> PyTorch execution
```

The existing normal compiler path additionally emits `graph.dot` and the whole-Graph `generated.py`. See [Architecture](docs/ARCHITECTURE.md).

## Prerequisites

The validated contributor environment is Linux or WSL with:

- CMake 3.20 or newer;
- Ninja;
- a C++20 compiler (GCC 13 is locally validated);
- Git;
- for `BUILD_TESTING=ON`, Python 3 and a PyTorch installation supporting `torch.load(..., weights_only=True)`.

`BUILD_TESTING=OFF` builds the compiler without selecting Python or importing Torch. Test builds prefer `THIRAN_PYTHON_EXECUTABLE`, then `.venv/bin/python`, then a Python 3 interpreter found by CMake. A `.venv` is convenient, not mandatory.

For a test environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
```

## Quick build

```bash
cmake --preset debug
cmake --build --preset debug
```

The Debug executable is `build/debug/Thiran`.

## Install locally

Linux/WSL source installation uses no sudo and installs only lowercase `thiran`:

```bash
./scripts/install.sh --prefix "$HOME/.local"
"$HOME/.local/bin/thiran" --version
"$HOME/.local/bin/thiran" doctor
```

The default prefix is `$HOME/.local`. Add its `bin` directory to `PATH` yourself if needed. To uninstall the most recent installation made from `build/install`:

```bash
cmake --build build/install --target uninstall
```

No remote installer or package-manager distribution exists.

## Run all tests

```bash
ctest --preset debug --output-on-failure
```

Exactly fifteen CTest registrations are expected. See [Testing](docs/TESTING.md).

## Inspect a plan

```bash
./build/debug/Thiran --plan examples/cnn.th
```

## Emit a plan artifact

```bash
./build/debug/Thiran --emit-plan examples/cnn.th /tmp/cnn-region-plan.txt
```

## Emit a Region executor

```bash
./build/debug/Thiran --emit-region-executor examples/cnn.th /tmp/cnn-executor.py
```

## Describe a Region executor

```bash
.venv/bin/python /tmp/cnn-executor.py --describe
```

Use the Python interpreter selected for the test build if it is not `.venv/bin/python`.

## Execute a Region executor

Create a deterministic input bundle:

```bash
.venv/bin/python -c "import torch; torch.save({'Image': torch.arange(64, dtype=torch.float32).reshape(1,1,8,8), 'Filter': torch.full((2,1,3,3), 0.25), 'Classifier': torch.arange(72, dtype=torch.float32).reshape(18,4) / 72}, '/tmp/cnn-inputs.pt')"
```

Run and safely load the result:

```bash
.venv/bin/python /tmp/cnn-executor.py --run /tmp/cnn-inputs.pt /tmp/cnn-outputs.pt
.venv/bin/python -c "import torch; outputs=torch.load('/tmp/cnn-outputs.pt', map_location='cpu', weights_only=True); print(tuple(outputs))"
```

The output key is `O`. The runtime rejects malformed bundles and never falls back to unrestricted loading.

## Repository layout

- `include/`, `src/`: compiler and Region runtime source.
- `tests/region/`: zero-dependency C++ test harnesses.
- `tests/integration/`: CMake-driven CLI and numerical tests.
- `tests/fixtures/`: source fixtures.
- `examples/`: sample `.th` programs.
- `docs/`: architecture, CLI, testing, operation, and status guides.
- `docs/FUNCTIONS.md`: immutable tensor-function syntax and deterministic inlining rules.
- `scripts/validate.sh`: local contributor validation.
- `.github/workflows/ci.yml`: continuous-integration mirror.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [CLI reference](docs/CLI.md)
- [Testing and validation](docs/TESTING.md)
- [Adding an operation](docs/ADDING_AN_OPERATION.md)
- [Project status](docs/PROJECT_STATUS.md)
- [Modules](docs/MODULES.md)

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing compiler semantics.

## Known limitations

The source language and operation set are limited. Current Region execution is correctness-oriented PyTorch, not native AOT/JIT execution. No performance or recompilation-locality result is established. Linux/WSL is the validated environment.

## License status

The repository contains an empty `LICENSE` file; redistribution and reuse terms have not yet been selected. Selecting a license is an owner decision and remains a release blocker.
