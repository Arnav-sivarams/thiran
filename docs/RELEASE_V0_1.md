# Thiran 0.1.0 release scope

**Release class:** V0.1 experimental technical release
**Language generation:** V0
**Qualification base:** `79f8fe5096d70100b9793b0d5018048f57a18c92`

This document is the public support contract for Thiran 0.1.0. “Supported”
means the bounded behavior is implemented and qualified in the environment
below; it does not mean a stable 1.0 compatibility promise or broad ecosystem
maturity.

## Status vocabulary

- **SUPPORTED / QUALIFIED:** implemented and included in V0.1 qualification.
- **EXPERIMENTAL:** implemented research surface with a deliberately narrow or
  unstable contract.
- **DEFERRED:** designed or identified, but intentionally outside V0.1.
- **NOT SUPPORTED:** no V0.1 claim or implementation.

## Support matrix

| Area | Status | V0.1 boundary |
|---|---|---|
| Source scalars and tensors | SUPPORTED / QUALIFIED | Implemented structured subset uses `bool`, `i64`, `f32`, rank-0/1/2 typed tensors where admitted, tensor literals through rank 2, and checked shapes/numerics. Broader constitution types are design targets, not blanket implementation claims. |
| Indexing and slicing | SUPPORTED / QUALIFIED | Checked indexing and positive-step read views in the accepted subset; no general indexed assignment. |
| Functions and control | SUPPORTED / QUALIFIED | Typed functions, tuples, `if`, range `for`, `while`, `break`, `continue`, lexical mutation/rebinding, modules under existing frontend rules. Iterable tensor `for` remains deferred. |
| Ownership | SUPPORTED / QUALIFIED | Immutable aliases, explicit `copy`/`move`, bounded mutable borrow analysis, view-root lifetimes, and conservative rejection. No raw pointers or general unsafe escape hatch. |
| Scan | SUPPORTED / QUALIFIED | Forward structured Scan semantics and reference execution, including scalar/tensor state and captures. Native Scan lowering is not supported. |
| Reverse-mode AD | EXPERIMENTAL | Bounded pure straight-line subset with explicit saved values and generated forward/backward IR. Structured/recurrent reverse AD for Scan is deferred. |
| Extension derivative | EXPERIMENTAL | Optional validated scalar derivative recipe lowered into ordinary AD IR. |
| Native CPU | SUPPORTED / QUALIFIED | AOT `.tha`, external-toolchain JIT, and direct source run/build for the bounded scalar/rank-1/rank-2 TensorRegion subset. Generated loops are scalar and non-vectorized. |
| Native GPU | SUPPORTED / QUALIFIED | Explicit NVIDIA CUDA Driver/PTX backend for the bounded TensorRegion subset; no evaluator, framework, or CPU fallback. |
| Async lifetime model | SUPPORTED / QUALIFIED | Internal pending-operation ownership, reservations, observation, and safe drop-drain. No public async source syntax or cancellation API. |
| Planning and fusion | SUPPORTED / QUALIFIED | Deterministic slot reuse and conservative linear elementwise fusion inside verified TensorRegions. No global optimizer or general performance claim. |
| Training | EXPERIMENTAL | Deterministic reference training state transition, SGD/momentum state, transactional `.thc` checkpoint, and a small affine reference model. No public training CLI or native training. |
| `.tha` deployment | SUPPORTED / QUALIFIED | Versioned trusted native artifact with CPU ELF or GPU PTX payload and local runtime inspection/execution. Not portable single-file application shipping. |
| `.thc` checkpoints | EXPERIMENTAL | Versioned training checkpoint format for the reference training contract. |
| `.thm` models | SUPPORTED / QUALIFIED | Versioned reference-model bundle with frozen parameters and embedded CPU/GPU artifacts; narrow scalar-spelled one-input CLI ABI. |
| CPU numerical graphics | EXPERIMENTAL | Headless typed 3D transform, clipping, raster/depth/interpolation runtime. |
| GPU graphics | EXPERIMENTAL | Native CUDA compute renderer with shared CPU contract and host-materialized output. No game engine, windowing, graphics API integration, or source syntax. |
| Research extensions | EXPERIMENTAL | Explicitly loaded trusted `.so` compiler plugins declaring bounded f32 scalar recipes, backend masks, fusion, and optional derivative. Artifacts embed recipes and run without the plugin. |
| CLI | SUPPORTED / QUALIFIED | `check`, CPU/GPU `run`, CPU/GPU `build`, artifact inspect/run, model inspect/run, and repeated explicit `--extension`. |
| Package ecosystem / IDE / LSP | NOT SUPPORTED | No registry, dependency solver, package manager, IDE, or LSP. |
| Portable single-file shipping | DEFERRED | TH-026 owns cross-machine/single-file deployment. |
| Proof system | DEFERRED | No theorem-proving or proof infrastructure in V0.1. |
| Local AI / System-1 / ThiranQL | DEFERRED | No assistant model, query language, or local-AI runtime in V0.1. |

## Qualified environment

The physical qualification environment is Ubuntu 24.04 on WSL2, x86_64,
glibc 2.39, CMake 3.28.3, Ninja 1.11.1, and GCC 13.3. GPU qualification is
only for an NVIDIA GeForce RTX 3050 Ti Laptop GPU, compute capability 8.6,
with NVIDIA driver 610.62. Other NVIDIA devices are architecturally intended
for the PTX/Driver design but were **not physically qualified**. Native Linux
outside this WSL2-focused environment was not separately qualified. Windows
native and macOS releases are not supported.

## Known limitations and performance status

These limitations are part of the release contract:

- CPU code generation is scalar/non-vectorized and has no CPU threading.
- TH-023 measured the current CPU system much slower than competent optimized
  C++20, NumPy, and single-thread PyTorch on its qualified elementwise chain.
  TH-015 fusion reduced that CPU chain runtime by about 21–24% by median, but
  this does not make the backend performance-competitive.
- GPU calls currently repeat H2D/D2H transfers, allocation, module load, and
  CUDA Driver JIT work. There is no persistent generic device-resident Tensor
  model. Transfer-inclusive GPU execution was much slower than installed
  PyTorch’s inclusive path on the qualified TH-023 workload.
- TH-024 qualified same-environment artifact byte reproducibility, not
  cross-machine reproducibility. Cross-machine and cross-NVIDIA-GPU
  qualification were unavailable.
- Source `run`/`build` currently require a zero-parameter entry (default
  `main`) and support only the bounded native TensorRegion subset.
- Research extensions are trusted native compiler plugins. There is no
  malicious-plugin sandbox, signature verification, remote registry, or
  package discovery.
- Artifacts, models, and checkpoints have structural integrity checks but no
  signatures or hostile-input/native-code sandbox.
- Reverse AD does not support recurrent Scan differentiation.
- There is no single-file portable executable, binary installer, package
  ecosystem, external adoption claim, or broad compatibility guarantee.

Read the complete [TH-023 performance qualification](checkpoints/TH-023.md)
and [TH-024 robustness qualification](checkpoints/TH-024.md). Do not compare
device-resident PyTorch measurements to transfer-inclusive Thiran rows as
equivalent scopes.

## Build and runtime dependencies

### Required to build

- Linux/WSL2 x86_64 in the qualified profile;
- CMake 3.20 or newer;
- a C++20 compiler (GCC 13.3 was qualified);
- Ninja for the supplied presets;
- standard Linux development/runtime facilities including `dlopen` support.

### Required to run CPU workflows

- the built/installed `thiran`, `thiran-artifact`, or `thiran-model` tool as
  appropriate;
- ordinary ELF/glibc/libstdc++ runtime libraries;
- a host C++ compiler for source CPU run/build and CPU JIT, but not for later
  execution of an already-built CPU `.tha` or `.thm` payload.

Python, NumPy, and PyTorch are not production native-runtime dependencies.
They are used by contributor tests, legacy Region/Python workflows, and
qualification/benchmark comparisons.

### Required to run GPU workflows

- a GPU-enabled build (`-DTHIRAN_ENABLE_NATIVE_GPU=ON`);
- dynamically loadable `libcuda.so.1`, an NVIDIA driver, and a compatible
  NVIDIA device. The accepted backend emits PTX and uses Driver JIT; `nvcc`
  is not required for this path.

### Optional qualification dependencies

- Python 3 and PyTorch for `BUILD_TESTING=ON` full legacy/integration tests;
- NumPy/PyTorch for TH-023 comparison controls;
- ASan/UBSan-capable compiler/runtime for sanitizer qualification.

## Build from source

Release is the recommended user configuration:

```bash
cmake --preset release
cmake --build --preset release
build/release/thiran --version
```

The supplied Release preset builds tests and therefore selects Python and
PyTorch. A production-only CPU build has neither dependency:

```bash
cmake -S . -B build/user-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DTHIRAN_ENABLE_NATIVE_GPU=OFF
cmake --build build/user-release
build/user-release/thiran --version
```

For a GPU-enabled source build:

```bash
cmake -S . -B build/user-gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DTHIRAN_ENABLE_NATIVE_GPU=ON
cmake --build build/user-gpu
```

The backend dynamically loads the CUDA Driver. CMake may report that no CUDA
compiler was found; `nvcc` is not required by the accepted low-level PTX path.
Local installation is available through `./scripts/install.sh --prefix PATH`.

## CLI quick start

```bash
build/release/thiran --version
build/release/thiran --help
build/release/thiran check examples/basic_tensor.th
build/release/thiran run examples/basic_tensor.th --backend cpu
build/release/thiran build examples/basic_tensor.th --backend cpu \
  -o /tmp/basic.tha
build/release/thiran artifact inspect /tmp/basic.tha
build/release/thiran artifact run /tmp/basic.tha
```

Replace `cpu` with `gpu` only in a GPU-enabled build on compatible hardware.
An explicit GPU request never falls back. See [CLI.md](CLI.md) for exact flags
and [the extension example](../examples/research_extension/README.md) for
explicit plugin loading and plugin-free artifact execution.

## Evidence and ABI inventory

- [ABI/format inventory](ABI_V0_1.md)
- [Machine-readable release manifest](../release/THIRAN-0.1.0.json)
- TH-023 SHA-256:
  `04f7c9d26d4dd749d308328dd69bd5f83f0c4c3786fe380de65a7d4fc0858e31`
- TH-024 SHA-256:
  `505b82dd929ebf5bf93dbbdf392ac4d5c4ffebeb56524bfbf6d16407daa7217f`

The release is licensed under Apache License 2.0; see `LICENSE`.
