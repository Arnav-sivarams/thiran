# Changelog

## 0.1.0 — V0.1 experimental technical release

First scoped source release of the accepted V0 compiler/runtime system.

### Capabilities

- Statically checked structured numerical source with scalar/tensor values,
  indexing and slicing, explicit ownership operations, control flow, and Scan.
- Bounded compiler-native reverse differentiation and deterministic reference
  training/checkpoint machinery.
- Native CPU and NVIDIA PTX/CUDA Driver execution for the qualified
  TensorRegion subset, with explicit no-fallback behavior.
- Memory planning and conservative elementwise loop/kernel fusion.
- `.tha` native artifacts, `.thm` reference model bundles, and `.thc` training
  checkpoints with versioned formats and integrity validation.
- Headless CPU numerical 3D graphics and a native GPU compute renderer.
- Trusted research extensions using public scalar recipes and optional reverse
  derivatives; built artifacts do not need the extension library at runtime.
- Public `check`, `run`, `build`, `artifact`, and `model` CLI workflows.

### Qualification

The release boundary preserves the accepted TH-023 performance evidence and
TH-024 robustness/reproducibility evidence. See
[`docs/RELEASE_V0_1.md`](docs/RELEASE_V0_1.md) for the exact support matrix,
qualified environment, and claim limits.

### Known limitations

CPU generated execution is scalar and non-vectorized and was substantially
slower than competent optimized C++20 on the qualified TH-023 chain. Current
GPU execution repeatedly transfers data and loads/Driver-JITs modules; its
transfer-inclusive time was substantially slower than installed PyTorch on
that workload. Qualification is Linux/WSL2-focused and covers one RTX 3050 Ti
Laptop GPU (CC 8.6), not cross-machine or cross-GPU portability. There is no
portable single-file executable, package ecosystem, extension sandbox,
artifact signing, or recurrent reverse AD for Scan.
