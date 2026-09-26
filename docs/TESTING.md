# Testing and validation

## Python and Torch selection

Tests are conditional on `BUILD_TESTING`. CMake selects:

1. `THIRAN_PYTHON_EXECUTABLE`, when supplied;
2. `.venv/bin/python`, when present;
3. a Python 3 interpreter found by CMake.

Configuration fails if the selected interpreter cannot import Torch. `BUILD_TESTING=OFF` performs no Python selection or Torch check.

## Preset builds

Debug:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Release:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

Warnings:

```bash
cmake --preset warnings
cmake --build --preset warnings
ctest --preset warnings --output-on-failure
```

Sanitizers:

```bash
cmake --preset sanitizers
cmake --build --preset sanitizers
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --preset sanitizers --output-on-failure
```

If ptrace prevents LeakSanitizer, `detect_leaks=0` may isolate ASan/UBSan behavior, but it is not leak acceptance. The gate remains incomplete until the same build passes with leak detection enabled outside ptrace.

Compiler only:

```bash
cmake --preset no-tests
cmake --build --preset no-tests
ctest --test-dir build/no-tests -N
```

This must report zero tests.

## CTest inventory

Exactly fifteen registrations:

1. `RegionIRTests`
2. `RegionFormationTests`
3. `StrategyClassificationTests`
4. `RegionPlanTests`
5. `RegionPlanCliSupportTests`
6. `RegionPlanCliTests`
7. `RegionPythonEmitterTests`
8. `RegionExecutionTests`
9. `RegionPythonRuntimeTests`
10. `RegionRuntimeCliTests`
11. `InstallationSupportTests`
12. `InstallationCliTests`
13. `SourceManagerTests`
14. `ModuleLinkerTests`
15. `ModuleCliTests`

Run one test:

```bash
ctest --test-dir build/debug -R '^RegionRuntimeCliTests$' --output-on-failure
```

The integration suites are CMake scripts invoked by CTest. Run one directly with the exact variables shown by `ctest --test-dir build/debug -N -V`; normally prefer the registered command above.

## Direct C++ harnesses

```bash
./build/debug/region_ir_tests
./build/debug/region_formation_tests
./build/debug/strategy_classification_tests
./build/debug/region_plan_tests
./build/debug/region_plan_cli_support_tests
./build/debug/region_python_emitter_tests
./build/debug/region_python_runtime_tests
./build/debug/installation_support_tests
./build/debug/source_manager_tests
./build/debug/module_linker_tests
./build/debug/function_semantic_tests
./build/debug/function_lowering_tests
```

TH-016 native artifact qualification adds:

```bash
./build/gpu/v0_native_artifact_tests
./build/gpu/v0_native_artifact_gpu_integration_tests
```

The first covers CPU AOT/JIT, relocation, dependency inspection, typed ABI and
artifact corruption. The second must run with physical CUDA device access for a
TH-016 PASS; CTest code 77 means unavailable, not qualified. It distinguishes
persistent pre-generated PTX (`thiran_ptx_generation=0` during execution) from
Thiran-side PTX generation at GPU JIT compile time.

TH-017 reference-model qualification adds:

```bash
./build/debug/v0_model_deployment_tests
./build/gpu/v0_model_gpu_integration_tests
```

The first performs actual TH-011 training, checkpoint destroy/reload/resume,
checkpoint and bundle adversarial validation, native CPU execution, and a
compiler/source/checkpoint-absent relocated-process run. The second must run
with physical CUDA device access for a TH-017 PASS; CTest code 77 means
unavailable, not qualified. It compares reference, CPU, and GPU predictions,
checks TH-014 pending-resource release, records TH-015 fusion, and distinguishes
persistent PTX from CUDA Driver JIT.

TH-019 native CPU graphics qualification adds:

```bash
./build/debug/v0_graphics_tests
ctest --test-dir build/debug -R '^V0GraphicsTests$' --output-on-failure
```

The suite covers checked transform/camera math, geometry/index validation,
complete homogeneous frustum clipping, top-left coverage and a shared-edge
quad, strict depth behavior in both submission orders, perspective-correct and
clipping-created color interpolation, zero/overflow/malformed inputs, PPM
output, and a frozen perspective-cube color/depth digest. It is a native C++
runtime test and invokes no Python or graphics framework.

TH-020 native GPU graphics qualification adds:

```bash
./build/debug/v0_graphics_gpu_tests
./build/gpu/v0_graphics_gpu_tests
./build/gpu/v0_graphics_gpu_integration_tests
ctest --test-dir build/gpu -R '^V0GraphicsGpu' --output-on-failure
```

The integration executable must run on a physical CUDA device; CTest code 77
means unavailable, not qualified. It compares exact coverage masks and tight
color/depth values for the required scene matrix, exercises async reservations
and independent pending renders, checks same-device byte determinism, and feeds
an observed color Tensor into existing native GPU numerical execution. The
test records the explicit graphics D2H and later numerical H2D boundary; it is
not zero-copy evidence.

TH-021 research-extension qualification adds:

```bash
./build/debug/v0_extension_tests
./build/gpu/v0_extension_gpu_integration_tests
```

The first builds and explicitly loads an out-of-core shared library, checks
registry/descriptor failures, reference and AD behavior, CPU AOT/JIT, planning,
fusion, structured composition, and plugin-free fresh-process CPU artifact
execution. The second requires physical CUDA access (77 means unavailable) and
checks native GPU, persistent PTX AOT, GPU JIT/cache identity, zero-size and
invalid-device behavior, and plugin-free physical fresh-process execution.

## Numerical equivalence

`RegionExecutionTests` and `RegionRuntimeCliTests` construct deterministic tensors, compare Region-controlled execution with whole-Graph PyTorch, check output keys/shapes/dtypes, and use `torch.testing.assert_close` with fixed tolerances. Do not replace these with uncontrolled random inputs.

## Normal-mode compatibility

Normal stdout contains pre-existing `Optimization Time` and `Compile Time` measurements. Compatibility compares stdout byte-for-byte after removing only lines anchored with those labels, while requiring exactly one of each. Stderr, exit status, `generated.py`, and `graph.dot` compare exactly.

Restore tracked artifacts after normal-mode smoke tests:

```bash
git restore -- generated.py graph.dot
```

## Compilation-duplication audit

```bash
ninja -C build/debug -t commands
```

Count each required source/test translation unit and confirm it compiles once. CMake integration scripts must not be compiled.

## Clean-worktree checks

```bash
git status --short
git diff --check
```

Build directories, `.pyc` files, tensor bundles, and scratch executors must not appear. Use `scripts/validate.sh` for the common local gate.
