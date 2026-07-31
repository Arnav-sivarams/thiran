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
