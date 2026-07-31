# Contributing to Thiran

## Supported environment

Linux and WSL are validated. Use CMake 3.20+, Ninja, a C++20 compiler, Git, and—when tests are enabled—Python with PyTorch. The compiler-only `no-tests` preset requires neither Python nor Torch.

## Repository setup

```bash
git clone <repository-url>
cd thiran
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

For an isolated local installation, use `./scripts/install.sh --prefix "$HOME/.local"`. The build-tree command remains `Thiran`; the installed command is `thiran`.

When using another interpreter, configure with:

```bash
cmake --preset debug -DTHIRAN_PYTHON_EXECUTABLE=/path/to/python
```

## Branches and change scope

Keep each change focused. Preserve unrelated work and avoid broad formatting. Accepted Graph, Region, and RegionPlan semantics must not be changed casually. A semantic change needs a specific rationale, deterministic diagnostics, and regression tests at the affected layer.

Architecture boundaries are described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Strategy classification, Region formation, planning, emission, and execution are separate layers. Do not hide classification in formation or runtime policy in the emitter.

## Adding tests

- Put structural C++ coverage in `tests/region/` using the existing zero-dependency harness style.
- Put source-management and module-linking coverage in `tests/frontend/`.
- Put function semantics and lowering coverage in `tests/frontend/`; preserve the no-function Graph compatibility path.
- Put end-to-end CLI and numerical coverage in `tests/integration/`.
- Put minimal source programs in `tests/fixtures/`.
- Register tests under `BUILD_TESTING` in `CMakeLists.txt`.
- Every semantic change requires a regression test.
- Use deterministic tensors; do not depend on uncontrolled randomness.

See [docs/TESTING.md](docs/TESTING.md) and [docs/ADDING_AN_OPERATION.md](docs/ADDING_AN_OPERATION.md).

## Local validation

Run the normal contributor gate:

```bash
scripts/validate.sh
```

Then validate affected configurations:

```bash
cmake --preset warnings
cmake --build --preset warnings
ctest --preset warnings --output-on-failure

cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

For sanitizer work:

```bash
cmake --preset sanitizers
cmake --build --preset sanitizers
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --preset sanitizers --output-on-failure
```

Do not disable leak detection and call the sanitizer gate complete. A ptrace-limited run may isolate ASan/UBSan with `detect_leaks=0`, but acceptance remains incomplete until a leak-enabled run succeeds.

## Warning requirements

New production and test code must compile without warnings under `-Wall -Wextra -Wpedantic -Wshadow`. Record pre-existing warnings; do not repair unrelated code as part of a scoped contribution.

## Generated artifacts

Normal mode writes `generated.py` and `graph.dot` in the working directory. Do not leave tracked copies unintentionally modified. Do not commit build directories, Python bytecode, temporary `.pt` bundles, or emitted scratch executors.

## Documentation truth

Use these categories explicitly when status could be ambiguous: `IMPLEMENTED`, `ENGINEERING LIMITATION`, `PLANNED`, and `RESEARCH HYPOTHESIS`. Current AOT, JIT, and Fallback Regions all execute through PyTorch. Planned AOT/JIT behavior must not be described as implemented. Performance claims require reproducible benchmarks.

## Commit hygiene

- Review `git status --short`, `git diff`, and `git diff --check`.
- Do not commit unrelated changes or validation outputs.
- Keep generated artifacts restored unless they are intentionally in scope.
- Use a concise commit subject describing the completed change.
- Build directories must not be committed.

## Pull-request checklist

- [ ] Scope and architecture boundaries are stated.
- [ ] Required tests cover every semantic change.
- [ ] Debug tests pass.
- [ ] Warning validation passes without new warnings.
- [ ] Sanitizer validation, including leak detection, passes.
- [ ] Release and relevant `BUILD_TESTING=OFF` checks pass.
- [ ] Numerical equivalence is tested when lowering or execution changes.
- [ ] Documentation matches implemented behavior.
- [ ] `generated.py` and `graph.dot` have no unintended changes.
- [ ] `git diff --check` passes.

No proprietary tool, paid service, or AI assistant is required to contribute.
