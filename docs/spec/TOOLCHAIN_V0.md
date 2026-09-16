# TH-009 V0 developer toolchain

TH-009 exposes the accepted bounded V0 path as a reusable C++ driver and an experimental `thiran-v0` executable. It is development infrastructure, not the stable production CLI, a package manager, a stable artifact format, or a production-AOT claim. The legacy `Thiran` executable and installed `thiran` command are unchanged.

## Driver and stages

`tooling::CompilerDriver` composes existing authorities in this order:

```text
exact source bytes -> V0 parser -> semantic analyzer -> semantic verifier
  -> ownership/effect analysis -> ownership fact audit
  -> STRICT_NATIVE TensorRegion extraction -> deterministic C++20 emission
  -> configured host compiler -> native artifact -> new process
```

The driver does not contain a parser, checker, ownership analysis, extractor, or emitter. Its structured results carry success, failure stage, diagnostics, native coverage, generated text, artifact path, development build record, and subprocess exit/output where applicable. Failure stages remain distinct: input, syntax, semantic, semantic-verifier, ownership/effect, backend, host compiler, artifact execution, and internal tool. Source paths are diagnostic identity only and do not enter deterministic region or generated-source identity.

`checkSource` ends after the ownership fact audit. Native coverage is deliberately not a condition of language correctness. Consequently a valid MatMul program passes `check` and fails native extraction/build with `BACKEND-UNSUPPORTED`, `MatMul unsupported-native`, and `fallback: NONE`.

TH-010 f32 source accepted by the semantic/effect stages is likewise meaningful to `check`. The current TensorRegion/native path may reject f32, `stop_gradient`, generated gradient programs, and generated AD helper operations as `BACKEND-UNSUPPORTED` with `fallback: NONE`. TH-010 adds a C++ reference VJP/grad API but no production or developer CLI `grad` command.

## Developer commands

```text
thiran-v0 check file.th
thiran-v0 emit-region file.th --entry main
thiran-v0 emit-cpp file.th --entry main
thiran-v0 build file.th --entry main -o /tmp/program
thiran-v0 run file.th --entry main
```

`check` reports `CHECK PASS` after the complete non-backend pipeline. `emit-region` prints the TH-008 TensorRegion developer dump, never legacy Graph. `emit-cpp` prints deterministic C++20 without creating an inspection file. `build` writes the final native artifact only at the requested output. `run` builds in temporary storage and executes that artifact as a new process. Build and run use STRICT_NATIVE; neither invokes the reference evaluator, Python, PyTorch, Triton, legacy Graph, or a fallback.

Input is one `.th` file read in binary mode as exact bytes. Its supplied filesystem spelling is retained in located diagnostics. Imports retain the current semantic-stage restriction; the tool does not perform module linking.

## Host toolchain and process safety

The compiler executable, include roots, V0 static archives, and fixed compiler arguments are driver configuration. The experimental executable receives its compiler and build paths from CMake metadata; `/usr/bin/c++` is not language semantics. A host C++20 compiler is currently a bootstrap build dependency. Cross-toolchain selection is deferred.

On TH-009 hosts, process creation is POSIX `fork` plus `execvp` with an explicit argv vector and captured stdout/stderr pipes. No shell command string, `system()`, shell expansion, or user-path quoting parser is used. Source, output, and artifact paths containing spaces and shell metacharacters are individual argv elements. Generated C++ and compiler output live under a `mkdtemp` directory outside the checkout. This implementation is qualified for Linux/WSL developer hosts only; it makes no Windows tooling claim. Language semantics remain platform-independent.

Additional native link requirements are accepted only through verified structured link items: static archive path, shared library name, or search path. Arbitrary linker text and `-Wl,...` pass-through are not an interface.

## Output and record

The development build record contains current stage, entry function, native coverage, optional retained emitted-source path, output artifact path, and host compiler exit status. It is intentionally not a stable artifact manifest; TH-016 owns mature artifact/cache/JIT contracts. Generated C++ is developer inspection output and not a stable source format or native ABI.

The output artifact is copied from the controlled temporary build only after successful host compilation. `run` captures and relays deterministic artifact stdout/stderr and preserves its exit status category. The finished TH-008-style ELF has ordinary host runtime dependencies and no framework dependency. No numerical performance claim is made.
