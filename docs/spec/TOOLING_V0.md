# Tooling workflow V0

TH-022 makes the accepted V0 frontend, ownership analysis, native backends,
artifacts, models, and research extensions available through the production
`thiran` executable. This is a built-repository developer workflow, not a
binary distribution or package manager.

## Command surface

```text
thiran check <source.th> [--extension <library>]...
thiran run <source.th> --backend <cpu|gpu> [--entry <name>]
           [--extension <library>]... [--device <ordinal>]
thiran build <source.th> --backend <cpu|gpu> [--entry <name>]
             [--extension <library>]... -o <artifact.tha>
thiran artifact inspect <artifact.tha>
thiran artifact run <artifact.tha> [--device <ordinal>]
thiran model inspect <model.thm>
thiran model run <model.thm> --backend <cpu|gpu> --input <f32>
                 [--device <ordinal>]
```

Every major command accepts `--help`. Repeated `--extension` options are
loaded in command-line order. There is no directory scan, environment lookup,
or network lookup. `--verbose` reports factual selection and execution
evidence on stderr; successful program values and requested inspection text
remain on stdout.

The older planning, emitter, version, and doctor options remain compatibility
surfaces of the same installed executable. `thiran-artifact` and
`thiran-model` remain focused launchers over the same TH-016 and TH-017
libraries; the primary CLI does not introduce another artifact or model
format. The historical `thiran-v0` developer target remains qualification for
the earlier standalone-CPU path and is not required by the V0 user workflow.

## Source checking

`check` reads exact source bytes and runs the V0 lexer/parser, semantic
analysis, semantic verifier, TH-006 ownership/effect analysis, and ownership
fact audit. It neither lowers nor executes the program. An empty or unreadable
file is rejected. If extensions are supplied, the real TH-021 registry loads
and validates them before semantic analysis and freezes during analysis.

The parser accepts finite decimal f32 literals already tokenized as `Real` and
represented by the existing semantic `Float` instruction. This closes the
narrow source-to-existing-IR seam needed to construct a zero-argument f32
extension example; it does not add casts, implicit mixed dtypes, or new
numeric operations.

## Native run and build

Backend selection is mandatory for `run` and `build`. `cpu` uses strict
TensorRegion CPU extraction and TH-016 ELF shared-object AOT. `gpu` uses
strict GPU extraction and TH-016 persistent PTX. A source `run` builds a
temporary `.tha`, loads it with the artifact runtime, and executes its native
payload. It never invokes the semantic evaluator. A GPU failure is returned
as a GPU error; there is no CPU retry.

The V0 source CLI accepts only a zero-parameter executable entry, named `main`
unless `--entry` is supplied. There is no new general argument or I/O syntax.
Program results use the existing artifact JSON value formatter. CPU artifact
compilation invokes the configured host C++20 compiler by argv through the
existing safe process layer; no shell command string is constructed. GPU AOT
does not invoke a host compiler.

`build` requires an explicit output and refuses to overwrite its input source.
The TH-016 writer validates the output parent, writes a same-directory
temporary file, and atomically renames it into place. Artifact identity is
derived from the lowered region, target, specialization, and plan rather than
the source or extension pathname.

## Artifacts and extensions

Inspection reports the existing stable field order: format and ABI versions,
backend, target, entry signature, planning flags and digest, payload kind,
size and digest, lowered-region size and digest, and runtime requirements.
Artifact execution validates the complete container and accepts only the V0
zero-parameter CLI ABI. The C++ runtime retains its typed argument API.

Extension shared libraries are compiler inputs only. The registry copies the
validated TH-021 descriptor and scalar recipe into semantic and TensorRegion
state. TH-016 serializes that owned recipe into the artifact. Consequently,
`thiran artifact run` neither loads nor needs the extension `.so`. Multiple
extensions remain explicit and collision failures retain their TH-021 error
codes.

## Diagnostics and process contract

Text diagnostics have the form:

```text
path/to/source.th:12:9: error[TH005-...]: message
  12 | source text
     |         ^^^^^
```

The renderer uses existing diagnostic categories and real source spans. It
does not invent a location when none exists. `--diagnostic-format json` emits
one JSON object per diagnostic on stderr with `severity`, `code`, `message`,
`source`, and nullable start/end line and column fields; it consumes structured
diagnostics rather than formatted text.

Exit status 0 means success. Status 2 is command-line misuse, 3 is rejected
input/compilation/loading, 4 is native execution failure, and 5 is a caught
internal tooling failure. Diagnostics and tool failures use stderr. Successful
program results and requested inspect records use stdout.

## CPU-only behavior

With `THIRAN_ENABLE_NATIVE_GPU=OFF`, `check`, CPU source run/build, CPU artifact
run, and CPU extension workflows remain available and the primary CLI has no
mandatory CUDA dependency. An explicit GPU request returns
`GPU-BACKEND-NOT-BUILT`; it never changes backend.

## V0 limitations

This workflow does not provide source imports beyond existing frontend rules,
general runtime arguments, stdin tensor decoding, package management,
extension discovery, hostile-plugin sandboxing, installer/distribution
qualification, remote compilation, IDE services, serving, retraining, or
benchmarks. Artifacts and models retain their existing trusted-local-input
contracts. The semantic evaluator remains an internal oracle and is not
presented as a native backend.
