# Thiran language constitution (TH-001)

**Status:** design target, not an implemented-language claim. The repository at `0.2.0-alpha-dev` remains an R12 tensor-dataflow compiler prototype. [SEMANTICS_V0.md](SEMANTICS_V0.md) records the initial semantic contract; [DECISIONS.md](DECISIONS.md) records its tradeoffs.

## Identity and users

Thiran is intended to be a stable, memory-safe numerical systems language for students, researchers, and native production applications. The complete path should support opening and inspecting data, numerical programs, custom models, differentiation, training, experiments, checkpoints, native compilation, and deployment without mandatory Python or PyTorch. The same ordinary language constructs should support geometric and selected rendering applications; simple plotting should be available through a future library or interop surface. Numerical notation should be concise; programs should still have modules, functions, lexical scopes, explicit control flow, structured data, and predictable errors.

The first-year student should be able to use vectors, matrices, indexing, reductions, and future plotting libraries without knowing compiler IR or writing visible borrow syntax for ordinary read-only numerical calls. Binding a tensor value for another read must neither invalidate the first binding nor secretly copy all its elements; independent storage and consuming transfer remain explicit choices. The researcher should be able to compose new mathematical mechanisms without editing a compiler operation enum. A genuinely new primitive needs an explicit extension interface with type, shape, effect, derivative, and backend contracts. Production users need source stability and reproducible artifacts.

## Boundaries and non-goals

Thiran is neither Python/MATLAB source compatible nor a NumPy, PyTorch, or Rust clone. It is not solely a graph DSL, inference compiler, or graphics language. Python/PyTorch may be interoperability tools and reference oracles; their behavior does not define Thiran semantics. Nerivu is separate model research and does not receive source syntax here. No model mechanism, backend, or speed claim is frozen by TH-001.

## Safety promise

The **future safe subset** must prevent memory unsafety through safe source constructs: ownership and borrowing prevent dangling or conflicting views, indices are checked, and FFI and raw pointer operations require a checked `unsafe` boundary. Safe code may still return an error, fail on a documented checked precondition, exhaust resources, or have a numerical result that requires interpretation. This is a design promise, not a claim that the present compiler implements or formally proves memory safety. GPU operations must preserve borrow and storage lifetimes through asynchronous completion.

## Stability promise

Once a language edition is declared stable, accepted ordinary source should keep its specified meaning across compatible compiler releases. Changes to syntax, typing, indexing, ownership, numeric contracts, or public library behavior that alter existing accepted programs require an explicit breaking edition or major version. Deprecation has a published lifecycle. Stable-language conformance tests, independent of any one backend, govern the promise. The current `0.2.0-alpha-dev` language is experimental; it has no stable-source or native-ABI promise. See [COMPATIBILITY.md](COMPATIBILITY.md).

## Execution modes

The intended modes are interactive/reference execution, native CPU, native GPU, training, and deployment. Each consumes the same typed semantics. A reference mode is a testing and teaching aid, not the architectural authority. A backend may differ in scheduling and permitted rounding within an explicitly documented numerical profile, but may not silently change indexing, shape, ownership, error, effect, or model-state behavior. Numerically relaxed compilation must be explicit. Native production deployment must not require Python/PyTorch.

## User experience target

The eventual single `thiran` command should expose coherent verbs such as `run`, `check`, `test`, `build`, `fmt`, `repl`, and `doctor`. These are design candidates, not commands available in the current CLI. Installation should provide a working compiler, standard library, and a diagnostic of available CPU/GPU capabilities; Python must be optional. A first numerical program should fit in one file and use `let A = [1, 2; 3, 4]`, `A[0, 1]`, and `A * B`. A first model should be a normal function using tensor values. A first training loop should make parameters, RNG, state, gradient, and update explicit. A first native build should state target, numeric profile, and artifact requirements. A first compiler error should point to the source span, explain the failed rule, and suggest a concrete fix without showing internal Graph names.

## Performance-claim contract

“HFT speed” is not a benchmark. Before a claim, freeze exact workload, model and weights, quality condition, hardware and software, precision and numerical profile, batch, context and state sizes, cold/warm status, compilation and cache status, transfer boundaries and synchronization, sample count, latency quantiles (at least median, p95, and p99), peak/steady memory, and an independently repeatable method. Report kernel time, full-model prefill, stateful step/decode, complete application request, and renderer frame time separately. Targets require named hardware and workload; no target is set here.
