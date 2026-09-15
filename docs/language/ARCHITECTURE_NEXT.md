# Future compiler and execution architecture (TH-001)

**Design, not implementation.** The current R12 path is a C++20 frontend flattened to one single-result Graph, deterministic Region planning, and PyTorch emission. Planning labels `AOT`, `JIT`, and `FALLBACK` do not execute different native implementations. The `Input -> ReLU -> ReLU -> Output` `--plan` crash is a TH-002 baseline defect and is not addressed here.

## Minimum pipeline and responsibility

```text
Source / module resolver
  -> lexer, parser, spanned AST                 syntax and provenance
  -> typed structured semantic IR              values, scopes, calls, branches,
                                               loops, tuples, state, types, shape contracts
  -> ownership / alias / effect analysis        read aliases, explicit moves/copies,
                                               exclusive mutation, I/O,
                                               RNG, transfer, async lifetimes
  -> requested AD transformation                explicit primal/derivative programs
  -> tensor/dataflow regions                    pure or controlled-effect subgraphs,
                                               explicit multi-value boundaries
  -> kernel/loop IR                              iteration, reductions, scalar ops,
                                               device/legal layout choices
  -> buffer/storage IR                          allocations, views, lifetimes,
                                               transfers, events, saved AD values
  -> native CPU/GPU lowering + runtime          executable code and checked services
```

The AST owns source spans and syntax, not execution semantics. Typed structured IR is the **universal semantic authority**: it keeps function calls, lexical blocks, branch/loop structure, multi-results, state transitions, shape checks, typed failures, and explicit effect ordering. An ordinary tensor value is a mathematical logical value, not a uniquely owned physical buffer. Typed IR distinguishes non-consuming read alias, borrowed view, explicit deep copy, explicit consuming move, and exclusive mutable access, even when common read-only calls have no visible borrow notation. Ownership/alias analysis validates live storage conflicts and escaping aliases there, producing constraints preserved downstream. It must not require read-borrow ceremony just to make buffer reuse easier. Effect analysis distinguishes pure math, mutation, RNG, I/O, host/device communication, persistent state, and asynchronous completion; its tokens/dependencies prevent illegal reordering. Public effect syntax can be staged, but IR effect facts cannot be omitted. Shape checks remain visible operations when dimensions are dynamic.

AD is requested for typed functions or regions and produces a checked transformed program with explicit gradient/state results. It is neither an invisible global tape nor a backend-specific reinterpretation. The transformation records saved values and recomputation legality; it must preserve mutation versions, RNG ordering, state outputs, errors, and effect ordering. Forward mode may use the same semantic boundary later. Custom derivatives are verified against declared types/effects and backend availability. A backend cannot silently differentiate an effectful or nondifferentiable operation.

Tensor/dataflow regions contain portions that are legal to treat as dataflow, with explicit value, state, effect, and alias/lifetime boundaries. Arbitrary `if`, loops, recursion, I/O, and stateful steps remain in structured IR unless a proven legal transformation extracts a region. Kernel/loop IR maps logical tensor iteration and reductions to lowerable computations without hard-coding every custom composition into the frontend operation enum. Buffer/storage IR is the first place physical layout, allocation, reuse, handle retention/elision, transfers, and asynchronous events become concrete; its verifier must enforce all live aliases, views, exclusive reservations, and saved-AD lifetimes. Physical reuse or in-place lowering of a logical result is legal only when no source-visible alias, numerical result, effect, FFI obligation, or failure behavior changes. A hidden copy-on-write allocation is not a legal substitute for rejected mutable access. Each lowering must have a verifier and semantic equivalence tests. Internal IR serialization is not a public compatibility format.

## R12 reuse boundary

| Current component | Future role | Boundary |
| --- | --- | --- |
| `SourceManager`, source spans, deterministic module discovery/diagnostics | Reuse and extend | Preserve relative explicit imports and provenance; add packages/ordinary scopes without Graph flattening. |
| Current frontend lexer/parser/AST | Adapter or replacement | Existing assignment/function grammar cannot express V0 values or control flow; do not make it the semantic authority. |
| `Graph`, `Node`, `GraphVerifier`, rewrites, shape inference | Reuse only for legal pure tensor subgraphs after redesign/verification | Single-result, acyclic Graph cannot universally encode tuples, branches, loops, mutation, effects, or persistent state. Defect correction is TH-002. |
| Region formation/`RegionPlan`/verifiers and explicit boundary ordering | Adapt useful determinism and partition contracts | Current plan borrows Graph nodes and has tensor-only boundaries. Future regions require multi-value/effect/state/lifetime contracts and freshness checks. |
| PyTorch emitters and generated Region runtime | Optional oracle/interop adapter | Not native runtime, final dependency, or architecture authority. Numeric conformance must compare against the language contract. |
| Scheduler, device/partition/communication descriptions, `BackendIR`, Triton stubs | Evidence/scaffolding to reassess | Descriptive stages are not executable CPU/GPU or distributed execution; no AOT/JIT/cache claim follows from labels. |

The first native vertical slice should be a small, end-to-end typed function using `i64` scalars and a rank-2 `i64` tensor with checked indexing, checked elementwise addition, and an explicit result; it must pass exact reference/native value and checked-failure equivalence plus safe-lifetime checks. It need not include floating-point envelopes, AD, packages, GPU, or matmul. TH-002 establishes baseline correctness before extending the pipeline. TH-004 through TH-008 build the language and numerical semantics; this document does not start those checkpoints.

## Backend choices to evaluate

Keep the C++ implementation while measuring its correctness and maintainability. Compiler implementation language and source-language safety are separate; a Rust rewrite is unjustified without evidence that C++ blocks the product. Evaluate LLVM for CPU scalar/control-flow/native object lowering when the structured and kernel IR contracts are ready. Evaluate MLIR selectively if its tensor/shape/affine infrastructure lowers real Thiran programs more clearly than bespoke IR; do not add it merely to obtain a modern stack. Use established BLAS for matrix kernels where contracts, hardware, precision, and licensing fit; writing every kernel ourselves is not presumed superior. Start with one GPU path, provisionally CUDA because it offers mature compiler/library tooling and can be tested against existing optional CUDA environments, then measure portability costs before expanding. GPU unavailability must not affect CPU language semantics. Optional Triton interoperability may lower selected kernels or import externally authored ones under explicit type/shape/effect/ownership/derivative contracts; current stubs are not such a backend. External native libraries enter through versioned FFI/ABI wrappers and may supply optimized kernels, codecs, visualization, and device services.

## Equivalence and benchmarking boundary

Interactive/reference, CPU, GPU, training, and deployment use the same typed program meaning. Backends may change storage and scheduling only within ownership/effect rules and the declared numerical profile. Each supported operation/dtype must publish exactness or tolerance expectations, including reduction order/accumulation, before cross-backend equivalence is claimed. Fast/relaxed modes are opt-in, recorded in artifacts, and compared under their own quality conditions. Python/PyTorch is a development oracle for selected cases, never the normative definition. Benchmarks distinguish kernel, prefill, stateful step, full request, and renderer frame time and freeze the fields in [CONSTITUTION.md](CONSTITUTION.md).

## State and Nerivu boundary

An explicit state value enters a function and an explicit next-state value leaves it; multiple results and typed state checkpoint APIs support reset, prefill, step, and sequence/chunk execution. Persistent runtime state owns its buffers and records outstanding device work. Transformer, recurrence, state-space, hybrid, and future custom mechanisms are ordinary Thiran functions/compositions. Nerivu remains separate research; integration later tests general state, AD, custom math, and native low-latency execution rather than adding Nerivu-specific syntax.
