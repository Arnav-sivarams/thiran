# TH-001 decision register

Each ID is stable. “Unresolved” names remaining detail rather than reopening the chosen semantic boundary. Checkpoints are owners of implementation/specification follow-through, not authorization to start them. The decisions were screened against student simplicity, research composability, static checkability, hidden cost, CPU/GPU and AD coherence, production stability, unnecessary imitation, migration cost, and independent conformance testing.

## TH-LD-001 — Product and safety boundary

**Question:** What language is Thiran, and what does safe source promise? **Chosen:** A stable, memory-safe numerical systems language with a checked safe subset and explicit unsafe/FFI boundary; Python/PyTorch remain optional. **Alternatives:** Graph DSL, framework wrapper, Python-compatible syntax, compiler rewrite into Rust. **Why:** Students need ordinary programs; research needs custom composition; production needs native semantics. Source safety is independent of compiler language. **Costs:** Ownership and runtime/FFI contracts need substantial verification. **Unresolved:** Precise unsafe wrapper/ABI specification. **Owner:** TH-006 and TH-010.

## TH-LD-002 — Static type model

**Question:** Static or dynamic values, and how much inference? **Chosen:** Static scalar/tuple/record/function/tensor/buffer types; infer locals and private acyclic results; annotate parameters, public interfaces, and recursion. **Alternatives:** Dynamic typing, full global inference, object hierarchy. **Why:** Predictable native compilation with manageable student syntax and stable public contracts. **Costs:** Some annotations at library boundaries. **Unresolved:** Generic constraint grammar and diagnostic policy. **Owner:** TH-004 and TH-005.

## TH-LD-003 — Numeric dtypes and conversion

**Question:** Which scalar types, promotions, and overflow? **Chosen:** `bool`, signed/unsigned 32/64-bit integers, `f32`/`f64`; untyped integer literals fit exactly and untyped real literals round contextually once under a specified backend-independent rule; already typed mixed operands require explicit casts; checked integer arithmetic and IEEE default floats. **Alternatives:** A small language-owned safe promotion lattice, NumPy-style promotion, exact-only real literals, wrapping default. **Why:** A partial safe lattice still leaves common `i64`/`f32` and signed/unsigned cases without one lossless result and can make dtype changes hard to predict. Exact-only decimal literals would make ordinary `f32` constants like `0.1` unusable. Contextual literal rounding is not silent narrowing of an already typed value. **Costs:** Explicit casts in genuinely mixed-typed work and decimal conversion conformance. **Unresolved:** `f16`/`bf16`/complex and kernel rounding envelopes. **Owner:** TH-004 and TH-008.

## TH-LD-004 — Scalar, tensor, and buffer relation

**Question:** Are scalars, rank-0 tensors, and general arrays one type? **Chosen:** Distinct scalar and `Tensor<T,R>` types; numeric rectangular literals create tensors; `Buffer<T>` is CPU-addressable storage for records and bytes, not a substitute numerical tensor API. **Alternatives:** Everything a tensor, implicit scalar arrays, tensors of arbitrary records. **Why:** Concise mathematical literals with general systems storage and FFI byte/record access. **Costs:** Explicit conversions between buffers and tensors. **Unresolved:** Dynamic-rank existential tensor API and buffer growth surface. **Owner:** TH-004 and TH-008.

## TH-LD-005 — Rank, shape, and layout

**Question:** Which tensor dimensions are static? **Chosen:** Rank in the common type, extents checked at runtime or compile-time constants, logical row-major indexing independent of physical layout. Rank-polymorphic generics handle specialization-known rank; a rank-inspected existential is required for data-dependent ingestion before claiming the full dataset workflow, but not the first fixed-rank slice. **Alternatives:** All dimensions in type, runtime rank everywhere, one mandatory physical layout. **Why:** Native specialization without excluding dynamic spatial/token extents or truly dynamic-rank data at an explicit boundary. **Costs:** Runtime shape guards, rank-specific APIs, and later existential dispatch. **Unresolved:** Shape-constraint notation and dynamic-rank dispatch. **Owner:** TH-004 and TH-008.

## TH-LD-006 — Index and range convention

**Question:** Index base and slice bounds? **Chosen:** Zero-based indices, half-open ranges, checked nonnegative indices, positive steps. **Alternatives:** MATLAB one-based/inclusive, Python negative-index shorthand. **Why:** Coherent loops, buffers, tensors, and native address calculation. **Costs:** MATLAB users must adapt. **Unresolved:** None foundational; omitted-bound grammar is TH-004. **Owner:** TH-004.

## TH-LD-007 — Mathematical operator spelling

**Question:** Meaning of `*`, `.*`, `/`, and matrix literals? **Chosen:** Rank-2 `A * B` matmul, `A .* B` elementwise; tensor `+/-` elementwise, tensor/tensor `/` rejected, `./` elementwise; rectangular `[1, 2; 3, 4]` literal. **Alternatives:** All `*` elementwise, function-only matmul, MATLAB matrix division/implicit concatenation. **Why:** Concise and visually mathematical with explicit computational categories. **Costs:** Batched matmul needs a named operation; literal grammar must disambiguate ranges. **Unresolved:** Power/dot/contract API names and operator precedence table. **Owner:** TH-004 and TH-008.

## TH-LD-008 — Broadcasting

**Question:** How do elementwise shapes align? **Chosen:** Trailing-axis equal-or-one broadcast, including zero with one; scalar acts as rank zero; result is a new logical value without mutating operands. **Alternatives:** No broadcast, unrestricted named-axis broadcast, writable broadcast views. **Why:** Familiar numerical convenience with a checkable shape rule. **Costs:** Dynamic shape failures and possible temporary allocation. **Unresolved:** Named-axis API, if any. **Owner:** TH-008.

## TH-LD-009 — View versus copy

**Question:** What do slicing, transpose, reshape, and concatenate return? **Chosen:** Slices/transpose are borrowed read views tied to a live source handle; mutable views need exclusive access; `reshape_view` checks layout; `reshape_copy` and `copy` explicitly create independent storage; concat/arithmetic/matmul return new logical values without mutating operands. **Alternatives:** Everything copies, silent reshape copying, implicit writable aliases. **Why:** Makes views and large independent copies visible without forcing physical buffer ownership into mathematical expressions. **Costs:** More explicit APIs; backend layout support needs validation. **Unresolved:** Legal negative-stride/device view subset and disjoint mutable split API. **Owner:** TH-006 and TH-008.

## TH-LD-010 — Ownership and assignment

**Question:** What does binding assignment do? **Chosen:** Ordinary tensor/buffer binding is a non-consuming read-only logical value alias without a deep copy; `copy` creates independent storage, `move` explicitly consumes a binding, `x = y` rebinds, and indexed/in-place operations require statically justified exclusive access. No hidden copy-on-write allocation. The compiler may turn an alias into physical transfer after source last use. **Alternatives:** Move-by-default, implicit deep copies, shared mutable tensors, hidden copy-on-write. **Why:** `let B = A; let C = A + B` must be valid at no hidden O(n) cost while conflicting mutation remains rejected. **Costs:** Alias/liveness analysis and possible runtime storage-handle bookkeeping at escaping/async boundaries; explicit uniqueness decisions for mutation. **Unresolved:** Compound in-place operator vocabulary and handle-elision proof rules. **Owner:** TH-006.

## TH-LD-011 — Borrow, AD, and asynchronous lifetime

**Question:** How are views, saved values, and GPU work kept safe? **Chosen:** Immutable value aliases may share storage; views borrow a live handle; exclusive mutation requires no potentially live alias/view/save/event on that storage; AD and async reservations last through their actual use/completion. **Alternatives:** Move-by-default uniqueness, GC alone, unchecked aliases, implicit snapshots. **Why:** Safe read sharing with no hidden deep copy and predictable mutation costs. **Costs:** Static alias/liveness analysis plus runtime retention where values escape or GPU work is asynchronous; proven single-owner paths may erase bookkeeping, but universal zero-cost counting is not promised. **Unresolved:** Event ownership/cancellation API and recomputation legality details. **Owner:** TH-006, TH-007, and TH-008.

## TH-LD-012 — Functions and multiple results

**Question:** Are functions forever compile-time Graph macros? **Chosen:** Ordinary typed runtime functions with non-consuming read-only tensor/buffer parameters by default, explicit mutable and consuming modes, tuple results, explicit recursive signatures, bounded generics; legal inlining is optimization. All modes remain explicit semantic IR facts. **Alternatives:** R12-only expansion, universal no-recursion rule, visible read-borrow syntax on every call, implicit mutation. **Why:** Nested model and helper calls need concise read use while updates/FFI remain visibly exceptional. **Costs:** Structured IR/native calls and mode-aware escape analysis for alias-valued results and views. **Unresolved:** Closure/higher-order capture syntax, view-lifetime signature spelling, and generic constraints. **Owner:** TH-005.

## TH-LD-013 — Structured control flow

**Question:** How do branches and loops interact with tensors? **Chosen:** Scalar-bool `if`/`while`, half-open `for`, leading-axis read iteration, break/continue; tensor masks use explicit selection; control flow remains structured. **Alternatives:** Flatten all control flow into Graph, implicit tensor truthiness, compile-time-only loops. **Why:** Student and production programs need predictable ordinary control flow, including stateful models. **Costs:** AD/control lowering complexity and device synchronization effects. **Unresolved:** Differentiability rules for data-dependent loop conditions and GPU loop lowering. **Owner:** TH-005 and TH-007.

## TH-LD-014 — Error model

**Question:** Typed errors or exceptions? **Chosen:** Spanned static rejection; checked runtime contract failures for bounds/overflow/dynamic shapes; `Result`/`Option` alternatives; and resource/runtime failures (allocation/device) as a distinct category with recoverable APIs where needed. No safe-source failure becomes undefined memory behavior. **Alternatives:** Exceptions, sentinel values, every ordinary operator returning `Result`, unchecked failure. **Why:** Introductory code remains readable, while production can select recoverable operations; resource loss is not blamed on a programming contract. **Costs:** Result plumbing, allocation-sensitive variants, and GPU error observation rules. **Unresolved:** Error-type standard-library taxonomy and propagation spelling. **Owner:** TH-005 and TH-008.

## TH-LD-015 — AD boundary

**Question:** Where does differentiation live? **Chosen:** Explicit request on eligible typed functions; reverse-mode transformation with owned/lifetime-checked gradients, saved values, custom derivatives, stop-gradient/no-grad, and effect legality. **Alternatives:** Hidden global tape, backend-specific AD, compiler enum for each model. **Why:** Research composability and state/mutation correctness. **Costs:** IR transformation, verifier, numerical tests. **Unresolved:** Forward mode scheduling, custom derivative interface, nondifferentiable branch/loop rules. **Owner:** TH-007.

## TH-LD-016 — State and effects

**Question:** How do pure math, state, RNG, I/O, transfer, and async work differ? **Chosen:** Explicit state input/next-state output and distinct ordered IR effects; public mutation/IO/RNG capabilities must be visible in interfaces, even if full surface effect syntax is staged. **Alternatives:** Hidden global state/tape, pure Graph for everything, mandatory research-grade effect types in V0 source. **Why:** Correct training and native deployment without student-facing ceremony. **Costs:** Effect inference/declarations and token ordering. **Unresolved:** Public effect annotation spelling and state/checkpoint library APIs. **Owner:** TH-005, TH-007, and TH-008.

## TH-LD-017 — Modules and packages

**Question:** Preserve R12 imports or replace them? **Chosen:** One file per module, explicit relative import with alias and `export` visibility; future manifest/locked versioned dependencies and local projects. **Alternatives:** Implicit search paths, global imports, package registry as prerequisite. **Why:** Deterministic resolution and approachable single-file start. **Costs:** Manifest/version compatibility work. **Unresolved:** Manifest grammar, registry, re-export policy. **Owner:** TH-005 and TH-009.

## TH-LD-018 — Semantic IR and old Graph

**Question:** Which IR owns whole-program meaning? **Chosen:** Typed structured semantic IR is authority; ownership/effect analysis and requested AD precede extraction of legal tensor/dataflow regions; Graph/Region machinery is adapted only inside its legal boundary. **Alternatives:** Universal R12 Graph, direct AST-to-backend, replace all R12 pieces immediately. **Why:** Calls, multiple results, loops, state, and effects cannot be faithfully flattened to the present Graph. **Costs:** New IR hierarchy and verified lowering. **Unresolved:** Concrete node schema and extraction proofs. **Owner:** TH-004 through TH-008.

## TH-LD-019 — Backend and reference roles

**Question:** What implements native execution? **Chosen:** Retain C++ for now; evaluate LLVM CPU, selective MLIR, established BLAS, one provisional CUDA path, optional Triton/FFI adapters; reference is a conformance aid. **Alternatives:** Full Rust rewrite, all bespoke kernels, PyTorch as production runtime. **Why:** Incremental evidence and established numerical infrastructure. **Costs:** Versioned dependencies and backend conformance work. **Unresolved:** LLVM/MLIR adoption evidence, GPU portability, exact library/kernel selection. **Owner:** TH-008 and TH-010.

## TH-LD-020 — Equivalence and numeric profiles

**Question:** Can reference/native/training/deployment change math? **Chosen:** Same typed semantics and effects; strict profile by default, explicit artifact-recorded relaxed profile; publish per-op/dtype exactness or tolerance before backend support claims. **Alternatives:** PyTorch as authority, silent fast math, bitwise promise without feasible implementation evidence. **Why:** Reproducible production behavior across modes. **Costs:** Per-op conformance and numerical envelope work. **Unresolved:** Floating matmul/reduction/transcendental envelopes and allowed accumulation order. **Owner:** TH-008; this blocks claiming those native operations equivalent, but not the specified exact `i64` addition first slice.

## TH-LD-021 — Stability and ABI

**Question:** What can users rely on? **Chosen:** Versioned project-owned specification, stable editions and semver, staged experiments, migration/deprecation, conformance, versioned artifact metadata; no native ABI stability before an ABI exists. **Alternatives:** Stable-by-default pre-alpha syntax, perpetual ABI promise, undocumented compiler behavior. **Why:** Avoid recurring rewrites without promising impossible binary compatibility. **Costs:** Release governance and compatibility testing. **Unresolved:** First stable-edition gate, artifact schema, native C ABI. **Owner:** TH-009 and TH-010.

## TH-LD-022 — UX and benchmark evidence

**Question:** How should users enter the language and assess speed? **Chosen:** One coherent `thiran` command family and low-ceremony first program/model/training/build path; performance claims freeze workloads, quality, hardware, precision, transfers, compilation/cache, quantiles, and memory by category. **Alternatives:** Many unrelated compiler/runtime tools, “HFT speed” as target, kernel-only performance claims. **Why:** Student accessibility and reproducible production evidence. **Costs:** CLI/runtime packaging and benchmark harness. **Unresolved:** Exact verb spelling, first installer/REPL implementation, named benchmark suites. **Owner:** TH-009 and TH-010.

## TH-LD-023 — Nerivu separation

**Question:** Does Nerivu define Thiran syntax? **Chosen:** No; Transformer, recurrent, state-space, hybrid, and future mechanisms use general functions, explicit state, AD, and native execution. **Alternatives:** Nerivu-specific operations/syntax and compiler special cases. **Why:** Keeps model research independent and tests language generality. **Costs:** General primitive extension and state performance must stand on their own. **Unresolved:** Specific Nerivu mechanism/integration workload, owned by separate Nerivu research after general Thiran capabilities exist. **Owner:** Separate Nerivu program; Thiran capability follow-through TH-007 through TH-010.

## Open decisions and blockers audit

The following details remain open; none reverses zero-based/half-open indexing, non-consuming immutable value aliasing with explicit move/copy/exclusive mutation, tensor value/view semantics, structured IR authority, or cross-mode semantic equivalence.

| Open item | Blocking owner and reason | Blocks first native vertical slice? |
| --- | --- | --- |
| Generic constraints, closures/higher-order captures, public effect spelling, full operator precedence | TH-004/TH-005: surface and checker design need examples and conformance. | No; first slice uses typed simple functions. |
| Dynamic rank, named axes, complex/half dtype, specialized matmul/reduction/contract APIs and envelopes | TH-008: numerical workload evidence and per-op tests are needed. | No for elementwise slice; yes before claiming those operations equivalent. |
| Negative-stride views, device view support, disjoint mutable split, GPU event/cancellation and AD save/recompute rules | TH-006/TH-007/TH-008: verifier and runtime design must enforce lifetimes. | No for synchronous CPU owner/read view slice. |
| Error taxonomy and propagation syntax; state/checkpoint and derivative APIs | TH-005/TH-007/TH-008: library and transformed-program conformance needed. | No for checked bounds/simple Result. |
| Package manifest/registry, stable-edition gate, artifact schema, public native ABI | TH-009/TH-010: compatibility tests and release design needed. | No for local matched compiler/runtime slice. |
| LLVM/MLIR/BLAS/CUDA selection, exact CLI verbs, named benchmark suites, Nerivu mechanism | TH-008/TH-010 or separate Nerivu research: choose from evidence, not premise. | No for minimal native CPU slice. |

**First native vertical slice blocker count from TH-001 language decisions: 0.** Its core conventions and IR boundary are fixed. Implementing it still depends on later checkpoint work and TH-002 baseline correctness; this is not permission to begin that work during TH-001.
