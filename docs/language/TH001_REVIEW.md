# TH-001R adversarial semantics review and freeze

**Scope:** Future-language design fixtures, not executable current Thiran source. This review changes no compiler, optimizer, parser, IR implementation, or test. `CONSTITUTION.md`, `SEMANTICS_V0.md`, `COMPATIBILITY.md`, `ARCHITECTURE_NEXT.md`, and `DECISIONS.md` remain the normative TH-001 design documents; this file records the challenge and rationale. The target is a safe, concise numerical systems language for supported Python + NumPy + PyTorch workflows, research, native production, and selected numerical rendering—not compatibility with those languages.

## Challenged choices

### 1. Tensor assignment and ownership

- **Current TH-001 choice:** Move-by-default for owning tensors, buffers, and records containing them; `copy` for independent storage.
- **Strongest competitor:** Non-consuming immutable logical value aliases; exclusive mutation; explicit `copy` and consuming `move`; no hidden copy-on-write. **Chosen: REVISE.** A physical transfer may implement an alias after source last use, but source meaning does not change.
- **Program:** `let A = [1, 2; 3, 4]; let B = A; let C = A + B; print(C)`. Both `A` and `B` remain readable; binding `B` is O(1) at source, not an O(n) copy. `C` is a new logical result. If `A` dies, its handle can be physically transferred. `let mut A = ...; let B = A; A[0,0] = 7; print(B)` is rejected while `B` is potentially live; `let mut A = copy(B)` is independent and mutable.
- **Safety:** Multiple immutable handles to one storage are safe. Static alias/liveness analysis rejects exclusive access if another live handle, borrowed view, AD save, or async reservation can observe the storage. An explicit move consumes its binding, not surviving aliases. No runtime copying rescues illegal mutation. Safe views cannot outlive their source handle; GPU completion retains physical storage. FFI consumption needing uniqueness requires proof or an explicit copy.
- **Ergonomics:** Move-by-default makes the first matrix exercise fail and propagates ceremony through helpers and nested models. Alias-by-default preserves ordinary mathematical reading, while mutation is visibly exceptional. A diagnostic should name the conflicting alias and its last use, not lecture about physical buffers.
- **Performance:** Move-by-default gives cheap unique ownership and easy buffer reuse, but explicit copies of large tensors may be forced merely to reuse a source binding. Alias-by-default avoids hidden O(n) copies; sharing may require a handle/reference count when values escape, across dynamic call boundaries, or across async GPU events. Static last-use and uniqueness proofs can erase bookkeeping in hot paths, but universal zero counting is not guaranteed. Atomic counting, delayed GPU frees, AD retained values, and FFI ownership conversion are latency risks to measure. Alias-by-default may reduce in-place reuse while a live alias exists; that cost is observable as inability to mutate, not a surprise allocation.
- **Implementation:** Requires path-sensitive enough storage-identity/liveness analysis for tuples, records, calls, closures, AD saves, and events; a runtime retention scheme for escaping values; and storage-IR verification. Move-by-default is simpler to implement but mismatched to the product target. There is no hidden copy-on-write fallback.
- **Affected IDs:** TH-LD-009, TH-LD-010, TH-LD-011, TH-LD-018.

### 2. Function parameter modes

- **Current TH-001 choice:** Ordinary parameters are passed by value/move unless declared with a visible borrow mode.
- **Strongest competitor:** Annotated `Tensor<T,R>`/`Buffer<T>` parameters are non-consuming read-only by default; explicit mutable and consuming modes; returned views have explicit caller-tied lifetimes. **Chosen: REVISE.** Typed IR still represents every read, escape, mutation, and consumption.
- **Program:** `fn normalize(x: Tensor<f32, 2>) -> Tensor<f32, 2> { return x ./ norm(x) }`; `let y = normalize(A); let z = normalize(A)`. The calls do not consume `A`. A helper may return a tensor logical value alias to an input; it may not return a borrowed view from a local handle. A function that updates `params` must declare mutable access; one that hands a tensor to a consuming FFI must declare consumption.
- **Safety:** Read-only calls cannot mutate through a hidden reference. Escaping alias-valued returns retain storage; borrowed-view returns carry an explicit caller lifetime. Mutable/consuming modes remain part of public signatures and checker facts.
- **Ergonomics:** Normalization, nested layer calls, loss helpers, and new-value results need no repeated `borrow` syntax. Optimizers and native ownership-transfer APIs remain visibly exceptional. Persistent model state may be read and returned as a next-state logical value without consuming the old state, unless a declared consuming fast path is selected.
- **Performance:** Default reads can be lowered to non-retaining borrows when the value cannot escape, or retained aliases when it does. The latter may cost handle operations; copies are never inserted just for a call. Consuming modes can enable uniqueness and buffer reuse.
- **Implementation:** Mode-aware call signatures, escape analysis, alias-valued return tracking, view lifetime checking, FFI wrapper verification, and IR read/access facts. Exact surface spelling is staged, not the distinction.
- **Affected IDs:** TH-LD-011, TH-LD-012, TH-LD-016, TH-LD-018.

### 3. Dtype conversion and promotion

- **Current TH-001 choice:** No promotion between already typed mixed operands; contextual literals only if exactly representable.
- **Strongest competitor:** A small deterministic, language-owned lossless lattice (for example `i32 -> i64`, `u32 -> i64`, `f32 -> f64` where valid), never copying NumPy/PyTorch tables. **Chosen: REVISE contextual real-literal typing only; KEEP no promotion between typed operands.** Untyped decimal real literals round once under a specified backend-independent rule in an `f32`/`f64` context. Untyped integer literals still must fit exactly.
- **Program:** `let weights: Tensor<f32,2> = ...; let scaled = weights .* 0.1` is legal with a contextually rounded `f32` literal. `let factor: f64 = 0.1; weights .* factor` is rejected until `factor` is explicitly cast with a declared mode. `i64_index + f32_value` is rejected; indexing and numerical value arithmetic should not be conflated. A typed `i32 + i64` also requires a cast in V0.
- **Safety:** Literal typing is not silent narrowing of an already typed value. No conversion changes device, shape, rank, or storage. Explicit casts state overflow/rounding modes. Integer overflow remains checked.
- **Ergonomics:** Exact-only real literal typing would reject nearly every useful `f32` decimal constant; contextual rounding fixes that. Typed mixed data preprocessing still needs explicit conversion, but the cast exposes a real policy choice. Reductions keep their input dtype/declared accumulation contract rather than silently promoting.
- **Performance:** Contextual literal rounding is compile-time. A partial promotion lattice adds runtime casts and can change kernel dtype; refusing it keeps cost and latency visible. A lattice cannot losslessly combine every important pair (`i64` with `f32`/`f64`, or `i64` with `u64`) into this initial scalar set, so its partial behavior would still surprise beginners.
- **Implementation:** Decimal-to-binary conversion needs exact specification and conformance across compiler/reference/native backends. Typed-operand checking stays simple; future mixed precision is an explicit library/contract decision.
- **Affected IDs:** TH-LD-002, TH-LD-003, TH-LD-020.

### 4. Rank and shape

- **Current TH-001 choice:** `Tensor<T,R>` has static rank, generally runtime extents; dynamic-rank existential deferred to TH-008.
- **Strongest competitor:** Dynamic rank on every ordinary tensor, avoiding specialization barriers for ingestion. **Chosen: KEEP common static rank; clarify existential gate.** A rank-polymorphic generic specializes on known rank, whereas data-dependent rank needs an inspected existential before claiming complete dataset workflows.
- **Program:** `fn encode(x: Tensor<f32,3>) -> Tensor<f32,3>` accepts variable batch/token extents; `fn reduce_all<T,const R>(x: Tensor<T,R>)` handles specialization-known rank. An image loader with data-dependent spatial extents still returns rank 3. A generic file loader whose rank is determined by file metadata returns a dynamic-rank existential and requires a checked rank match before a rank-specific model call. Ragged sequences use an explicit representation (offsets/mask/padding), not a fake rectangular tensor.
- **Safety:** Rank mismatch is statically rejected for known ranks; runtime extents and inspected existential dispatch are checked. Zero axes remain valid.
- **Ergonomics:** Students and model authors use ordinary fixed-rank matrices/images/token batches; ingestion pays dispatch complexity only where data truly varies in rank. Model parameters and explicit state remain fixed rank with runtime extents/shape contracts.
- **Performance:** Static rank enables specialization and fixed loop nests; runtime extents still permit dynamic images/batches. Existential dispatch has a bounded boundary cost, not a tax on every tensor.
- **Implementation:** Rank-polymorphic generic constraints and an eventual checked existential API, required before a full ingestion claim but not the first native fixed-rank slice.
- **Affected IDs:** TH-LD-004, TH-LD-005.

### 5. Failure categories

- **Current TH-001 choice:** Spanned compile errors; `Result`/`Option`; ordinary bounds/shape/overflow precondition failure; resource exhaustion grouped with panic.
- **Strongest competitor:** Every indexing/arithmetic/allocating operation returns `Result`, making all failures recoverable by default. **Chosen: REVISE taxonomy, not ordinary operator syntax.** Distinguish static rejection, checked runtime contract failure, recoverable API, and resource/runtime failure.
- **Program:** `A[999]` with a runtime extent fails a checked bounds contract at use; `get(A,999)` yields `None`. A provably invalid constant index can instead be rejected statically. `A + B` with dynamic incompatible shapes fails a checked shape contract; a `try_add`-style API returns an error. `i64_max + 1` fails checked overflow at runtime if the operands are dynamic and is rejected if provably constant; checked-result arithmetic returns an error. File/parse/transfer/device/checkpoint operations return `Result`. An allocation-sensitive production API returns an allocation error; a plain allocating numerical expression may terminate with a resource diagnostic rather than pretending the source violated a shape rule.
- **Safety:** None of these categories permits undefined memory behavior. Device work must report failure on observation and cannot yield a valid-looking stale tensor.
- **Ergonomics:** Simple matrix work keeps ordinary operators. Production services select result-returning variants at failure boundaries; first-year diagnostics say what failed and why.
- **Performance:** Checked guards have cost but are mandated; result variants need explicit control paths. Device observation and allocation handling cannot be silently optimized away.
- **Implementation:** Error taxonomy, propagation spelling, result-returning operation variants, resource diagnostics, and async failure retention/observation tests remain follow-through work.
- **Affected IDs:** TH-LD-014, TH-LD-020.

### 6. Buffer versus Tensor

- **Current TH-001 choice:** Numeric/bool `Tensor<T,R>` and general CPU `Buffer<T>` are separate.
- **Strongest competitor:** One general array type, with tensor operations implemented for numeric element types. **Chosen: KEEP distinction with a narrower semantic role.** `Buffer` is CPU-addressable record/byte systems storage, not a bypass for missing tensor features.
- **Program:** A dataset file's raw bytes and variable-length geometry records live in `Buffer<u8>`/`Buffer<Vertex>`; a decoded numeric batch and model weights live in `Tensor<f32,R>`; a CPU framebuffer of packed numeric pixels may be `Tensor<u32,2>`, while structured pixel records may use `Buffer<Pixel>`. Numeric FFI uses a checked tensor descriptor; byte-oriented FFI uses a buffer contract. Conversion is explicit.
- **Safety:** Both obey alias/exclusive mutation rules; Buffer does not offer an unsafe shortcut around tensor bounds or device ownership.
- **Ergonomics:** Mathematical arrays retain concise tensor notation; records/bytes do not pretend to support matrix arithmetic. General CPU Buffer must not be prescribed simply because Tensor lacks an array feature.
- **Performance:** Byte/record storage and CPU FFI avoid tensor metadata/device abstraction when irrelevant. Numeric data should not be routed through Buffer to evade tensor costs.
- **Implementation:** Distinct type/operation contracts and explicit conversion/FFI descriptors; avoid maintaining two competing numeric APIs.
- **Affected IDs:** TH-LD-004, TH-LD-010.

## Seven scientific-programming probes

These sketches specify intended semantics, not current parser syntax.

**E1 — Basic numerical use.** `let A = [1, 2; 3, 4]; let B = A; let C = A + B; print(C)`. `A` and `B` share immutable logical values/storage; neither binding is invalidated or deep-copied. `C` has the declared checked elementwise shape/dtype contract and a new logical value. Printing reads; after the final use the compiler can reuse storage only if the output and earlier aliases remain observationally unchanged.

**E2 — Read-only helper.** `fn normalize(x: Tensor<f32,2>) -> Tensor<f32,2> { return x ./ norm(x) }`. The parameter is read-only and non-consuming without call-site ceremony; result is a new logical tensor. The helper cannot mutate `x`; decimal constants in `f32` expressions are contextually rounded.

**E3 — Explicit mutation.** `let mut A = ...; let row = A[0,:]; A[0,0] = 5; use(row)` is rejected because `row` can observe the same storage after the write. `let mut B = copy(A); B[0,0] = 5` is legal because `B` has independent storage. If an alias/view is dead before the write and no AD/event use survives, exclusive access may be proven legal. No copy-on-write occurs.

**E4 — Slice/view.** `let A = ...; let row = A[0,:]; let y = reduce(row)`. `row` borrows a live `A` handle with rank one, checked bounds, and legal strides; it allocates no independent storage. `A` and its storage stay live until `row`'s last use; `y` is a new value/scalar according to reduction contract. A borrowed row cannot escape a function if `A` was local.

**E5 — Stateful model.** `fn step(x: Tensor<f32,2>, state: State, params: Params) -> (Tensor<f32,2>, State) { let y = layer(x, params); let next_state = update_value(state, y); return (y, next_state) }`. All unmarked inputs are non-consuming reads. Old `state` remains usable, `next_state` is a new logical value (possibly sharing unchanged immutable components), and `params` remains read-only. A consuming state-transition fast path would be declared `move` and its caller would explicitly choose it; persistent runtime storage and outstanding device events retain handles.

**E6 — Training.** `for batch in loader { let loss = model(batch, params); let grads = grad(loss); optimizer.step(borrow mut params, grads) }`. Model/grad reads do not consume `batch` or `params`; the optimizer's update is explicitly exclusive. If AD saved a parameter value or an async model read remains live, the update is rejected/deferred by the declared completion contract; `copy(params)` or a value-returning update creates independent next parameters. This is not a final training API.

**E7 — Rendering/numerical loop.** `let geometry = ...; let mut frame = zeros<u32>(height,width); for y in 0:height { for x in 0:width { frame[y,x] = shade(geometry, x, y) } }`. Geometry is a shared immutable read; `frame` is exclusively written. No read view of `frame` may remain live through a conflicting write. An indexed loop or checked disjoint write API permits normal framebuffer work; a blanket borrow of all arrays for the loop would be a design defect. Device output requires an explicit transfer/async effect.

## Five programs we expect students to write

All are compact semantic fixtures, not syntax implemented at HEAD. Library names are illustrative, not frozen APIs.

1. **Matrix numerical exercise:** `let A = [1,2;3,4]; let B = A; let C = A * B; print(sum(C,0))`. Read aliasing and ordinary read helpers need no ownership ceremony; matmul shape is checked.
2. **Linear regression:** `fn predict(x: Tensor<f32,2>, w: Tensor<f32,2>) -> Tensor<f32,2> { return x * w }; let yhat = predict(x,w); let error = (yhat - y) .* (yhat - y); let loss = mean(mean(error,0),0)`. Inputs stay readable; repeated `yhat` is legal, and rank-dropping reductions produce a scalar loss.
3. **Small MLP training:** `for batch in loader { let loss = mlp_loss(batch, params); let grads = grad(loss); optimizer.step(borrow mut params, grads) }`. The only visible ownership mode is the meaningful update; AD/device reservations must end before it.
4. **Explicit-state sequence step:** `let mut state = initial_state; let (y,next_state) = step(token,state,params); inspect(state); state = next_state`. Rebinding a mutable state handle does not mutate the old state's storage; unchanged state components may share read-only storage.
5. **Small numerical rendering loop:** `let mut frame = zeros<u32>(h,w); for y in 0:h { for x in 0:w { frame[y,x] = shade(vertices,x,y) } }; save(frame)`. Geometry reads and exclusive framebuffer writes coexist; file failure is recoverable.

## Architectural and freeze verdict

The revised semantics fit `Source -> AST -> typed structured semantic IR -> ownership/alias/effect analysis -> requested AD -> tensor/dataflow regions -> kernel/loop IR -> storage/buffer IR -> native backend/runtime`. Logical value identity and alias/mutation rules are source semantics; reference counts, physical moves, buffer reuse, event retention, and FFI descriptors are storage/runtime obligations. AD saved values and async uses are explicit alias/lifetime constraints, not hidden snapshots. The typed IR remains authority; pure regions are extracted only when effects and aliases permit it.

**Kept without reopening:** static checking; zero-based/half-open indexing; rank-2 matrix `*` and elementwise `.*`; structured control flow; no tensor truthiness; typed structured IR authority; lower-level pure tensor regions; explicit state/effects; compiler-transformation AD; Python/PyTorch non-normative; Nerivu separation; stable-edition discipline; no promotion between already typed operands; common static tensor rank; Buffer/Tensor distinction.

**Revised foundational decisions:** TH-LD-003 (contextual decimal real literals), TH-LD-010 (non-consuming aliases/explicit move), TH-LD-011 (shared-storage lifetime and exclusivity), TH-LD-012 (read-only parameter default), TH-LD-014 (resource failure distinct from contract failure). Other ID text clarifies consequences but does not reverse its chosen boundary.

**Remaining nonblocking specification questions:** exact exceptional-mode spelling; alias-valued return and borrowed-view lifetime signature syntax; storage-handle elision proofs and measured reference-count/async latency; checked mutable disjointness API; decimal conversion and numerical envelopes; dynamic-rank existential/shape dispatch; result propagation/error taxonomy; FFI uniqueness and GPU event cancellation contracts. These must be settled and conformance-tested before relevant features are stable, but they do not invalidate the first fixed-rank CPU slice design.
