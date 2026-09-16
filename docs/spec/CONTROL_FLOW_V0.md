# TH-005C V0 structured control flow

This document describes the isolated V0 frontend and semantic pipeline, not a stable edition or the production R12 CLI. The frozen `docs/language/` semantics remain authoritative.

TH-006 uses these ordered blocks for backward may-liveness and forward availability/resource-state analysis. Branch joins preserve any possible resource relationship and require definite binding availability. Return paths terminate before a following merge. Range and while loops solve deterministic back-edge fixed points for both future uses and potentially repeated moves; Break exits to the loop successor and Continue reaches the next condition/iteration. No loop unrolling or current constant-bound shortcut establishes general ownership legality. See [OWNERSHIP_V0.md](OWNERSHIP_V0.md).

## Source syntax and lexical scopes

Braced statements support `if condition { ... } [else { ... }]`, `for i in start:end { ... }`, `while condition { ... }`, `break`, and `continue`. `if` is a statement, not an expression. Conditions require scalar `bool`; tensor truthiness is rejected. Range bounds require `i64`. A colon after the first range-bound expression is grammatical in `for`, independent of type-annotation and tensor-selector colons. Blocks preserve TH-004 newline/semicolon separators; row semicolons inside tensor literals remain distinct.

Declarations belong to their immediate lexical block. A branch-local `let` or range induction variable is unavailable outside its block. A nested block may shadow an outer name with a distinct binding; duplicate declarations in one block are rejected. `break` and `continue` are legal only inside a loop and target the innermost enclosing loop. Same-block statements after unconditional `return`, `break`, or `continue` receive `TH005-AFTER-RETURN`; this is a narrow deterministic unreachable-code policy, not general path-reachability analysis.

## Structured IR and logical binding state

Each function has ordered steps in a root `Block`. `If` contains then and optional else blocks; `ForRange` contains a body block and an immutable induction BindingId; `While` contains a condition block and a body block. Each nested block has deterministic BlockId and parent BlockId. These are structured regions, not Graph nodes or backend jump offsets. Calls remain `Call`, tuples remain typed structured values, and runtime checks remain ordered `Check` steps.

TH-005's straight-line binding-to-current-ValueId map could not represent a path-dependent update. TH-005C therefore assigns a deterministic BindingId to each source binding (and parameter), separately from per-function ValueIds of expression results. `let` records a logical binding declaration and its incoming expression value; `x = value` records `BindingWrite` for the same mutable BindingId; identifier reads emit `LoadBinding` and receive a fresh expression ValueId at that program point. An immutable alias gets its own BindingId with the same initial logical value, not a deep semantic copy. A rebind replaces only that binding's current logical value; it does not mutate an earlier value or assume unique physical storage.

An `if` writes outer mutable slots only on its taken path. With no else, the false path retains the incoming logical value. A loop repeatedly writes the same outer slots; each iteration's `LoadBinding` observes the latest transition. This logical-slot model is chosen over premature join ValueIds because it directly preserves lexical identity and ordered transitions for later path-sensitive ownership/liveness, executed-path AD, recurrent/scan reasoning, and tensor-region extraction. It does not encode buffer allocation, storage identity, reference counting, or physical moves. Any later lowering into explicit join/block-argument form must prove equivalence to these source-level binding transitions.

## Execution

`if` evaluates its scalar condition once and executes one body. `for i in start:end` evaluates both bounds once, then visits successive signed `i64` values from `start` while `i < end`, in unit positive increments. Equal or reversed bounds execute zero iterations. Negative starts are not wrapped or converted to unsigned values; if used as tensor indices they remain subject to ordinary bounds checks. The evaluator never increments past the final representable signed value. `while` re-executes its condition region before every body iteration, so rebinding is visible to the next test. No control structure is compile-time-unrolled.

The evaluator uses explicit internal normal/Return/Break/Continue signals. Return exits the whole function from any nesting; Break exits the innermost loop; Continue starts its next iteration. A bounded loop-iteration guard reports `TH005C-STEP-LIMIT` for nonterminating/developer-hostile programs. This is evaluator tooling protection, not source-language loop semantics. Existing call-depth, total-step, and tensor-size guards remain. Canonical scalar/tensor/error observation still uses the TH-003 result boundary.

## Runtime checks and effect-order boundary

A runtime `Check` is an observable, trapping semantic operation in its actual ordered block. A bounds or shape Check inside a branch or loop executes only if that path/iteration executes. Hoisting a trapping Check across a control boundary is illegal without proof that failure behavior is unchanged. Ordered blocks also preserve a substrate for later mutation, RNG, I/O, transfers, and asynchronous effects; TH-005C does not define the full effect lattice.

## Deliberate boundaries

`for row in A` parses into a dedicated iterable-for AST form but semantic analysis reports `TH005C-ITERABLE-FOR-DEFERRED`. Borrowed leading-axis tensor-view iteration and its lifetime constraints belong to later ownership/view work; rows are not silently materialized as semantic copies. The exact evaluator may materialize existing indexing/slice results only at its observation boundary, as before; that does not validate borrowing.

Future pure tensor/dataflow regions can be extracted from an ordered block, branch, or loop after ownership/effect/AD obligations are checked. The entire structured function is not an acyclic tensor DAG. Structured semantic IR is the source-level authority; Graph/Region remain downstream, legal-region machinery only.

Control-flow support is not ownership safety, AD, native execution, or physical storage semantics. TH-007 [storage views](STORAGE_V0.md) provide a concrete leading-axis read-view substrate, but iterable-tensor for remains deferred until iteration-variable root/body lifetime facts can be represented and audited through all Break/Continue/Return paths without inventing returned-view syntax.
