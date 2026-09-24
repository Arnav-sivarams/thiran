# TH-012 structured stateful scan

TH-012 adds finite recurrent execution to the typed V0 semantic pipeline. It is generic compiler/runtime infrastructure, not an RNN, LSTM, GRU, SSM, Transformer, Mamba, or Nerivu operation. The semantic authority remains structured IR; scan is not lowered into the legacy Graph and is not defined by timestep unrolling.

```text
scan(step, inputs, initial_state, captures...)
    -> (stacked_outputs, final_state)

step(input_element, current_state, captures...)
    -> (output_element, next_state)
```

`step` names one ordinary semantic FunctionId. Its body exists once and may contain ordinary structured control. Every parameter must have Read access. Captures are explicit invariant arguments, not closure or model-object state. The result is a value-producing `Structured::Scan` step containing the sequence ValueId, initial-state ValueId, capture ValueIds, step FunctionId, result ValueId, output-element type, and output shape contract.

## Iteration and collection

Inputs are `Tensor<i64,1/2>` or `Tensor<f32,1/2>`. Scan traverses the leading extent from zero upward. A rank-one input supplies scalar elements; rank two supplies borrowed rank-one logical elements. Runtime descriptor/type validation and the evaluator resource limit check the bound before unsafe extent arithmetic. The evaluator additionally applies its bounded iteration guard. There is no unbounded scan form.

Output elements are i64/f32 scalars or rank-one tensors. Scalars stack to rank one; rank-one tensors stack to rank two. Tensor output trailing extent must be a single, fully known invariant shape derived from all step returns. This structural contract makes zero-length output `[0, ...known trailing extents]` without executing a discovery iteration. Each executed output is checked immediately against the contract, and collection preserves iteration order.

State uses existing executable semantic values: i64/f32/bool scalars, rank-zero through rank-two i64/f32 tensors, and recursively composed tuples. Next state must have exactly the initial state type. Runtime tensor shapes are invariant at every iteration, recursively through tuple state. Scalar and rank-zero tensor remain distinct types and runtime representations. Unsupported represented-but-nonexecutable values such as Buffer state reject during scan analysis.

Zero length calls the step zero times, performs no step effect, returns the structurally typed empty output tensor, and preserves the initial state as final state. One or more iterations execute deterministically in leading-axis order. A failure in input extraction, step execution, output validation, or state validation stops immediately; no result tuple is published and no later iteration runs.

## Ownership and effects

The input element is a per-iteration immutable logical borrow. Output stacking materializes a fresh logical collection, so an output may read/alias the element during the step without retaining the borrow. The next state may alias the current state or explicit captures under ordinary immutable sharing, but it may not retain the input-element resource; TH-006 reports `TH012-INPUT-BORROW-ESCAPE`. Scan itself inserts neither Copy nor Move. An explicit `copy(state)` creates independent resource provenance, while explicit `move(state)` consumes the step-local state binding under the existing rules.

Final-state provenance conservatively preserves possible aliases of initial state and captures plus a possible fresh next state. Collected output provenance is Fresh. Existing view-root, move, rebind, mutable-borrow, and live-alias diagnostics continue to apply. The reference evaluator uses logical values and may materialize a leading row; this does not change the semantic borrow or require a source-level deep copy.

The enclosing function summary includes the step summary. Pure and MayTrap steps are supported. Mutation, RNG, I/O, transfer, and async effects beneath scan produce `TH012-UNSUPPORTED-EFFECT`. A scan is never marked as one pure tensor-region candidate, although pure numerical calculations inside its step remain available to a future region extractor.

## AD and backend boundaries

TH-010 only transforms straight-line functions and has no executed-iteration save representation, reverse structured traversal, call differentiation, or stack/unstack adjoints. Differentiating a scan therefore reports `AD-ELIGIBILITY-SCAN`; no numerical gradient, hidden tape, or framework fallback is fabricated. Recurrent AD must later generate compiler-owned structured forward/backward IR with explicit saved values.

TH-008 STRICT_NATIVE reports `BACKEND-UNSUPPORTED: Scan lowering deferred`, records `Scan unsupported-native`, and records `fallback: NONE`. Reference scan is not native scan. TH-012 adds no GPU work.

## Current limits

The source spelling is an internal V0 intrinsic, not a stable public language API. Inputs and stacked outputs are bounded to the currently executable rank/dtype subset. Rank-one tensor output needs a statically known trailing extent. Tuple output elements are not collected, Buffer state/captures are not executable, and tuple projection/destructuring is not added. Recurrent reverse AD, native CPU lowering, GPU lowering, async/effectful scan, and physical buffer planning remain unsupported.
