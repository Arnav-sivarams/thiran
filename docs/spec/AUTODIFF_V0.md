# TH-010 reverse-mode automatic differentiation

TH-010 places requested differentiation after verified typed structured semantic IR and TH-006 ownership/effect analysis, and before future TensorRegion extraction. `ReverseModeRequest` names one semantic FunctionId and an ordered nonempty list of parameter indices. This is an explicit compiler request, not implicit recording and not final source `grad(...)` syntax; the user-facing training/language API is deferred.

## Domain and eligibility

The explicit differentiability predicate accepts exactly `f32`, `Tensor<f32,1>`, and `Tensor<f32,2>`. Cotangents have the same semantic type and runtime shape. Bool, integer, Buffer, tuple results, other dtypes, and other ranks are not differentiable. There is no integer-to-float promotion.

An eligible function is straight-line, import-free, independently verified, and has only Read parameters. Its TH-006 summary may be Pure or Pure+MayTrap. Mutates, RNG, IO, Transfer, and Async reject. Structured if/for/while, break/continue, rebind/mutation, MutableBorrow/Consume, copy/move, index/slice/view operations other than rank-two transpose, ordinary calls, tuple output, and unsupported operations reject deterministically. Forward Checks remain ordered primal semantics; MayTrap alone is legal. Control-flow differentiation must later follow the executed path and iteration count and is not claimed here.

## Generated contract

The deterministic transform emits an ordinary typed semantic module with two functions:

```text
F.__th010_forward(original parameters)
  -> (primal result, saved value 0, saved value 1, ...)

F.__th010_backward(saved values..., output cotangent)
  -> requested input cotangent(s), in request order
```

Both functions have deterministic FunctionIds, names, ValueIds, shape facts, source spans, and generated provenance (`sourceFunction`, `forward`/`backward`). Both pass the ordinary semantic verifier and are executable by the reference evaluator. Multiple gradients use an internal tuple result. The layout is unstable compiler ABI.

There is no global or thread-local tape, recording mode, evaluator callback, or ValueId-keyed runtime gradient state. The forward tuple and flattened backward arguments are ordinary explicit value flow. A temporary map is used only while constructing explicit backward instructions.

## Saves and ownership

The baseline saves every differentiable primal required by a derivative rule and each requested input needed as a `ZeroLike` reference. Saves are deduplicated and sorted by primal ValueId. Metadata records slot, primal ValueId, type, shape, reason, TH-006 provenance kind, and ResourceIds. A tensor save is a retained logical read alias, never an implicit deep copy. The explicit forward tuple keeps the saved handle alive until the caller invokes backward. TH-010 only admits nonmutating functions, so it does not claim the future complete SavedForBackward-versus-mutation checker; the retained ResourceIds are the hook for that checker.

## Reverse rules and accumulation

The output cotangent is the caller-supplied seed. Instructions are visited in reverse program order. Every operand contribution is an ordinary typed instruction; a second and later contribution emits ordinary `Add`, so repeated uses are visible in generated IR.

- Add: reduce `dz` to each operand shape.
- Subtract: reduce `dz` to the left shape and reduce then negate for the right.
- scalar Multiply: `dx = dz*y`, `dy = dz*x`.
- elementwise Multiply, including tensor/scalar scaling: multiply by the saved other operand, then reduce to the operand shape. The scalar cotangent therefore reduces all tensor elements.
- rank-two Matmul: `dA = dY * transpose(B)` and `dB = transpose(A) * dY`, using ordinary Matmul and Transpose instructions.
- rank-two Transpose: transpose the cotangent.
- constant-axis Sum: `BroadcastToShape(dY, X, axis)` inserts the removed axis and broadcasts across its original extent.
- `stop_gradient(x)`: primal identity, no contribution to `x`.

Three generated-only core operations are ordinary semantic IR, not evaluator opcodes hidden from lowering. `ZeroLike(reference)` returns same-type/same-shape f32 zeros. `ReduceToShape(value, reference)` sums trailing-broadcast dimensions back to the reference shape, including scalar reduction. `BroadcastToShape(value, reference, axis)` reverses a rank-dropping Sum. Their type/rank contracts are verified and runtime shapes are checked. Future native lowering must either implement them or reject them.

## Execution and f32 contract

`executeVjp` evaluates generated forward, extracts its explicit tuple, accepts an explicit output cotangent of exactly the output type/shape, and evaluates backward only after forward succeeds. Tensor outputs are never implicitly seeded. `executeGrad` requires scalar f32 output and supplies exactly `1.0f`.

Reference f32 uses a host `float` only when it is four-byte IEC 559 binary32 and the active rounding mode is round-to-nearest. Operations remain in semantic program order; matrix multiply forces a separately rounded binary32 product and addition and no fast-math/reassociation is enabled. Overflow, signed zero, infinity, and NaN follow the frozen IEEE behavior. This is a qualified host assumption, not a portable decimal serialization claim. Observations use enough decimal digits to round-trip finite float values and spell nonfinite values deterministically.

Finite differences are test-only independent audits. TH-010 uses central differences with epsilon `1e-3` for selected scalar, elementwise, and 2x2 matmul losses, with `0.002`, `0.01`, and `0.03` absolute tolerances respectively. Reverse AD never invokes finite differences.

## Verification and boundaries

The AD audit reports ADV01 invalid WRT ownership, ADV02 nondifferentiable WRT, ADV03 gradient type mismatch, ADV04 unknown saved primal, ADV05 duplicate/missing save identity, ADV06 unavailable backward save, ADV07 malformed cotangent/generated semantic IR, ADV08 invalid ReduceToShape, ADV09 invalid BroadcastToShape, and ADV10 missing/misordered requested gradient. The ordinary verifier independently rejects malformed helper operations and generated provenance.

The TH-008 native backend remains STRICT_NATIVE and rejects f32 and AD helpers as `BACKEND-UNSUPPORTED`, with `fallback: NONE`. Reference gradient execution is not native gradient execution. No production CLI command is added. Source `stop_gradient(expr)` is an intrinsic and cannot resolve to a user function; dynamic `no_grad` is deferred.

Future custom derivatives must identify the primal operation/function, differentiable parameters, saved values, derivative program, effect restrictions, and backend requirements. No model-specific operation is built in. TH-010 is first-order reverse mode only: it claims no training loop, optimizer, control-flow AD, forward mode, grad-of-grad, Hessian, Jacobian materialization, or higher-order guarantee.
