# TH-011 deterministic reference training

TH-011 composes verified V0 semantic IR, TH-006 ownership/effect facts, and TH-010 generated reverse-mode IR into one explicit reference-runtime state transition. It does not add source syntax, a hidden compiler loop, native f32 execution, or a production training API.

```text
semantic scalar loss + explicit trainable indices
  -> validated reusable TrainingPlan
  -> current TrainingState + non-trainable runtime inputs
  -> generated forward -> scalar loss + explicit saves
  -> generated backward(seed = 1.0f) -> ordered gradients
  -> checked optimizer transition -> new TrainingState
```

Reference training != native training. Training state != a hidden global parameter store. Optimizer update != AD. The optimizer result is a new logical state. A training loop != control-flow AD.

## Parameter identity and plan

`ParameterId` is a deterministic integer identity for one optimizer-visible source parameter in one plan. It is derived from semantic function identity and source parameter index. It is not a BindingId, ValueId, ResourceId, StorageObjectId, address, or storage handle. Each ordered `ParameterDescriptor` retains ParameterId, source index, source name, semantic type, rank/shape facts, and plan position. Trainables are selected only by the ordered indices in `TrainingPlanRequest`; f32 type or names such as `weight` never imply trainability. Plan order, including a deliberate order such as `[bias, weight]`, governs gradients, velocities, and updates.

Plan construction independently verifies semantic IR, runs TH-006 ownership/effect analysis and fact audit, requires a scalar f32 result, validates optimizer hyperparameters, issues one TH-010 `ReverseModeRequest`, audits generated AD metadata/IR, and qualifies generated forward/backward ownership/effects. Unknown/duplicate indices, unsupported type/rank, exceptional access, and unsupported AD reject before a plan exists. A successful plan records developer-only `generationCount = 1`; steps reuse its source, facts, and generated IR without parsing, semantic analysis, or differentiation.

The implementation exposes plan fields only so tests can independently corrupt and audit developer metadata. A successfully returned plan is immutable by contract. Its deterministic signature contains semantic structure, function, ordered descriptors, and optimizer specification; it contains no address.

## State and runtime inputs

`TrainingState` explicitly contains the plan signature, step, ordered parameter values, and `OptimizerState`. Each parameter state repeats ParameterId/type/concrete shape so compatibility is checked rather than inferred from vector position. Parameters are logical f32 scalar, `Tensor<f32,1>`, or `Tensor<f32,2>` values. Initialization validates each value and creates momentum velocities with explicit zero-like values. Equivalent inputs create equivalent states.

Every step validates plan signature, count, ParameterId/order, dtype, rank, concrete shape, optimizer kind/step, and velocities before loss evaluation. No cast or reshape is inserted. Non-trainable `RuntimeInput` values carry source parameter indices. The step rejects missing, duplicate, out-of-range, type/shape-mismatched inputs and any caller value for a trainable slot. It assembles the full source argument vector by source index, never by a name heuristic.

## One-step transaction and gradient contract

`trainingStep(plan, state, inputs)` performs exactly one logical transition:

1. audit the reusable plan and validate state/input compatibility;
2. execute the TH-010 generated forward once;
3. retain its scalar loss and explicit save tuple;
4. execute generated backward with scalar binary32 seed `1.0f`;
5. unpack gradients in plan order and attach ParameterIds;
6. require each gradient's exact dtype, rank, and concrete shape;
7. build all optimizer outputs in temporary new values;
8. publish a complete next state and increment both state/optimizer step once.

Forward, backward, gradient, or optimizer failure returns no successful next state. The input state is passed by const reference and is never mutated. Updating a parameter produces a new logical runtime value; it is not source binding mutation, storage mutation, or hidden copy-on-write. `applyOptimizer` exposes the same checked transition separately for independent optimizer and malformed-gradient tests.

An unused requested trainable retains its explicit TH-010 `ZeroLike` result and its plan position. Plain SGD therefore leaves it numerically unchanged. Momentum may still update it from an existing velocity.

## Saved values, ownership, and effects

The TH-010 generated forward returns ordinary `(loss, saves...)` value flow. The forward tuple remains live through its matching backward invocation. After backward returns or fails, that step no longer retains the tuple: this is the first concrete `SavedForBackward` lifetime boundary. Optimizer replacement begins only after successful backward and gradient validation, so it cannot conflict with saves from the same synchronous step. No asynchronous backward or overlapping-step safety is claimed.

Saved primal values retain TH-006 ResourceId provenance where available. A current parameter, saved primal read alias, gradient logical value, and optimizer result remain distinct semantic concepts even when their types/shapes match. Training is a runtime/library state transition and is not labeled Pure. The loss and generated backward keep their independently analyzed Pure/MayTrap facts; no fake IO/RNG/mutation effect is injected into source IR. Future source-level optimizer mutation needs an explicit state/effect contract.

## Determinism and fixture

For identical plan, initial state, runtime inputs, and qualified host binary32 assumptions, there is no RNG and the sequence is deterministic. The bounded end-to-end fixture is actual V0 source implementing `prediction = x * weight + bias`, squared error, and two explicit sums to scalar. Its batch is `x=[[0],[1],[2],[3]]`, `target=[[1],[3],[5],[7]]`; initial `weight=[[0]]`, `bias=[0]`; plan order is `[bias, weight]`.

At the initial state the exact mathematical loss is 84, `dLoss/dbias=[-32]`, and `dLoss/dweight=[[-68]]`. Plain SGD with learning rate 0.02 yields `bias=[0.64]`, `weight=[[1.36]]`. The predeclared convergence run uses 200 identical full-batch steps and absolute parameter tolerance 0.001 around bias 1 and weight 2. This is deterministic reference evidence, not a numerical-stability or performance claim. A tensor argument may already represent a mini-batch; TH-011 adds no Dataset, DataLoader, shuffle, iterator, or file ingestion.

## Boundary

`thiran-v0 check` continues to validate eligible loss source. `thiran-v0 build/run` remain STRICT_NATIVE and reject unsupported f32/matmul/AD with `BACKEND-UNSUPPORTED` and `fallback: NONE`; they never fall back to this evaluator. There is no public training CLI. Tests may drive an explicit host `for` loop over `trainingStep`, but no compiler training loop or control-flow AD is implied.

No final source spelling for `grad`, parameter discovery, optimizer step, or train is frozen. Native/GPU execution, mixed precision, distributed training, checkpoints, dataloaders, source structs/models, arbitrary module linking, asynchronous/overlapping steps, and production optimizer APIs remain future work.
