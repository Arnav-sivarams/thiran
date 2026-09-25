# Native model deployment V0

TH-017 connects the TH-011 reference model to TH-016 native artifacts through
an explicit frozen deployment state. The supported reference lifecycle is:

```text
TH-011 TrainingPlan/TrainingState
  -> transactional .thc checkpoint
  -> destroy original state
  -> validated checkpoint reload
  -> immutable DeploymentSnapshot
  -> embedded TH-016 CPU/GPU artifacts
  -> .thm model bundle
  -> thiran-model native execution
```

Training uses TH-010 generated forward/backward IR and TH-011 optimizer
transitions. Deployment uses neither the training loop nor the reference
evaluator.

## Shared semantic model

The reference source contains the accepted TH-011 affine loss and an inference
entry in the same analyzed module. The loss computes
`sum(sum((x * weight + bias - target) .* error, 0), 0)`. The inference entry
selects the only value from the trained `Tensor<f32,2>` weight, projects it to
`Tensor<f32,1>`, multiplies one public `Tensor<f32,1>[1]` input, and adds the
trained `Tensor<f32,1>` bias. Both entries therefore share the same parameters
and model definition; inference constants are not substituted for learned
values.

## Deployment snapshot

The only public snapshot constructor accepts a successful
`CheckpointLoadResult`, the verified plan, and an explicit deployment request.
It revalidates the restored state, derives ordered frozen parameter records,
and requires their digest to equal the checkpoint digest. The snapshot records
model and inference-entry identities, the public typed ABI, and deterministic
bindings from public inputs and `ParameterId` values to artifact input slots.
It contains no optimizer kind/configuration, velocity, training step, gradient,
backward save, or `TrainingPlan`.

For the affine model the public signature has one input while the artifact
signature has three:

```text
public input 0 -> artifact slot 0 (x)
weight ParameterId -> artifact slot 1
bias ParameterId -> artifact slot 2
```

Missing, duplicate, out-of-range, type-incompatible, or shape-incompatible
bindings are rejected. Vector position alone is never the parameter identity.

## Bundle and runtime

The little-endian `.thm` container uses `THIRANMB` magic, format version 0, and
model runtime ABI 1. Its integrity envelope contains the model/entry identity,
ordered frozen parameter table and digest, public input/result ABI, both
binding tables, backend declarations, and one or two complete TH-016 `.tha`
artifacts. Each embedded artifact carries the same model identity. A CPU-only
bundle is valid; a dual bundle may carry one CPU ELF and one GPU PTX artifact.

The loader bounds all strings, values, collections, and embedded artifacts;
checks the envelope digest; and delegates embedded artifact parsing and
integrity validation to the TH-016 loader. It rejects malformed versions,
identity/backend disagreement, duplicate backends/parameters/bindings,
missing/unexpected parameters, bad parameter digests, CPU/GPU signature
disagreement, public ABI mismatch, and parameter-to-artifact ABI mismatch.

`thiran-model run` requires explicit `--backend cpu` or `--backend gpu`. It
validates the public input, reconstructs artifact arguments from the public
binding and frozen parameter records, and calls the TH-016 artifact runtime.
It never silently changes backend. CPU execution loads the embedded ELF without
running a compiler. GPU execution loads stored PTX through the CUDA Driver,
which performs its allowed device JIT; Thiran does not regenerate PTX.

GPU submission reuses TH-014 `PendingGpuExecution`. Public input and frozen
parameter tensors, output reservation, module, stream, event, allocations, and
plan remain retained until observation or safe drop-drain. The synchronous CLI
is submit followed by observe.

## Planning, relocation, and scope

Artifacts are produced from the same verified inference `TensorRegion` using
TH-015 reuse/fusion options. The model bundle stores no absolute source, build,
artifact, or checkpoint path and remains executable after copying it beside the
runtime. The deployed process needs no model source, training checkpoint,
Thiran compiler, host compiler, Python, PyTorch, Triton, or semantic-evaluator
fallback. Ordinary native loader libraries are allowed; GPU execution requires
the CUDA Driver and a compatible device.

Bundles and embedded native payloads are trusted compiler outputs. Structural
checks detect corruption and incompatibility but provide no cryptographic
authenticity, malicious-code isolation, hostile-input sandbox, stable public
ABI guarantee, registry, service protocol, container platform, or performance
claim.
