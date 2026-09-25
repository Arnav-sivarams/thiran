# Training checkpoints V0

TH-017 adds a narrow persistent representation of a validated TH-011
`TrainingState`. A checkpoint is resumable training state, not deployment
state. It contains parameters, optimizer configuration and state, and step
metadata. It contains no source text, pointer identity, opaque host objects, or
framework serialization.

## Schema and container

`TrainingCheckpointSchema` is constructed from a verified `TrainingPlan` and
its step-zero initial state. It records a caller-supplied model identity, the
deterministic plan signature, and the exact ordered `ParameterId`, semantic
type, rank, and concrete shape table. A loader requires this expected schema
and the corresponding plan; file contents cannot define their own trusted
model schema.

The little-endian `.thc` container is:

```text
THIRANCP magic
format version (0)
training checkpoint ABI (1)
payload byte length
payload FNV-1a-64
model identity
training-plan signature
TrainingState step and OptimizerState step
optimizer kind, learning rate, and momentum
ordered-parameter digest
ordered parameter records
ordered optimizer-state records keyed by ParameterId
```

Each parameter record includes source-parameter index, plan position, stable
`ParameterId`, source parameter name, scalar/tensor kind, f32 dtype, rank,
shape, byte length, and the exact binary32 payload. Optimizer velocities use
the same checked value representation and are keyed by `ParameterId`. Plain
SGD requires no velocity records; SGD with momentum requires one compatible
record for every parameter. Step values must agree and the maximum `uint64_t`
value is rejected because it cannot be resumed.

The parameter digest covers the complete ordered parameter records, including
identity, name, type/shape, and value bytes. FNV-1a-64 detects accidental
corruption and gives deterministic identity; it is not cryptographic
authentication.

## Loading and validation

Loading is all-or-nothing. Bounds-checked readers reject bad magic, unknown
format or training ABI, truncation, trailing data, excessive lengths/counts,
payload-integrity failure, model or plan mismatch, missing/unexpected/duplicate
or reordered parameters, semantic identity changes, dtype/rank/shape mismatch,
malformed numeric byte lengths, optimizer mismatch, missing/unexpected or
malformed optimizer state, impossible steps, and parameter-digest mismatch.
Only after those checks and `verifyTrainingState` succeed is a new
`TrainingState` returned.

## Saving and recovery boundary

Saving first validates the plan, expected schema, complete state, exact shapes,
and resumable step. It serializes into memory, creates an exclusive
same-directory `<destination>.tmp`, writes it fully, calls `fsync`, closes it,
atomically renames it over the destination, and fsyncs the parent directory.
An existing temporary path causes failure before touching an existing valid
checkpoint. A failure after rename is reported explicitly as an installed file
whose directory durability could not be confirmed.

The checkpoint API neither claims hostile-input sandboxing nor executes data
from a checkpoint. Size limits and structural validation reduce accidental
resource abuse, but the format has no signature, encryption, access control,
anti-rollback protection, or stable cross-version compatibility promise.
