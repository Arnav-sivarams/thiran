# Robustness and reproducibility qualification V0

Status: frozen for TH-024 before machine qualification.

## Scope and claims

This specification qualifies the accepted TH-001--TH-017 and TH-019--TH-023
system under bounded repetition, corruption, controlled failure, concurrency,
resource churn, and fresh builds. It does not change language semantics or the
TH-023 performance workloads. It is not a security proof, a portability claim,
or evidence that arbitrary hostile native code is safe.

Determinism is reported separately as semantic, diagnostic, artifact semantic
identity, byte reproducibility, repeated execution, cross-configuration
semantic reproducibility, and cross-machine reproducibility. Stable semantic
digests do not imply stable container bytes. Cross-machine and cross-GPU claims
require genuinely distinct physical systems.

The canonical machine record is `robustness/results/TH-024-machine.json`.
`robustness/th024/qualify.py` replaces that record atomically after every
completed phase. A killed phase remains pending or interrupted; completed
phases and their evidence remain on disk. Temporary artifacts are created
outside the source tree and canonical accepted inputs are never mutated.

## Frozen parameters

- Generator seed: `240024`.
- Secondary differential seed: `240025`.
- Diagnostic repetitions: 20 per representative invalid input.
- Small CPU workload repetitions: 100 per selected workload.
- Small physical-GPU workload repetitions: 100 per selected workload.
- CPU and GPU artifact builds: 10 each, in distinct temporary directories.
- Deterministic generated fusion cases: 100.
- Deterministic reference/CPU/GPU differential cases: 100.
- Extension differential cases: 100.
- Extension registry load/freeze/destroy cycles: 100 successful cycles plus
  100 controlled failing loads.
- Host-safe async operations: 500. Physical-GPU async operations: 150.
- CPU soak: 1,000 iterations. GPU soak: 500 iterations.
- Concurrent CPU artifact runs, source checks, and independent builds: 4 each.
- Concurrent physical-GPU artifact runs: 2.
- Child-process address-space limit where used: 1 GiB; CPU time limit: 120 s;
  open-file limit: 256. A test needing more is redesigned, not silently raised.
- All generated ranks are 0--4, extents are 0--17, and materialized generated
  tensors contain at most 4,096 elements. Near-overflow shapes are validated
  without allocating their declared size.

These counts are acceptance inputs. They are not reduced in response to a
failure. A methodology correction increments the manifest methodology
revision, records the old and new rule, and reruns every affected phase.

Methodology revision 1 (before the accepted clean-copy run): the initial fresh
configure selected an unrelated Python 3.13 interpreter that could not import
the already-required Torch test dependency. The clean-copy command now passes
the exact existing `THIRAN_PYTHON_EXECUTABLE` recorded by the accepted baseline
configuration. This is an external test-tool dependency, not a copied build
artifact; no compiler output or CMake decision is reused. Counts and product
configuration are unchanged.

## Frozen source and numeric set

Adversarial frontend inputs include empty bytes, invalid UTF-8, embedded NUL,
a 4,096-byte identifier, a 16,384-byte line, 128 nested expressions, 64 nested
structured blocks, malformed numeric/comment/lexical/parser forms, integer and
finite-f32 boundaries, undefined names, type/rank/shape errors, ownership and
move errors, aliased mutation, unsupported backend operations, and unloaded
extension calls. These bounds are deliberately not parser bombs.

Numeric qualification includes zero/one/odd/non-power-of-two shapes, maximum
V0 rank, checked extent arithmetic, `INT64_MIN`, `INT64_MAX`, checked add,
subtract, multiply, and negation, plus `+0`, `-0`, smallest normal, a selected
subnormal, large finite values, infinities, and NaNs wherever the subsystem's
existing contract permits them. No fast-math mode is introduced.

Generated safe-i64 values are selected from `[-1024, 1024]`; f32 values use
exactly representable halves in `[-32, 32]`. Every generated failure records
the seed and case index.

## Frozen corruption set

Mutations operate only on copies. Each case records file type, mutation,
expected category, actual code/category, crash, partial execution, and
fallback. The canonical set is:

1. truncate to 0, 1, magic-length-minus-one, fixed-header-minus-one, half, and
   file-size-minus-one bytes;
2. flip the first magic byte;
3. corrupt format, compiler ABI, runtime ABI, backend, and payload-kind fields;
4. corrupt target/runtime-requirement bytes;
5. set selected declared lengths to zero, one, maximum u64, and off-by-one;
6. flip region, plan, payload, parameter, model, and outer digest bytes where
   the format contains them;
7. flip the first, middle, and last byte of serialized region/payload sections;
8. append one deterministic garbage byte where trailing bytes are forbidden;
9. corrupt model parameter/binding tables and embedded artifact bytes; and
10. corrupt checkpoint model identity, plan signature, parameter metadata,
    parameter payload/digest, and optimizer state.

Acceptance requires deterministic nonzero rejection, no assertion/stack trace,
no partial execution, no fallback, and safe cleanup. Decoder-specific codes may
differ between mutations, but a repeated mutation must keep the same code.

## Transactional writes and interruption

Artifact (`.tha`), model (`.thm`), and checkpoint (`.thc`) writers are tested
with controlled invalid destinations and, when a narrow test-only failpoint is
available, before temp creation, after temp creation, after partial write,
after the complete write, before rename, and after rename. A pre-rename failure
with an existing destination must preserve its exact bytes. No partial output
may be reported as success. Temp files cannot be treated as final artifacts.
No disk-, RAM-, or VRAM-exhaustion test is permitted.

Controlled `SIGTERM` may target only a qualification child writing inside its
own temporary directory. WSL, CUDA drivers, and unrelated processes are never
terminated or modified.

## Artifact and path reproducibility

Each CPU and GPU family is built ten times from identical semantic input and
configuration in separate directories. The report records the number of
unique full SHA-256, payload digest, region digest, plan digest, and execution
digest values. Byte reproducibility is claimed only when the full SHA-256 set
has cardinality one.

The same source is qualified through absolute, relative, two-directory, and
space-containing paths. Extension `.so` copies are loaded from two paths.
Where paths are non-semantic, registry identity, operation ordering, region and
plan identity, artifact semantic identity, and results must agree. Permitted
registration order is reversed and compared. JIT keys must hit for identical
operation version/recipe/shape/dtype/backend/planning options and miss when any
of those semantic inputs changes.

## Repetition, differential, and resource qualification

Representative integer, f32, matrix, structured control, Scan, AD, training,
extension, artifact, model, and graphics workloads are repeated. Exact bytes or
existing canonical digests are compared where required by their contracts.
Fused/unfused, reference/CPU/GPU, and reference/CPU/GPU/AOT/JIT extension
differential cases use the frozen seeds and counts above. Unsupported cases
must reject explicitly with `fallback=NONE`.

Training restarts from an identical state, compares the accepted loss/state
sequence, and compares continuous 200 steps with 80 steps, checkpoint/reload,
then 120 steps. Recurrent Scan AD remains unsupported.

Async stress covers submit/observe, submit/drop, repeated observation,
independent pending A/B, read/write reservations, deferred failure, and safe
drain. Reservation and retained-resource counts return to baseline. CPU/GPU
resource stress distinguishes Thiran-owned counters from allocator/driver
caching; RSS alone is not classified as a leak.

CPU soak executes 1,000 deterministic mixed iterations. GPU soak executes 500
bounded mixed iterations on the physical device. The report records failures,
digest mismatches, resource baseline/final, unobserved errors, and fallback;
duration is observational and never a threshold or performance claim.

## Concurrency and environment failures

Four independent CPU artifact processes, four checks, four builds, and two GPU
artifact processes run concurrently. Completion order is not constrained.
Same-output contention may produce one clean winner/one clean failure or a
valid atomic last writer; corrupt output or successful invalid output fails.

GPU-disabled, invalid-device, and driver-unavailable paths must state the
failure and never fall back. Driver unavailability is induced only through the
existing disabled-backend configuration or a controlled loader seam; system
libraries are not renamed. Malformed PTX is tested on copies. Real GPU OOM is
`NOT QUALIFIED` because deliberately exhausting the 4 GiB device is unsafe.
Host allocation failure uses checked arithmetic, a test seam, or a restricted
child; host memory is not intentionally exhausted.

## Clean-copy and configuration rule

The fresh-copy phase copies the current working source, including uncommitted
TH-024 files, while excluding `.git`, every `build` directory, CMake caches,
qualification temporaries, and Python caches. It configures and builds in a new
temporary root, then runs semantic, CPU, physical GPU, artifact, extension, and
CLI representatives. `git archive HEAD` is insufficient.

Debug, Release, warnings, CPU-only, ASan+UBSan, and physical-GPU builds compare
shared language results. LSan, TSan, and Compute Sanitizer are attempted only
when useful and are classified `UNAVAILABLE` rather than passed when the
environment prevents evidence. Physical CUDA is never run under TSan.

## Trust and explicit limitations

Extension shared objects and native payloads remain trusted native code. There
is no malicious-plugin sandbox, cryptographic artifact signature, hostile
multi-tenant isolation, arbitrary hostile-input hardening claim, cross-machine
claim without a second computer, or cross-GPU claim without a second physical
architecture. The qualified GPU is the RTX 3050 Ti Laptop GPU, compute
capability 8.6. TH-025 owns release decisions.
