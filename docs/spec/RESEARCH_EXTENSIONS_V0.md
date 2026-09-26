# Research extensions V0

TH-021 adds one deliberately narrow extension surface: compiler-loaded custom
numerical operations. It is not a general compiler plugin system. Extensions
cannot add syntax, rewrite AST or semantic IR, install passes, replace code
generation, or own hidden model state.

## Lifecycle and trust

The host creates an `ExtensionRegistry`, explicitly loads named shared-library
paths, validates every descriptor transactionally, and freezes the registry
before semantic analysis. There is no directory scan, network lookup, static
initializer registration, or hot reload. Lookup and the registry digest use
ordered canonical identities, so registration order is not semantic.

The library exports `thiran_extension_v0` and returns the public V0 descriptor
from `extension/v0/Extension.hpp`. Its ABI version must equal
`extensionAbiVersion`; this ABI is experimental and requires a matching Thiran
compiler/runtime development version. Duplicate extension or operation
identities, empty extensions, missing fields, invalid recipes, and malformed
derivatives reject the whole registration. Once frozen, registration fails.

The extension `.so` is trusted native compiler code. Loading it can compromise
the compiler process; V0 provides no sandbox, hostile-code isolation,
signature verification, or memory-safety guarantee. The generated `.tha` has
the separate existing TH-016 trusted-native-artifact model.

## Operation contract

Each operation has a stable extension ID and version, operation name and
semantic version, extension ABI version, arity, rank bounds, same-shape/result
rule, existing TH-006 effect class, backend mask, fusion declaration, forward
scalar recipe, optional reverse rule, canonical identity, and deterministic
descriptor digest.

V0 admits only:

- rank-1 or rank-2 `Tensor<f32,R>` inputs;
- same typed shape for every operand;
- a fresh tensor result shaped like one designated input;
- read-only, non-consuming inputs;
- `Pure`, or `MayTrap` for checked recipe primitives;
- explicit reference/CPU/GPU backend availability;
- bounded scalar recipes containing input lanes, finite f32 constants,
  add, subtract, multiply, checked divide, and negate.

It admits no aliases, borrowed results, mutation, consuming moves, external
pointers, hidden RNG, I/O, transfer, async submission, persistent state, or
extension callbacks in native execution. The compiler owns recipe validation,
reference interpretation, C++ emission, PTX emission, and physical allocation.
An operation cannot be marked fusible unless it is pure and has native recipe
lowering.

## Source and semantic IR

Normal call syntax resolves against ordinary functions first and then the
frozen registry. An unloaded name remains `TH005-UNKNOWN-FUNCTION`; there is no
runtime symbol guessing or fallback. A resolved call becomes a generic
semantic `Extension` instruction containing an owned descriptor copy. The
ordinary verifier revalidates identity, digest, arity, dtype/rank/shape,
read-only access, effect, recipe, and derivative metadata. TH-006 assigns the
result normal `Fresh` provenance and derives effects through its existing
effect summary.

Reference execution evaluates the owned scalar recipe lane by lane into a new
validated tensor. It is the semantic oracle only. CPU and GPU native paths do
not invoke the evaluator.

## Reverse differentiation

An optional reverse descriptor identifies differentiable inputs, saved primal
inputs/output, and one ordered scalar gradient recipe per input. Gradient
recipes use the declared primals, primal result, and output cotangent. They
must reference only declared saves, match arity, and depend on the cotangent.
TH-010 lowers the recipe into ordinary `Float`, `Add`, `Subtract`,
`ElementMultiply`, and `Negate` semantic instructions. Saved primals use the
existing TH-010 save table and TH-015 `savedForBackward` obligation; no plugin
tape exists. Missing derivatives are explicit AD ineligibility. Scan recurrent
AD remains deferred.

## TensorRegion, planning, and native lowering

TensorRegion has one generic `Extension` node carrying canonical identity,
digest, schema, and recipes. It has no operation-specific enum entry. Validated
pure recipes participate in ordinary TH-015 elementwise fusion; consumer
counts, save obligations, materialization barriers, liveness, interference,
reuse, output retention, and zero-size handling remain unchanged.

The CPU backend emits ordered scalar C++ operations without fast-math. The GPU
backend emits ordered PTX operations through the CUDA Driver path. Unsupported
backend masks reject with `fallback: NONE`; GPU never retries on CPU. Checked
divide publishes `TH021-DIVIDE-BY-ZERO`, not an output.

## Artifacts and JIT

TH-016 serializes the owned descriptor and recipes, never a function pointer or
library path. Artifact loading revalidates recipe, canonical identity, digest,
TensorRegion, and physical plan. CPU ELF and GPU PTX payloads therefore execute
without the compiler extension library. JIT executables likewise own their ELF
or PTX payload and can outlive the registry/library.

Region serialization feeds artifact integrity and JIT cache identity. It
therefore covers extension/package ID, operation ID and version, ABI, schema,
effect, derivative, recipe, backend, dtype/rank/shape specialization, and
planning/fusion options. A changed identity or recipe cannot reuse a stale JIT
entry.

## Deliberate limitations

V0 has no package manager, imports for extensions, remote registry, sandbox,
syntax extension, arbitrary dependent shape callback, arbitrary host callback,
custom allocation, views/aliases, opaque stateful operator, recurrent AD, or
general native kernel injection. Researchers compose explicit state as
`step(input, state, params) -> (output, next_state)` in ordinary functions and
forward `Scan`.
