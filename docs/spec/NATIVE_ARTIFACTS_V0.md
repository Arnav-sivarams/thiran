# Native artifacts and JIT V0

TH-016 defines the first durable boundary between Thiran compilation and later
native execution. The V0 format is deliberately small, local, and unstable. It
is not a package format, deployment system, stable ABI promise, authenticity
scheme, or hostile-code sandbox.

## Precise terms

**AOT** means that Thiran parsing, semantic verification, ownership/effect
qualification, TensorRegion extraction, physical planning/fusion, and backend
payload generation happen before a later execution process. A CPU artifact
contains a native ELF shared object. A GPU artifact contains pre-generated PTX;
therefore the accurate description is *Thiran AOT compilation to a persistent
PTX payload with CUDA Driver JIT at device-load time*. PTX is not claimed to be
machine-specific SASS.

**CPU JIT** is an external-native-toolchain JIT: at JIT time Thiran emits the
already planned/fused C++ lowering, invokes the configured native C++ compiler
to create a shared object, loads it with `dlopen`/`dlsym`, and returns a retained
typed executable. It is not an in-process machine-code JIT.

**GPU JIT** performs Thiran physical planning and PTX generation at JIT time,
then retains that PTX and loads it through the CUDA Driver for execution. This
differs from GPU AOT, where the Thiran PTX generator ran before the artifact was
written and artifact execution loads the stored bytes verbatim.

Historical whole-Graph Region labels named `AOT`, `JIT`, and `FALLBACK` remain
classification labels for the legacy generated-Python path. They are not these
native AOT/JIT implementations and are never consulted by the V0 artifact
runtime.

## Container and versions

A `.tha` artifact is one deterministic little-endian binary container:

```text
THIRAN16 magic
format version (0)
compiler/artifact ABI (1)
native runtime ABI (1)
backend and payload kind
target and declared runtime requirements
typed entry signature and shape specialization
physical reuse/fusion options and plan digest
serialized verified TensorRegion size/digest
payload size/digest
serialized TensorRegion
ELF shared object or PTX payload
```

Strings and collections are length-prefixed and bounded. All reads use checked
remaining-length arithmetic. Loading rejects bad magic, truncation, trailing
data, excessive lengths/counts, unknown versions or backend, target/payload
mismatch, malformed types/shapes, incompatible entry metadata, invalid lowered
regions, missing payloads, size/digest mismatches, invalid ELF/PTX structure,
and a recomputed physical-plan identity mismatch. Digests use deterministic
FNV-1a-64 for accidental-corruption detection and identity; they provide no
cryptographic authenticity.

The serialized TensorRegion is lowered compiler metadata, not source and not a
replacement whole-program semantic authority. Loading independently verifies
it. The runtime deterministically reconstructs the physical host-orchestration
plan from the stored options, verifies its digest, and uses the already emitted
native payload. It does not parse source, rerun semantic/ownership analysis, or
regenerate backend payloads.

## Typed native entry ABI

Each entry records its name, parameter count, scalar/tensor kind, i64/f32 dtype,
rank, optional static extents, and result type. Scalar and rank-zero tensor are
distinct; the latter is outside the current native TensorRegion subset. Runtime
values are validated before native loading/execution. CPU artifacts currently
require concrete tensor extents and expose a versioned internal C ABI made only
of fixed-width fields, shape/data pointers, and element counts. No `std::any`,
Python object, framework tensor, or untyped semantic value crosses the ABI.

The CPU payload statically contains the needed Thiran storage implementation.
The installed or co-located `thiran-artifact` launcher is the minimal runtime:
it validates/extracts the payload and loads its single C entry symbol. The
payload otherwise depends only on ordinary host native libraries. AOT execution
never invokes the host compiler.

GPU artifacts accept the same typed values through the C++ runtime API. The
only GPU-specific runtime dependency is `libcuda.so.1` and a compatible device
(V0 PTX declares PTX 6.0, `sm_50` minimum). CUDA concepts remain below language
semantics. No `nvcc` or `ptxas` is required to execute the PTX artifact.

## Planning, constants, numerics, and lifetime

Both artifact builders consume the TH-015 `PhysicalPlanOptions`, record a digest
of the exact verified plan, and emit from that plan. CPU payloads contain the
planned slots and fused loops. GPU payloads contain the fused PTX entries; the
validated plan supplies host-side allocation and launch orchestration. Compiler
constants are present in the serialized lowered region and native payload.
Model checkpoint/parameter packaging is not part of V0.

Checked i64 helpers, bounds checks, scalar/tensor distinctions, and ordinary
IEEE f32 operation order are unchanged. There is no fast math, reassociation,
silent conversion, evaluator fallback, GPU-to-CPU fallback, or framework
fallback.

Stored-PTX execution uses the TH-014 submission/observation implementation.
The pending state owns its PTX module, context, stream, event, allocations,
staging, inputs, output reservation, and physical plan until observation or safe
drop-drain. Synchronous artifact execution remains submit followed by observe.

## JIT and cache

`JitCompiler` accepts a verified TensorRegion, concrete runtime specialization,
backend, and physical-plan options and returns a retained `JitExecutable`.
Compilation and loading failures are distinct from entry execution failures.
CPU temporary source/container files are removed; a loaded shared-object handle
and its extracted payload directory live exactly as long as the executable or
cache entry. GPU executable ownership includes its verified plan and generated
PTX.

The cache is local and in-memory. Its deterministic key hashes the serialized
lowered computation, backend, target, compiler/runtime ABI versions, full typed
entry specialization, and reuse/fusion options. Thus dtype, rank, extent,
backend, ABI/target, computation, or code-affecting option changes cannot hit an
incompatible executable. PTX remains device-portable above the declared minimum
and the Driver performs device specialization at load, so the physical device
ordinal is intentionally not in the Thiran PTX key. Counters expose successful
compilations, hits, and misses. There is no disk, network, distributed, or
shared cache entry to repair or trust.

## Tooling and trust boundary

`thiran-artifact inspect` deterministically reports versions, backend, target,
entry signature, planning flags/digest, payload kind/size/digest, region
size/digest, and runtime requirements. `thiran-artifact run` is the minimal
zero-parameter command-line adapter; typed parameterized execution is available
through the runtime API. Neither command invokes a compiler or framework.

Artifacts are trusted compiler outputs containing executable native code.
Structural validation protects compatibility and catches corruption; it does
not establish provenance, signatures, malicious-code isolation, memory-safe
execution of adversarial binaries, or cryptographic integrity.

## Deliberate limits

V0 supports only the accepted native TensorRegion subset. The CPU backend uses
the external system compiler rather than LLVM or an in-process JIT. The cache is
process-local. Native binary bit-for-bit reproducibility is not promised even
though metadata and cache identity are deterministic. There is no model bundle,
checkpoint format, registry, service, container workflow, or TH-017 deployment
surface, and no performance claim.
