# Performance Qualification V0

Status: frozen before TH-023 final measurements on 2026-09-26.

This specification defines the machine-local TH-023 qualification. It is a
measurement contract, not a performance target. Unfavorable results are valid
results. Workload sizes, sampling, ordering, synchronization, and baseline
rules below must not be changed after inspecting final timings except for a
documented correctness, resource-safety, or methodological defect. Every such
change must be recorded in the amendment log before replacement measurements.

## Scope and timing clock

The qualification uses `std::chrono::steady_clock` for in-process host wall
time and `CLOCK_MONOTONIC`-equivalent process elapsed time in the process
driver. Durations are stored as integer nanoseconds. Reported values are
normalized to nanoseconds per logical invocation when a sample contains a
fixed batch of invocations.

Final timings come from an optimized Release qualification executable with the
native GPU backend enabled. Sanitizer results are correctness evidence only.
The Thiran kernels and artifact entry wrappers are the production emitters and
runtimes; there is no benchmark-only Thiran implementation.

## Deterministic inputs and correctness

All rank-one inputs have `f32` dtype and length `N`. For zero-based element
index `i`, inputs are generated once outside timed regions as:

```
x[i] = f32(((i * 17) % 1009 - 504)) / f32(37)
y[i] = f32(((i * 29 + 11) % 1013 - 506)) / f32(41)
```

The integer arithmetic is performed in signed 64-bit arithmetic and the two
explicit conversions and divisions are rounded to `f32`. The frozen sizes are:

| label | elements | input bytes | output bytes |
| --- | ---: | ---: | ---: |
| small | 1,024 (`2^10`) | 8,192 | 4,096 |
| medium | 1,048,576 (`2^20`) | 8,388,608 | 4,194,304 |
| large | 16,777,216 (`2^24`) | 134,217,728 | 67,108,864 |

The large case is below the observed 8 GB WSL memory and 4 GB GPU memory
capacity even for the V0 unfused allocation structure. Inputs include positive,
negative, and non-integer values and contain no intentional NaN or infinity.

Before timing each case, every implementation is run and compared with a
scalar `f32` reference preserving the stated operation boundaries. Exact bit
equality is required for Thiran fused versus unfused and extension versus its
built-in equivalent. External implementations must also match exactly; if a
compiler or framework contract prevents exact equivalence, its ratio is
reported `NOT COMPARABLE` rather than relaxing correctness. Each output is
consumed after (not inside) the timed sample using FNV-1a over the little-endian
raw `f32` bit patterns. Input digests and output digests are retained.

## Workloads

### A. Linear elementwise chain

The primary workload is:

```
a = x + y
b = -a
c = b .* y
d = c - x
e = d .* y
return e
```

The fused and unfused Thiran cases use the same `TensorRegion`, inputs, output,
and arithmetic. Fused uses `{reuse=true, fusion=true}`. Unfused uses
`{reuse=false, fusion=false}` so both fusion and physical temporary retention
are observable. The report includes plan group count, fused group count,
loop/kernel count, logical intermediates, materialized intermediates, slots,
planner-owned allocations, and logical temporary bytes.

### B. Generic extension chain

The extension workload loads the existing TH-021 fixture and uses:

```
a = -x
b = research_square_linear(a)  // a*a + a
c = b + y
return c
```

The exactly equivalent built-in workload is:

```
a = -x
s = a .* a
b = s + a
c = b + y
return c
```

Both use `{reuse=true, fusion=true}` and the frozen size is medium (`2^20`). The fixture is loaded only while parsing
and lowering. Generated C++ and PTX are audited for direct scalar recipe
operations and for absence of registry lookup, dynamic loading, callback, or
evaluator calls in the hot loop.

### C. Materialization barrier workload

The retained/multiple-consumer workload uses the frozen medium size (`2^20`):

```
a = x + y
b = -a
c = a .* y
d = b + c
e = d - x
return e
```

`a` has two consumers and must materialize. This workload is diagnostic and is
not used for a language-speed ratio against workload A. It is measured with
normal planning and reports groups, loops/kernels, slots, allocations, and
temporary bytes.

### D. Production source and artifact workflow

A checked-in zero-parameter source fixture contains two deterministic
32-element `f32` literals and the accepted standalone four-op chain
`a=x+y; c=a.*y; d=c-x; e=d+y`. The production `thiran` executable measures
these distinct scopes:

1. `thiran build SOURCE --backend cpu --output ARTIFACT`;
2. `thiran build SOURCE --backend gpu --output ARTIFACT`;
3. `thiran run SOURCE --backend cpu`;
4. `thiran run SOURCE --backend gpu`;
5. `thiran artifact run CPU_ARTIFACT`;
6. `thiran artifact run GPU_ARTIFACT`.

Each command is a fresh process. "Fresh-process" does not claim cold filesystem
or machine caches. Build and source-run outputs are unique temporary paths;
system caches are not cleared. Artifact sizes, contained ELF/PTX payload sizes,
and exit/output correctness are recorded.

### E. TH-017 deployment overhead

The existing reference affine model is trained for the accepted 200 SGD steps,
checkpointed, destroyed, reloaded, snapshotted, and bundled with production CPU
and GPU artifacts. The public input is `f32[1] = [2.0]`. CPU and GPU execution
must match the reference prediction. The workload is labelled **deployment
overhead workload**, not a neural-network performance benchmark.

The in-process API is measured with 5 CPU warmups or 10 GPU warmups followed by
30 samples. Fresh-process `thiran model run` is measured 10 times per backend.
Artifact/module initialization and the current GPU H2D/D2H boundary are
included. No persistent-device-residency claim is made.

## Sampling policy

| scope | warmups | measured samples | invocations per sample |
| --- | ---: | ---: | ---: |
| Thiran CPU warm in-process | 5 | 30 | 1 |
| Thiran GPU warm-process transfer-inclusive | 10 | 30 | 1 |
| C++ small | 5 | 30 | 8,192 |
| C++ medium | 5 | 30 | 8 |
| C++ large | 5 | 30 | 1 |
| NumPy/PyTorch small, if available | 5 CPU / 10 GPU | 30 | 256 |
| NumPy/PyTorch medium, if available | 5 CPU / 10 GPU | 30 | 8 |
| NumPy/PyTorch large, if available | 5 CPU / 10 GPU | 30 | 1 |
| independent CPU/GPU builds | none | 10 | 1 |
| fresh-process source/artifact/model | none | 10 | 1 |

Warm in-process means the benchmark process, inputs, and JIT executable are
already initialized. It does not imply that the V0 backend retains CPU output
buffers, a CUDA context, module, or device allocations across calls; the report
must state what the implementation actually retains.

The CPU elementwise implementations are sampled in a deterministic rotating
order each round: Thiran fused, C++ fused, Thiran unfused, C++ materialized,
rotated left by `round mod 4`. GPU fused/unfused use repeated `A/B/B/A` blocks,
discarding only the first 10 calls of each implementation as predeclared
warmup. Extension/built-in pairs use the same `A/B/B/A` policy. Inputs are not
regenerated between implementations.

## GPU timing and synchronization

Transfer-inclusive GPU host wall time begins immediately before production
submission/execution and ends only after `PendingGpuExecution::observe()` (or
the synchronous wrapper that performs the same observation) has completed.
Observation waits with `cuEventSynchronize`; therefore H2D, launch, kernel
execution, D2H, and result publication are complete before the clock stops.

The current production API creates/loads the CUDA Driver module as part of each
execution and does not expose a timing-enabled CUDA event around kernels alone.
TH-023 will not alter that API solely to manufacture a kernel number. Thus
compute-oriented/kernel-only time is pre-registered as `UNAVAILABLE` unless a
non-semantic, independently synchronized measurement becomes possible during
implementation; any such addition requires an amendment below. Host wall time
and any CUDA event time must never be combined in an unlabeled ratio.

GPU evidence records H2D/D2H counts and bytes, device-to-device copies, kernel
launches, device allocations, planner allocations, synchronizations, module
loads/Driver-JIT observations, and output digest. `nvidia-smi` state is captured
before and after major GPU runs when WSL exposes each field.

## Mandatory C++20 CPU baseline

The checked-in baseline uses the same input arrays, `float`, operation order,
and output materialization. It is single-threaded and has two competent forms:

* manually fused: one output allocation and one loop;
* materialized: five output/temporary vectors and five straightforward loops.

It is compiled by GCC 13.3 with exactly:

```
-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -std=c++20
```

No `-march=native`, fast-math, OpenMP, hand assembly, or volatile handicap is
used. Results are digested outside timing. Thiran's generated CPU artifact is
compiled with the production flags discovered during archaeology, even when
those flags are weaker; the report shows both flag sets prominently and does
not characterize the comparison as pure loop codegen when ABI copies or
allocations are included.

## Optional installed baselines

No package may be installed. System `python3` and the interpreter already used
by repository tests are probed separately. If NumPy is present, eager
vectorized expressions with `float32` and ordinary temporary behavior are
measured. If PyTorch is present, eager `float32` is measured at a frozen single
CPU thread for the matched CPU result; its discovered default thread count may
be reported separately but is not the primary ratio. CUDA PyTorch, when it can
access the physical GPU, reports separately:

* device-resident synchronized eager computation; and
* synchronized host-to-device, computation, and device-to-host execution.

`torch.compile` is excluded. Python-loop baselines are excluded. `nvcc` CUDA is
included only if `nvcc` was already installed before qualification. MATLAB and
Rust are not installed and are excluded when unavailable.

## CPU affinity and conditions

CPU microbenchmarks and the mandatory C++ baseline run under `taskset -c 8` if
a preflight proves that affinity works. The same affinity is used for every
compared CPU implementation and optional CPU framework baseline. If CPU 8 is
not available, all run unpinned and the deviation is logged. Build and GPU
process measurements are not pinned. Load average is captured before the
qualification. The machine is WSL2, so results are machine-local and subject to
Windows host scheduling; they are not native-Linux or cross-machine claims.

## Statistics and records

For every primary row the raw samples and these aggregates are retained:
sample count, median, arithmetic mean, population standard deviation, nearest-
rank/interpolated p10 and p90 (the harness uses linear interpolation at index
`p*(n-1)`), minimum, maximum, and coefficient of variation where meaningful.
Median is the central comparison. Ratios use:

```
baseline median / Thiran median
```

A value above one means Thiran was faster for that specifically labelled scope;
a value below one means Thiran was slower. No significance claim is made from
the distributions alone. Throughput is `N / median_seconds`. Approximate
logical bytes are reported only with an explicit formula and are not presented
as physical traffic.

Machine-readable JSON records contain `benchmark_name`, `implementation`,
`backend`, `size`, `build_type`, `warmup_count`, `sample_count`, `batch_count`,
`timing_scope`, raw normalized samples, aggregates, input/output digests,
commit, environment metadata, and applicable GPU/planner counters. Process RSS
is reported only if measured by a clearly labelled peak-RSS mechanism; it is
never inferred from requested allocation bytes.

## Exclusions and claim boundaries

No fragile timing threshold is added to CTest. Structural self-tests may assert
fusion, kernel, allocation, no-fallback, and cache behavior. Sanitizer timings
are excluded. Graphics is optional and omitted unless mandatory qualification
finishes with enough time to add it without changing this plan. TH-023 does not
add vectorization, threading, persistent GPU tensors, allocator redesign,
autotuning, new fusion, CLI optimization, shipping, portability, stress, or AI
model work.

## Amendment log

1. Before any final timing, the extension and materialization sections were
   clarified to use the already frozen medium size (`2^20`). Their size had
   been inadvertently omitted from the initial prose; no measurement had been
   run and no workload or candidate size was changed.
2. The first qualification invocation stopped before compilation, warmup, or
   sampling because the harness called the production driver's standalone
   source helper, whose contract rejects parameterized entries. The harness was
   corrected to use the same checked module and strict non-standalone
   `TensorRegion` extraction used by artifact/JIT qualification. Workloads,
   sizes, samples, and runtime paths are unchanged; no timing was retained.
3. After the completed CPU run, audit showed that the dynamic-shape physical
   plan correctly leaves `byteCount` unknown, so its static temporary-byte sum
   is zero. The merge step now preserves that value as
   `logical_temporary_bytes_static_plan` and derives the exact concrete runtime
   value as `materialized_intermediates * N * sizeof(f32)`. No timing or
   structural count was changed and no sample was discarded.
4. The first fresh-process driver invocation stopped on its first build because
   the production CLI spells the output option `-o`, not `--output`. No timing
   sample was retained. The driver invocation was corrected to the documented
   production spelling; scope and sampling are unchanged.
5. The corrected CLI invocation then stopped before its first successful build
   because strict standalone lowering rejects tensor negation sourced from
   literals (`tensor Negate not in native subset`). Changing language semantics
   is outside TH-023, so workload D alone was amended to the already accepted
   standalone four-op `add/multiply/subtract/add` chain. Primary workload A is
   unchanged. No process timing was retained before this amendment.
6. The standalone rejection remained because negative numeric literals also
   lower through unary negation. Workload D's two literal inputs were therefore
   changed to positive non-integer values; its subtraction still exercises a
   signed result path. The mandatory parameterized workloads retain the frozen
   mixed-sign formula. Again, no successful process sample had been retained.
7. One CPU process run completed its core samples in memory but failed before
   writing JSON when the driver resolved the test-environment Python symlink to
   `/usr/bin/python3.12`, thereby losing the virtual environment. The path is
   now kept absolute but unresolved. The entire process suite is rerun; none of
   the discarded in-memory samples are reused.
8. Recovery after the WSL VM was terminated preserved the completed original
   merged result byte-for-byte at `/tmp/TH-023-machine.pre-recovery.json`. The
   merger now writes its same-directory temporary file, flushes and fsyncs it,
   atomically replaces the canonical JSON, and fsyncs the containing directory.
   This changes only crash safety: workloads, ordering, warmups, samples,
   timing scopes, statistics, and existing measurements are unchanged. The
   completed pre-restart batch remains audit evidence and is not combined with
   the clean post-recovery batch.
