# TH-014 device-neutral asynchronous lifetime contract

TH-014 introduces an internal compiler/runtime contract for outstanding native work. It does not freeze public source syntax and does not turn a program into a whole-program dataflow graph. The insertion point is below verified structured semantic IR, TH-006 ownership/effect analysis, and TensorRegion extraction: an eligible region can be submitted to a native backend and returns a pending execution handle. Ordinary callers continue to use the synchronous API, implemented as submit followed by observe.

**logical value != physical storage != async resource != backend fence.** `ResourceId` remains the TH-006 logical provenance identity, `StorageObjectId` remains TH-007 physical host-storage identity, and `AsyncResourceId` identifies a runtime reservation root. Aliases and views of one storage object expose the same async resource; an explicit deep copy exposes an independent one. CUDA streams and events implement submission and completion but never become language values or generic ownership concepts.

## Operation state and observation

An `AsyncOperation` records a submitted transition, enters Pending after backend submission succeeds, and may be polled into Completed or Failed. Physical completion does not publish a result or release its lifetime obligations. `observe` narrowly waits for that operation, surfaces its deferred error or completed result, releases its reservations, and records Observed. Repeated observation returns the cached success/failure without waiting twice. A malformed handle/resource/dependency is rejected deterministically.

`PendingGpuExecution` is distinct from `GpuValue`. `value()` returns no value before successful observation, even if a very short kernel has physically completed. Its output async resource carries a write reservation until observation. Successful observation constructs an ordinary fresh host `storage::Tensor` (or scalar); a failed observation constructs no completed value.

The generic state machine intentionally distinguishes backend completion from semantic observation. This permits deterministic tests of pending lifetime obligations without sleeps or a race against fast hardware.

## Reservations and storage behavior

Submission atomically acquires a normalized set of root reservations. Supplying one aliased input more than once creates one reservation. A read reservation retains the resource, coexists with other async and ordinary immutable reads, and blocks mutation, destructive move/replacement, explicit free, and an unordered async writer. A write reservation retains the resource and blocks ordinary completed-value reads, mutation, move/free, and unordered readers or writers.

`storage::Tensor` consults its shared async root at physical read, view creation, mutable-reference construction/use, move, and replacement boundaries. A caller may drop an ordinary host wrapper while its resource is reserved: the reservation contains an opaque lifetime retention, so the physical `StorageObject` cannot be freed. This runtime retention does not legalize a source view rejected by TH-006 and does not use handle count as mutation permission.

Conflicting submissions reject with `ASYNC-RESERVATION-CONFLICT` unless they name an explicit predecessor dependency. The conservative, single-host-thread V0 dependency implementation retains the requested resource handles, narrowly observes only declared predecessors, then acquires the consumer reservations and starts the consumer. Unrelated work is not synchronized or released. Multiple operations over independent resources and multiple readers of one immutable resource may remain pending simultaneously. Operations inside one GPU TensorRegion are ordered on its private backend stream, so producer kernels precede their consumers without a host wait between kernels.

## Failure and drop policy

Preflight, device selection, and backend setup failures occur before reservation acquisition where possible. Once reservations are acquired, a submission callback failure invokes the backend's safe-abort/drain callback before releasing reservations and returning the immediate error. A failure discovered by event synchronization is deferred: observation returns it and releases reservations without fabricating a value.

An unobserved move-only handle is not silently discarded. Its non-throwing destructor performs a narrow drain/observation, releases reservations, and places any deferred failure into the runtime's unobserved-error channel. The enclosing runtime must consume that channel with `takeUnobservedAsyncErrors`; tests prove a dropped checked-overflow failure remains present. This is the conservative V0 policy in the absence of a public must-observe type. It is not cancellation, and it explicitly permits blocking during drop to preserve safety.

## Effects

No second effect model is introduced. The backend API reports the existing TH-006 `Async | Transfer` effect set through `nativeGpuAsyncEffects`. The source language currently has no public operation that can construct a pending handle, so no source instruction is falsely marked Pure and no new semantic IR node is required. When a source/compiler intrinsic is introduced, its ordered instruction must carry these existing effect classes and its resource operands must lower into the reservations specified here.

## CUDA adapter

The first physical implementation owns one CUDA context, nonblocking stream, completion event, PTX module, device allocations, upload staging, download staging, and error flag for each pending execution. H2D copies, kernels, D2H copies, and the completion event are enqueued in one stream. There is no per-kernel `cuCtxSynchronize`. Observation synchronizes the recorded event only. Context activation makes cleanup safe when independent pending executions own different contexts.

Pageable input values are copied into backend-owned staging before `cuMemcpyHtoDAsync`; staging remains alive through event completion. D2H targets are backend-owned and cannot be published before observation. CUDA is permitted to synchronize internally for pageable memory, so this contract makes no overlap or host-progress claim. Zero-byte transfers are skipped while a completion event still supplies the logical pending/observation boundary.

Device allocations, the module, context, stream, event, driver library, retained input wrappers, and staging remain owned by the pending state until safe completion and RAII destruction. Cleanup never throws. The synchronous TH-013 entry point uses the same implementation as `submitNativeGpuAsync(...).observe()`.

## Deliberate V0 limits

There is no public async source syntax, CPU async executor, cross-handle device-resident value, cancellation API, multi-threaded handle contract, pinned-host allocation, concurrency/overlap promise, allocator cache, fusion, memory planner, global scheduler, async AD/training, AOT/JIT artifact, or performance claim. Explicit dependencies are conservatively predecessor-observing; a later backend-neutral fence-to-fence dependency may remove that host wait without changing reservation semantics. These are later extensions, not implied by TH-014.
