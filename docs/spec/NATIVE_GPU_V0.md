# TH-013 native GPU backend with TH-014 async lifetime extension

TH-013 adds a bounded native GPU implementation below the existing typed structured semantic IR. The device-neutral path is `verified semantic IR -> TH-006 ownership/effect facts -> TensorRegion -> backend`. `TensorRegion` remains a straight-line numerical region rather than whole-program semantic authority. Structured control, including `Structured::Scan`, stays in semantic IR and rejects explicitly when a whole function cannot be represented by the GPU subset.

The first implementation adapter is CUDA, but CUDA blocks, grids, warps, streams, pointer types, PTX, and allocation APIs are not source-language constructs. `NativeGpu` is the only layer that loads the CUDA Driver API, owns device handles, supplies PTX kernels, chooses the physical device, maps work to launches, and translates driver failures. It loads `libcuda.so.1` directly and uses driver JIT loading for backend-owned PTX. It has no Python, PyTorch, Triton, JAX, TensorFlow, CuPy, generated-Python, reference-evaluator, or native-CPU execution dependency.

`THIRAN_ENABLE_NATIVE_GPU` is OFF by default. A disabled build retains discovery stubs that report `GPU-BACKEND-NOT-BUILT`; it does not require CUDA. The enabled implementation needs an NVIDIA CUDA driver at runtime but does not require nvcc or CUDA headers at build time. CMake probes and reports an available CUDA compiler for diagnostics without making it a semantic or build requirement for this Driver/PTX path.

## Supported region subset

The GPU extractor accepts read-only scalar `i64`/`f32` parameters used by the region, materialized contiguous offset-zero row-major rank-1/rank-2 `Tensor<i64,R>` and `Tensor<f32,R>` inputs, tensor literals representable by current semantic IR, immutable aliases, explicit tensor copies, tensor Negate, equal-shape tensor Add/Subtract/ElementMultiply, and full-rank Index where admitted by current semantics. Dtypes are never promoted or narrowed. Broadcasting, scalar arithmetic lowering, views, slice, transpose, matmul, sum/reduction, calls, tuples, mutation/move lowering, AD helper operations, imports, structured control, and scan are backend-unsupported.

TH-015 groups eligible linear elementwise chains into explicit verified PTX kernels and assigns materialized values to explicit device slots. Rank maps to a flat row-major element range only after runtime layout/shape validation. A fixed block size and computed one-dimensional grid remain CUDA-adapter choices. Default IEEE f32 instructions are emitted in IR order without fast-math, contraction, reassociation, `.approx`, or `.ftz`. Every ordered signed i64 negate/add/subtract/multiply operation sets the device error flag on overflow; observation reports `TH-SPEC-I64-OVERFLOW` without publishing a wrapped result. Index remains unfused, checks coordinates before launch, and reports `TH-SPEC-BOUNDS`.

## Storage, transfer, ownership, and synchronization

`storage::Tensor` remains host physical storage for a logical tensor. A GPU execution creates private RAII device allocations carrying dtype, shape, byte length, and a retained allocation handle. TH-015 obtains owned intermediate/output allocations from a verified per-execution slot map; different pending executions never share this scratch. Raw device pointers never become language values. TH-014 `submitNativeGpuAsync` retains the plan, slot allocations, H2D staging, device execution resources, D2H staging, and explicit read/write reservations until observation/drain. `executeNativeGpu` remains submit-then-observe. Evidence distinguishes logical/materialized intermediates, slots, groups, allocations, launches, pending reservations, observation, and release. There is no automatic CPU/reference retry.

Immutable TensorRegion aliases share one device allocation handle. Repeated input arguments with the same host StorageObjectId, dtype, and shape share one uploaded allocation after each descriptor has independently passed type, shape, and layout validation. Explicit host deep copies have distinct StorageObjectIds and therefore distinct uploads. An explicit region `Copy` uses a checked device-to-device copy into an independent planned root. The backend does not mutate input allocations and does not implement copy-on-write. General cross-execution residency, caching, and allocator pooling are not introduced.

Zero-element tensors allocate zero device bytes, perform no zero-byte CUDA allocation/transfer, launch no kernel, and preserve dtype/shape on host observation. Checked TH-007 element and byte arithmetic precedes allocation and launch. Non-contiguous or view descriptors reject with `GPU-UNSUPPORTED-LAYOUT`; no hidden materialization occurs.

The TH-014 path queues copies and kernels on a backend-private stream and records one backend-private completion event. It does not perform unconditional per-kernel synchronization. Observation synchronizes that event, checks the region-wide i64 error flag, publishes the staged output, and releases reservations. CUDA streams/events remain implementation details; the device-neutral state/reservation contract is specified in [ASYNC_LIFETIMES_V0.md](ASYNC_LIFETIMES_V0.md). No transfer/compute overlap or performance improvement is claimed.

## Failure categories

Discovery and execution distinguish backend unavailable, invalid device, backend unsupported, runtime/resource/driver failure, and supported semantic failure. Driver loading, initialization, device enumeration/selection, capability, context/stream/event creation, module JIT loading, symbol lookup, allocation, transfers, launch, event recording, synchronization, and result transfer are checked. Failure after reservation acquisition drains submitted stream work before releasing obligations. Deferred failures surface at observation. Dropping pending work drains it and sends an error to the device-neutral unobserved-error channel; no destructor throws. RAII releases events, modules, streams, allocations, contexts, and the driver library on every path.

GPU integration tests use return code 77 for unavailable hardware so CPU-only CI remains green while still distinguishing unavailable from pass. A skipped device suite is not TH-013 qualification; checkpoint PASS requires the same suite to execute kernels on a physical GPU.

TH-016 adds two explicit uses of this adapter. Persistent GPU AOT stores the
already generated PTX and loads those exact bytes later; it is Thiran AOT to PTX
with CUDA Driver JIT at device-load time, not fully ahead-of-time SASS. GPU JIT
generates PTX at JIT time. Both retain the payload in the TH-014 pending state,
and neither uses `nvcc`, `ptxas`, a framework, CPU fallback, or evaluator fallback
at execution.
