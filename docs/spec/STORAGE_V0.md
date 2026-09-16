# TH-007 concrete host tensor storage (developer substrate)

TH-007 adds `thiran_v0_storage` after TH-006 abstract ownership analysis. It does not replace typed structured semantic IR, the independent reference evaluator, or the tensor/dataflow DAG architecture. Concrete storage is not native compiled Thiran, AOT/JIT, or a GPU runtime.

## Three identities and lifetime

`ResourceId` is the compiler's logical source alias/mutation-safety resource. `StorageObjectId` is a deterministic runtime allocation-object ID. An address is an unobservable implementation detail. `ResourceId != StorageObjectId != address`; neither identity is derived from a pointer. The conservative first bridge maps each fresh ResourceId to a distinct storage object, aliases/views to the same object, and `IndependentCopyOf` to a new one. This is not a language invariant: a later verified memory planner may reuse one physical allocation for distinct non-overlapping ResourceIds without changing source-visible semantics.

`StorageHandle` is a narrow retention abstraction currently implemented with a standard-library shared lifetime handle. It retains owned host bytes while a tensor or view needs them. Element access does not alter its count. Shared handle count is not mutation permission, and reference counting is not borrow checking. A view may keep bytes physically alive after a C++ root object disappears, but a source view whose root moved, escaped, or conflicts with mutation remains illegal under TH-006. Storage lifetime prevents use-after-free in accepted lowered programs; it cannot legalize rejected source. StorageObjectId equality proves concrete sharing, not source exclusivity.

The conservative resource bridge snapshots the created-resource facts it needs, rather than retaining a potentially dangling reference to an analysis result.

## Storage and descriptor

The host storage object owns a byte vector, its byte length, and a deterministic StorageObjectId. Typed i64 access uses checked byte offsets and `memcpy`, not an unproven aligned `i64*` cast. No raw mutable pointer is exposed by safe APIs. No mmap, file, device, or tensor serialization is implemented. These bytes are not a stable file format, native ABI, checkpoint representation, or device ABI.

The verifiable `TensorDescriptor` contains dtype, concrete nonnegative extents, strictly positive element strides, element offset, StorageHandle, materialized/view classification, and a debug contiguity claim. Rank is shape length. Unknown extents are semantic `ShapeFact` slots, never runtime descriptor extents. Semantic bridge verification requires Tensor rank and dtype equality, known extent equality, and correct unknown-slot rank; concrete extents must fit the V0 nonnegative i64 extent domain.

| Dtype | Width in host bytes | TH-007 value execution |
| --- | ---: | --- |
| bool | 1 | descriptor only |
| i32, u32, f32 | 4 | descriptor only |
| i64 | 8 | checked load/store, views, logical copy |
| u64, f64 | 8 | descriptor only |

No f16, bf16, complex, or arithmetic kernel is claimed.

## Layout and views

New materialized i64 tensors are row-major contiguous with offset zero and checked conventional strides: `[2,3] -> [3,1]`, `[2,3,4] -> [12,4,1]`. Materialized descriptors require exact byte length, zero offset, and logical row-major contiguity. A C++ copy of a Tensor handle is an ordinary shared-storage alias, not an element copy; `let B = A` must not induce O(numel) work. A view relationship is explicitly marked in the descriptor and retains the same handle. It is not an independent source value alias.

Full scalar access takes exactly rank-many zero-based coordinates and rejects a missing, negative-at-source, or out-of-range coordinate with `TH-SPEC-BOUNDS`. Partial integer indexing removes selected axes and creates a read view with composed offset and retained strides. Half-open slices use omitted start `0`, omitted end `extent`, positive nonzero step, `0 <= start <= end <= extent`, and extent `0` or `1 + (end-start-1)/step`. Integer and slice selectors may mix. Zero or negative step and invalid slice bounds yield `TH-SPEC-SLICE`. Rank-two transpose swaps shape and strides without allocating. Views of views compose the existing offset/strides and keep base storage. Negative-stride views are absent.

Contiguity is a logical row-major layout predicate, not an offset-zero test. Size-one axes do not constrain a logical stride; a contiguous row view may have nonzero offset. Empty tensors have zero logical elements and may have zero bytes. Their descriptor offset is permitted at most one-past the allocation's element count, never dereferenced; an empty derived view whose arithmetic cursor would pass that position is canonicalized to the one-past cursor. All strides remain positive. Contiguity for an empty index space is vacuous when descriptor arithmetic is valid.

`copy` iterates source logical row-major coordinates and creates fresh contiguous storage and a fresh StorageObjectId, including for strided/transposed views. `reshapeView` only accepts equal checked element count and a contiguous source; it never copies. `reshapeCopy` explicitly creates independent contiguous i64 storage. Ordinary handle aliasing does not perform copy-on-write. A later low-level write to shared storage is seen through aliases/views, while an explicit copy is isolated.

## Checks and mutable boundary

Constructors and the independent descriptor verifier use checked numel, stride, byte count, offset addition/multiplication, and maximum reachable offset. The verifier rejects malformed manually built descriptors, including rank mismatch, zero stride, invalid dtype/handle, arithmetic overflow, outside storage, impossible materialized layout, and false contiguity claims. Validation occurs before pointer arithmetic or `memcpy`.

`MutableTensorRef` is move-only, retains its Tensor descriptor/handle even if the C++ wrapper used to obtain it expires, and provides checked i64 writes to the existing object with no hidden clone. It is runtime infrastructure, not a safe source `mutable_data()` API. Future lowering must follow `TH-006 exclusive mutable proof -> obtain MutableTensorRef -> write`. Neither shared-handle count nor storage-ID equality substitutes for that proof. TH-007 tests the primitive directly under an assumed legal lowering context; arbitrary source indexed assignment remains deferred.

Standalone shape utilities implement the frozen trailing-axis equal-or-one broadcast rule (including zero with one and rank-zero scalar), rank-two matmul inner-dimension equality, and zero-based rank-dropping sum-axis shape. They do not execute numerical operations. Runtime shape guard failures normalize to `TH-SPEC-BROADCAST`, `TH-SPEC-SHAPE`, and `TH-SPEC-AXIS` respectively.

The storage observation reports dtype, shape, logical i64 row-major values, and materialized/view classification. A separate developer layout dump may add StorageObjectId, offset, strides, and contiguity. Neither prints addresses; neither is a stable artifact schema.

`Buffer<T>` remains a distinct TH-006 semantic resource. The owned byte-storage primitive could support a future Buffer runtime, but TH-007 does not expose a competing general Buffer API; user-visible Buffer operations remain deferred.
