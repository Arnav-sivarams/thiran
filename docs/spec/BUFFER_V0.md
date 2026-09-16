# TH-009 host Buffer V0 substrate

`Buffer<T>` remains a semantic resource distinct from `Tensor<T,R>`. A Buffer has element type and element count; it has no rank, shape, strides, broadcasting, matrix operation, or tensor-literal meaning. TH-009 adds an internal `storage::HostBuffer` runtime foundation and does not invent a source constructor, indexing syntax, or conversion.

## Representation and supported execution

`HostBuffer` contains a dtype, checked `uint64` element count, and the same narrow retained `StorageHandle` used by TH-007 Tensor storage. Sharing the byte-storage/lifetime machinery does not make the C++ descriptors or language types identical. `ResourceId` remains compiler safety identity, `StorageObjectId` remains deterministic runtime allocation identity, and neither is an address.

Concrete execution is bounded to `Buffer<u8>`:

- zero-length construction and construction from byte values;
- exact checked length and checked byte reads;
- an internal checked mutable write wrapper;
- ordinary C++ handle aliasing that shares storage;
- explicit deep copy to a distinct storage object;
- no copy-on-write;
- retained lifetime through copied handles.

Other represented dtypes can allocate metadata-consistent storage but element execution is deferred. Byte count multiplication and index bounds are checked before access. Failures use internal categories such as `TH009-BUFFER-BOUNDS`, `TH009-BUFFER-SIZE-OVERFLOW`, and `TH009-BUFFER-DTYPE-EXECUTION-DEFERRED`; no unfrozen `TH-SPEC-*` Buffer ID is fabricated. Access uses byte storage and `memcpy`, not unverified typed pointer arithmetic or reinterpret casts. Debug output contains deterministic storage identity and never an address.

## Alias, copy, mutation, and ownership

An ordinary runtime alias retains the same storage object. Mutation through the low-level move-only `MutableHostBufferRef` is visible through that alias. A prior deep copy retains the old bytes and a different `StorageObjectId`; mutation never triggers hidden COW.

The mutable wrapper is infrastructure for a future lowering of `TH-006 exclusive mutable proof -> checked Buffer write`. Its constructor does not replace TH-006 borrow analysis, and handle counts do not grant source mutation permission. A retained handle prevents physical use-after-free, but it cannot legalize a source view escape, conflicting mutable access, or stale logical resource.

The frozen documents specify the Buffer/Tensor distinction and ownership modes, but do not yet specify a sufficiently complete public Buffer construction/access surface. Therefore TH-009 exposes no new source syntax. General Buffer values, record elements, growth, conversions, and public APIs remain deferred.
