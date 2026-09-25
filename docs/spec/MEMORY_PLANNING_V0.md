# TH-015 physical memory planning V0

TH-015 inserts a device-neutral physical plan between verified `TensorRegion` and the native CPU/GPU adapters. The plan is compiler metadata, not source syntax or language semantics. A semantic `ValueId`, TH-006 `ResourceId`, TH-007 `StorageObjectId`, TH-014 `AsyncResourceId`, and `PhysicalSlotId` remain distinct identities. A logical mathematical result is not redefined merely because two non-interfering roots are assigned the same physical slot.

## Requirements and lifetime

Every tensor entry records its logical value, common physical root, dtype, rank, symbolic/concrete shape, checked element and byte count when statically known, natural element alignment, contiguous row-major layout, memory device, inclusive definition/last-obligation positions, classification, materialization status, and optional slot. Concrete element/byte counts use TH-007 checked arithmetic. Negative extents, element-count overflow, byte-count overflow, invalid alignment, insufficient capacity, layout incompatibility, and cross-device assignment reject.

Language liveness and physical lifetime are different. Ordinary consumers establish the minimum root lifetime. Aliases extend their common root lifetime. Region output, externally observed value, view, AD save, parameter, training state, and active async read/write reservation retain the root through the region boundary. Completed-but-unobserved async work is still active because TH-014 releases reservations only on observation. The helper that snapshots an `AsyncResource` therefore records its active read/write obligation even after backend completion.

V0 reuse is deliberately narrow. Only a compiler-created, materialized, contiguous TensorRegion temporary with an independently known root may be reusable. Inputs, outputs, aliases, views, possible aliases, parameters, state, AD saves, externally observed values, and async-reserved roots are non-reusable. An explicit `Copy` has a new root; immutable `Alias` shares the source root. Values eliminated inside a fusion group have a plan entry but no materialization or slot. A statically zero-byte value likewise has no allocation.

## Interference and assignment

Intervals are inclusive because an operation's inputs must remain readable while that operation produces its output. Consequently, a producer ending at instruction `n` interferes with an output born at instruction `n`; V0 does not assume an in-place elementwise kernel. Distinct roots may share a slot only when their physical intervals do not overlap and dtype, rank, exact symbolic shape, layout, alignment, and device match. Dynamic equal-shape slots are rechecked against concrete runtime byte size. Assignment walks definitions in stable region order and selects the lowest compatible slot, so plans and dumps are deterministic.

External slots describe caller-owned physical inputs and are never reused. Owned non-reusable slots describe stable outputs/protected roots. Owned reusable slots describe per-execution scratch. CPU emission turns these slots into typed scratch vectors. GPU execution creates one allocation per used owned slot and shares that allocation only among the slot's proven non-interfering roots. Each asynchronous execution owns a separate slot-allocation map; there is no global allocator pool or cross-execution scratch reuse.

## Verification and observability

`verifyPhysicalPlan` independently recomputes tensor entries, roots, requirements, lifetimes, fusion materializations, and deterministic legal groups from the verified region and obligations. It also validates bidirectional value/slot membership, IDs, capacity, device/layout/dtype compatibility, output/protected-root isolation, zero-size absence, and pairwise interference. Backends reject a plan that does not verify.

The deterministic dump lists every logical tensor value, root, classification, requirement, lifetime, materialization decision, reusable flag, slot, slot capacity/members, and fusion group. Execution evidence separately reports logical intermediates, materialized intermediates, physical slots, reusable temporary slots, reused assignments, fusion groups, fused groups, planner-owned allocations, and currently retained planner allocations. These are structural facts for a particular workload/configuration, not performance claims.

## Conservative boundaries

The planner never flattens structured semantic IR. `Structured::Scan`, control flow, calls, mutation, transfer, I/O, RNG, and unsupported views remain outside a whole-function TensorRegion. TH-010 saved values and TH-011 state are protected through explicit obligation facts; V0 does not attempt a cross-forward/backward or cross-training-step storage schedule. Runtime input aliases discovered by common TH-007 storage identity still share one GPU upload, but that runtime deduplication never grants overwrite permission.
