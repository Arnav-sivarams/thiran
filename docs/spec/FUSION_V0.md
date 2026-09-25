# TH-015 kernel and loop fusion V0

TH-015 fusion is an explicit, deterministic grouping of eligible operations already admitted to a verified straight-line `TensorRegion`. It does not rewrite structured semantic IR, reassociate expressions, simplify algebra, introduce fast math, or make the historical `Graph`/`Region` optimizer semantically authoritative.

## Eligible groups

The V0 fusible subset is same-dtype, same-shape, contiguous elementwise tensor `Negate`, `Add`, `Subtract`, and `ElementMultiply`. A group is a linear producer/consumer chain. Immutable binding aliases may be followed to their common root without creating a kernel boundary. The current tail must have exactly one effective consumer, the next operation must consume it exactly once, and every other next-operation input must already be materialized before the group begins. This excludes branches, diamonds, repeated operands that would alter dependency reasoning, and intervening materializations.

`Index` remains a singleton kernel group. `Copy`, tensor literals, calls, unsupported operations, structured control/scan, mutation/effects, transfers, externally observed intermediates, explicit fusion barriers, AD saves, parameter/state protection, views/possible aliases, and async observation boundaries split groups. A group terminal is materialized according to the memory plan; non-terminal results remain logical typed values evaluated in ordered per-element registers.

## Numerical and failure semantics

The generated operation order is the original TensorRegion order. For f32, CPU and PTX emit one ordinary operation for each IR operation. CPU group-local results use explicit volatile scalar rounding boundaries; PTX uses separate `.rn.f32` instructions. There is no reassociation, approximate instruction, `.ftz`, fast-math flag, or invented fused multiply-add. `(A + B) .* C` therefore computes the addition result before the multiplication for each element.

Every fused i64 operation retains its own overflow check. CPU loops call checked negate/add/subtract/multiply helpers that avoid signed-overflow undefined behavior. GPU PTX records overflow immediately after each ordered operation in the region-wide error flag. Later device instructions may execute after a lane reports overflow, but observation discards the staged result and reports `TH-SPEC-I64-OVERFLOW`; no wrapped result is published.

Shape/layout validation occurs before the group launch/loop writes its terminal output. Zero-element groups allocate no zero-byte device buffer and launch no kernel. Bounds-checked indexing is not fused.

## Backend consumption

Native CPU emission declares one typed scratch vector per owned physical slot and emits one native loop per fusion group. Reused slot names and group-local scalar expressions are present in the generated C++ rather than being diagnostic-only metadata. The optimization-disabled qualification option retains the same emitter but requests singleton groups and a conservative slot plan.

Native GPU emission appends one deterministic PTX entry per elementwise group. Execution resolves its external device tensors, obtains the terminal allocation from the verified physical slot, and launches that entry once for a nonempty group. Planner allocations are owned by one pending execution until TH-014 observation/drain. The disabled option uses singleton PTX groups through the same path. There is no CPU, evaluator, framework, or generated-Python fallback.

TH-016 artifact metadata records the reuse/fusion options and digest of the
actual verified plan. CPU ELF payloads contain its loops/slots; persistent GPU
payloads contain its fused PTX entries. Artifact load recomputes and checks plan
identity before the stored payload can execute, and JIT cache identity includes
both options.

The verifier recomputes legal grouping and rejects missing, duplicated, reordered, ineligible, or malformed groups and materialization boundaries. Group count and physical launch count are structural evidence only; they are not latency, throughput, utilization, or general memory-performance claims.
