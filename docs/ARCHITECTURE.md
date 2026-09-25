# Thiran architecture

## Current pipeline

```text
.th source
  -> Lexer and Parser
  -> Graph (owns Nodes)
  -> GraphVerifier
  -> canonicalization, constant folding, rewriting, fusion, DCE
  -> GraphVerifier
  -> ShapeInference
  +-> planning path
  |    -> BaselineStrategyClassifier
  |    -> TopologicalRegionFormer
  |    -> immutable RegionPlan + RegionPlanVerifier
  |    -> plan text or RegionPythonEmitter
  |    -> generated Region runtime -> PyTorch
  |
  `-> normal path
       -> analyses and execution scheduling
       -> device, partition, and communication descriptions
       -> BackendIR and PythonExecutorEmitter
       -> generated.py -> PyTorch when run separately
```

## Frontend

The CLI routes source programs through `frontend::SourceManager`, source-spanned tokens, per-module AST parsing, deterministic module linking, function declaration collection, semantic and recursion validation, and flattening into one ordinary Graph. Functions are compile-time-only and retain side-table provenance; no function construct enters Graph or Region IR. Imported modules are lowered before importers and canonical files are lowered once. The legacy parser remains a compatibility surface. Frontend failures occur before Region planning.

## Graph IR ownership and verification

`Graph` owns Nodes through `std::unique_ptr` in Graph storage order. Nodes borrow producer and consumer pointers; Regions never own Nodes. `GraphVerifier` checks ownership, adjacency, arity, operation validity, unique names, and acyclicity. Graph storage order is a deterministic tie-break, not a substitute for dataflow order.

## Graph preparation and shape metadata

`src/main.cpp::prepareGraph` is the shared planning-safe preparation path. It runs parsing, verification, canonicalization, constant folding, rewrite rules, fusion, dead-code elimination, a second verification, and shape inference. Planning modes branch after this point and before scheduler/backend work.

`Node::shape` is the shape consumed by strategy classification and Region Python emission. Empty Input shape represents unknown rank in the generated runtime. `-1` is a runtime-specializable dimension.

## Strategy classification

`BaselineStrategyClassifier` assigns exactly one deterministic decision to each live Node. It uses Graph IR and resolved shape metadata. Classification labels are `AOT`, `JIT`, and `FALLBACK`; they are planning labels, not distinct runtime implementations.

TH-016 native AOT/JIT is a separate V0 subsystem below verified TensorRegion
and PhysicalPlan. It does not reinterpret these historical Region strategy
labels or make the legacy Graph/Python route a native backend.

TH-017 composes the independent V0 semantic/AD/training path with that native
artifact subsystem. A validated training checkpoint reload produces a frozen
deployment snapshot; a model bundle then stores typed public-input and
`ParameterId` bindings, frozen parameter bytes, and CPU/GPU `.tha` artifacts.
The installed `thiran-model` runtime validates the bundle, supplies frozen
parameters internally, and executes the selected native backend without the
legacy Graph/Python route or reference evaluator.

TH-019 adds a separate typed native CPU graphics runtime below the same storage
and ownership architecture. Rank-two tensors carry geometry, colors, and
matrices; the runtime returns fresh rank-two/rank-one framebuffer tensors.
Transform math is naturally eligible for future TensorRegion planning, but
homogeneous clipping, raster traversal, ordered depth tests, and exclusive
framebuffer mutation remain structured loops. The renderer is not a scene
graph, does not use the historical Graph/Region path, and is not forced into a
whole-renderer dataflow DAG or TH-016 artifact ABI.

TH-020 adds a second implementation of that same graphics contract through the
TH-013 CUDA Driver/PTX substrate. Shared CPU preparation validates, transforms,
clips, and creates a deterministic screen-triangle packet. A backend-owned
one-work-item-per-pixel kernel performs coverage, depth, and perspective color
on the physical GPU. TH-014 reservations retain Tensor inputs and execution
resources until observation. Observation crosses an explicit D2H boundary and
publishes ordinary Tensor storage; later native GPU TensorRegion work may H2D
that storage again. There is no zero-copy claim and rasterization is not forced
into TensorRegion or the TH-016 artifact ABI.

## Region formation

`TopologicalRegionFormer` computes a deterministic Kahn ordering using Graph storage index as the ready-node tie-break. It forms contiguous, strategy-homogeneous Regions. A Node joins the current Region only when a direct producer is already inside it, which guarantees weak connectedness of each induced Region subgraph.

## Immutable RegionPlan

`HybridRegionPlanner::build` classifies, forms, finalizes, and verifies a plan. `RegionPlan` owns its classification decisions and `RegionGraph`; it borrows its source Graph, which must outlive the plan. The plan is immutable after construction. `RegionPlanVerifier` detects stale plans and source mismatches.

## RegionPlan verification and diagnostics

Classification (`SC`), formation (`RF`), plan verification (`RP`), and Region Python emission (`RPE`) retain separate deterministic diagnostic layers. Printer utilities format but do not create or reorder diagnostics.

## Region Python emission

`RegionPythonEmitter` verifies the exact Graph and plan, computes Region dependency order, and emits one function per Region in Region storage order. Region calls follow deterministic dependency order. Graph-index symbols (`v_0`, `v_1`, …) represent tensor values. Crossing values are explicit parameters and tuple returns. Graph Inputs and Constants are materialized in their owning Regions; Output Nodes alias their operand.

## Generated runtime

The generated artifact is import-safe and exposes:

- `run_region_plan(inputs)`;
- `thiran_input_spec()`;
- `thiran_output_names()`;
- a guarded command-line runtime.

Bundle mode uses restricted `torch.load(..., weights_only=True)`, validates names, tensors, shapes, and devices, executes the RegionPlan, detaches outputs to CPU, and atomically replaces the output bundle. All strategies currently use the same correctness-oriented PyTorch operations.

## Existing whole-Graph path

Normal `Thiran <source-file>` continues through analyses, scheduler structures, device planning, partition and communication planning, BackendIR construction, and `PythonExecutorEmitter`. It writes `graph.dot` and `generated.py`. Thiran does not automatically run Python; the user invokes the generated artifact separately.

## Component reality

- Scheduler and device assignments are functional as descriptive compiler stages in normal mode, but do not select Region runtime implementations.
- Partition and communication graphs are constructed and printed; they do not execute distributed tensor movement.
- BackendIR describes grouped kernels; it is not an in-process executable tensor runtime.
- `Executor` is not the semantic normal execution path.
- Triton emitters are scaffolding/stubs, not a working execution backend.
- PyTorch emission remains the legacy Graph/Region execution path. The bounded
  V0 TensorRegion path also has native CPU/GPU artifacts and TH-017 model-bundle
  execution; neither path silently falls back to the other.
- The graphics runtime has independent native CPU reference and native CUDA
  compute implementations over V0 storage. It has no Python/framework or
  external graphics renderer, and GPU selection never falls back to CPU.

## Ownership and lifetime

- Graph owns Nodes.
- RegionGraph owns Regions.
- Regions borrow Nodes.
- RegionPlan borrows its source Graph and owns immutable planning products.
- RegionPythonArtifact owns generated source and retains no Graph, Node, Region, or plan pointer.
- Generated Python owns runtime dictionaries and tensor references supplied by its caller.

## Deterministic ordering

- Graph storage order: stable identity, input/output metadata, and tie-breaking.
- Topological Graph order: Region formation and Node execution legality.
- Region storage order: function definitions and IDs.
- Region dependency order with minimum Region ID tie-break: runtime calls.
- Stored Region input/output order: function parameters and tuple returns.
- Diagnostic category and storage order: visible error ordering.

No visible order depends on pointer values or unordered-container iteration.

## Future replacement points

The legacy Graph/Region Python path could still replace an individual generated
Python body in the future; that remains only a planned replacement point.
TH-016 native artifacts instead operate on the independently verified V0
TensorRegion/PhysicalPlan path. TH-017 packages those artifacts with validated
frozen checkpoint parameters for reference-model deployment. Neither makes a
performance claim.

TH-019 deliberately stays outside historical Graph/Region and the current
artifact ABI. A future source-level library surface can lower tensor-eligible
transform subregions normally while preserving structured clipping/raster
loops and explicit mutation; no graphics-specific source language is implied.
