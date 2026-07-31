# Project status

Status terms describe repository evidence, not intent.

| Area | Status | Evidence | Limitation | Next research step |
|---|---|---|---|---|
| Frontend | IMPLEMENTED | `src/parser/`, examples and integration tests | Small assignment language and fixed operation set | Expand only with specified semantics and tests |
| Graph IR | IMPLEMENTED | `include/ir/`, `src/ir/`, RegionIRTests | Single-result dataflow Nodes; no control flow | Preserve ownership invariants while evaluating extensions |
| Source modules | IMPLEMENTED | `include/frontend/`, `src/frontend/`, SourceManagerTests, ModuleLinkerTests, ModuleCliTests | Explicit relative imports and exported tensor values only; no functions or packages | Add functions only after module semantics stabilize |
| Tensor functions | IMPLEMENTED | semantic/lowering harnesses and FunctionIntegrationTests | Compile-time top-level functions only; no captures or recursion | Preserve deterministic inlining and provenance |
| Verification | IMPLEMENTED | `GraphVerifier`, RegionVerifier, RegionPlanVerifier | Covers current IR and operation model | Extend alongside any new semantics |
| Optimization | PARTIAL | canonicalization, folding, rewrite, fusion, DCE in normal preparation | Conservative fixed pass set; no global optimizer claim | Measure and specify future transformations |
| Shape analysis | PARTIAL | `ShapeInference`, classification/runtime shape tests | Limited operation rules; `-1` is the dynamic marker | Formalize broader shape semantics |
| Strategy classification | IMPLEMENTED | `BaselineStrategyClassifier`, 38 classification tests | Conservative fixed classifier; no runtime performance selection | Evaluate classifier quality with evidence |
| Region formation | IMPLEMENTED | `TopologicalRegionFormer`, 32 formation tests | Local connected-run baseline; no global merging | Study separate legal coalescing only if justified |
| RegionPlan | IMPLEMENTED | immutable `RegionPlan`, verifier, 36 plan tests | Borrows source Graph and becomes stale after mutation | Maintain freshness contracts for future artifacts |
| Plan CLI | IMPLEMENTED | `--plan`, `--emit-plan`, CLI tests | Text format only; no JSON/DOT plan | Stabilize format only when release needs require it |
| Region execution | IMPLEMENTED | one generated function per Region and explicit boundaries | Every strategy uses PyTorch | Replace individual implementations only after equivalence |
| Tensor-bundle runtime | IMPLEMENTED | restricted load, validation, atomic output tests | Torch `.pt` dictionaries only; CPU-loaded CLI input | Harden compatibility without unrestricted deserialization |
| Installation | IMPLEMENTED | CMake install rules, install preset, staged installation tests | Linux/WSL source installation only; no package manager | Validate future release archives |
| AOT | NOT IMPLEMENTED | No native Region artifact compiler | `AOT` is currently a label | Design and validate a real backend |
| JIT | NOT IMPLEMENTED | No native runtime compiler | `JIT` is currently a label | Design and validate a real backend |
| Cache | NOT IMPLEMENTED | No specialization/artifact cache | No cache reuse | Define keys and correctness before implementation |
| Adaptation | RESEARCH HYPOTHESIS | Strategy and Region infrastructure only | No profiling or runtime switching | Establish measurable policy hypotheses |
| Triton | PARTIAL | emitter/code-generator scaffolding | Not a working execution backend | Prove legal lowering and execution independently |
| Distributed planning | PARTIAL | partition and communication structures in normal path | Descriptive; not Region runtime movement | Define executable semantics |
| Distributed execution | NOT IMPLEMENTED | No executing distributed runtime | No RPC or device-partition execution | Research only after local correctness |
| Benchmarking | NOT IMPLEMENTED | Numerical correctness tests only | No performance result | Create reproducible benchmark methodology |
| Release status | PARTIAL | reproducible local presets/docs/CI workflow | No selected license, tag, release notes, or package | Resolve release governance and packaging |
