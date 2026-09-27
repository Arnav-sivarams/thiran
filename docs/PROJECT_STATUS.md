# Project status

Thiran 0.1.0 completes the current compiler/runtime phase through TH-025 as a
scoped experimental technical release. TH-024 supplies bounded robustness and
same-environment reproducibility evidence; TH-023 supplies the performance
evidence and records substantial CPU/GPU performance debt. This status is not
a security proof, cross-machine claim, cross-GPU portability claim, or stable
1.0 compatibility promise.

Status terms describe repository evidence, not intent.

| Area | Status | Evidence | Limitation | Next research step |
|---|---|---|---|---|
| Frontend | IMPLEMENTED | `src/parser/`, examples and integration tests | Small assignment language and fixed operation set | Expand only with specified semantics and tests |
| Graph IR | IMPLEMENTED | `include/ir/`, `src/ir/`, RegionIRTests | Single-result dataflow Nodes; no control flow | Preserve ownership invariants while evaluating extensions |
| Source modules | IMPLEMENTED | `include/frontend/`, `src/frontend/`, SourceManagerTests, ModuleLinkerTests, ModuleCliTests | Explicit relative imports; no package ecosystem | Preserve deterministic linking and provenance |
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
| Developer CLI | IMPLEMENTED (V0 SUBSET) | `thiran` check/run/build/artifact/model, CPU and physical-GPU workflow tests | Zero-parameter source entry; built-repository workflow only | Review before broader I/O or distribution work |
| Native AOT | IMPLEMENTED (V0 SUBSET) | `.tha` container, ELF/PTX payloads, typed loader, fresh-process CPU/GPU tests | Bounded TensorRegion subset; PTX uses CUDA Driver JIT | Expand only with typed backend coverage |
| Native JIT | IMPLEMENTED (V0 SUBSET) | external-toolchain CPU JIT and Thiran-PTX/CUDA Driver GPU JIT | CPU compilation is out-of-process; no LLVM JIT | Reassess in-process lowering only with evidence |
| JIT cache | IMPLEMENTED (LOCAL V0) | deterministic specialization keys and hit/miss tests | In-memory/process-local only | Preserve full correctness identity if extended |
| Training checkpoints | IMPLEMENTED (REFERENCE V0) | `.thc` schema/integrity validation, transactional save, exact destroy/reload/resume tests | TH-011 f32 state and local files only; no authentication | Preserve explicit schema and optimizer-state compatibility |
| Model deployment | IMPLEMENTED (REFERENCE V0) | `.thm` bundle, `thiran-model`, relocated CPU and physical-GPU equivalence tests | Narrow affine public ABI; trusted local native payloads; no service | Generalize only from additional accepted models |
| Numerical graphics | IMPLEMENTED (CPU + NATIVE GPU V0) | shared typed Tensor geometry/clipping, CPU reference, CUDA Driver/PTX pixel raster/depth/interpolation, deterministic cube, async/interoperability tests | Runtime/library API only; GPU outputs materialize through D2H and later GPU numerics H2D again; no zero-copy, scene graph, source syntax, artifact ABI, AD, or performance claim | Review the structured CPU/GPU boundary before broader graphics work |
| TH-018 | DEFERRED / OPTIONAL | checkpoint intentionally not accepted as required V0 scope | No implied missing release feature | Revisit only by separate authorization |
| Performance | QUALIFIED (TH-023 V0 SCOPE) | frozen methodology, machine-readable evidence, optimized C++/framework controls, physical GPU | Current CPU and transfer-inclusive GPU paths are substantially slower on the qualified workload; not a broad competitiveness claim | Preserve evidence; optimize only in a future authorized checkpoint |
| Robustness / reproducibility | QUALIFIED (TH-024 V0 SCOPE) | frozen methodology, machine-readable evidence, deterministic CPU/physical-GPU repetition, corruption, transactional writes, concurrency, soaks, sanitizers, and clean-copy qualification | One local machine and one RTX 3050 Ti architecture; real GPU OOM, cross-machine, cross-GPU, LSan, and Compute Sanitizer are not qualified | Conduct TH-024 architectural review before any release decision |
| Adaptation | RESEARCH HYPOTHESIS | Strategy and Region infrastructure only | No profiling or runtime switching | Establish measurable policy hypotheses |
| Triton | PARTIAL | emitter/code-generator scaffolding | Not a working execution backend | Prove legal lowering and execution independently |
| Distributed planning | PARTIAL | partition and communication structures in normal path | Descriptive; not Region runtime movement | Define executable semantics |
| Distributed execution | NOT IMPLEMENTED | No executing distributed runtime | No RPC or device-partition execution | Research only after local correctness |
| Benchmarking | IMPLEMENTED (TH-023 QUALIFICATION) | canonical result and structural validator | One machine/workload family; no marketing generalization | Add workloads only under frozen methodology |
| Release status | QUALIFIED (TH-025 V0.1 SCOPE) | version authority, release/support docs, Apache-2.0 license, examples, ABI/manifest checks, clean source-copy builds and CPU/GPU release smoke | Source/repository release only; no release commit/tag/push in TH-025 and no portable binary package | Human final review, then separately commit/tag/push |

## Future work (not implemented)

- **TH-026:** portable single-file and cross-machine deployment.
- **Future proof-system work:** theorem/proof infrastructure remains outside the
  current compiler phase.
- **Future local intelligence:** System-1, ThiranQL, and local-AI capabilities
  are not part of Thiran 0.1.0.
