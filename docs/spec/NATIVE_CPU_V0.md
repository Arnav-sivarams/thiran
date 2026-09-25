# TH-008 bootstrap native CPU vertical slice

The bounded pipeline is V0 source -> parser -> typed structured semantic IR -> independent semantic verifier -> TH-006 ownership/effect qualification -> TensorRegion extraction and verifier -> deterministic generated C++20 -> installed host C++ compiler -> standalone host executable. Generated C++ is lowering output, not semantic authority. No LLVM or MLIR is introduced. In TH-008 this was a bootstrap developer native-build path, not the versioned persistent AOT/JIT architecture later defined by TH-016, and the C++ backend is not the final backend architecture.

STRICT_NATIVE has no reference-evaluator or legacy fallback. Supported standalone entry is exactly one zero-parameter `fn main()` returning an i64/f32 scalar or rank-1/rank-2 i64/f32 tensor. This developer convention is not a public application ABI. Internal generated signatures use `std::int64_t`, `float`, `const storage::Tensor&`, and `storage::Tensor`; they are unstable and no C FFI is provided.

TH-015 extends the bounded CPU subset to rank-1/rank-2 i64/f32 tensor literals, immutable aliases, explicit copies, equal-shape Negate/Add/Subtract/ElementMultiply, and full-rank Index. A verified physical plan supplies typed scratch-vector slots and explicit fusion groups. Checked i64 helpers cover every ordered negate/add/subtract/multiply operation without C++ signed-overflow undefined behavior. F32 operations retain IR order and use no fast-math/reassociation policy. Full-rank Index uses checked flattened coordinates; invalid coordinates normalize to `TH-SPEC-BOUNDS`. Runtime unequal shapes remain an internal backend-incomplete failure, not a fabricated semantic broadcast failure.

The generated standalone entry prints one canonical TH-003/TH-005 scalar/tensor observation or a canonical supported semantic failure. Unknown C++/storage/compiler defects go to stderr with nonzero status, not a `TH-SPEC-*` language error. No IDs, addresses, paths, or timestamps enter canonical output. Host C++ is required at AOT build time; the finished ELF artifact needs only ordinary host runtime libraries and no Python, PyTorch, NumPy, Triton, generated.py, compiler process, or Thiran shared library. Artifact dependency auditing is required for qualification.

TH-012 structured Scan has no native lowering in this backend. STRICT_NATIVE reports `BACKEND-UNSUPPORTED: Scan lowering deferred` and `fallback: NONE`; it never invokes the reference evaluator.

TH-016 reuses this emitter after TH-015 planning/fusion to create a persistent
ELF shared-object payload behind a typed C ABI. The result is real CPU AOT when
stored in a validated `.tha` container and executed later by the native artifact
runtime. TH-016 CPU JIT invokes the native compiler at JIT time and loads that
payload dynamically; it is external-native-toolchain JIT, not in-process JIT.

Native execution is structural machine-code evidence, not speed evidence. TH-015 slot/group counts do not establish latency, throughput, or general memory savings. No performance superiority, stable native ABI, production CLI route, GPU support, or final backend claim is made.
