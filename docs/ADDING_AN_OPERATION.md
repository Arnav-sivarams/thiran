# Adding an operation

Adding an enum value alone is incomplete. Every exhaustive consumer must either implement the operation or reject it deterministically.

## Checklist

1. Add the operation to `Operation` in `include/ir/Operation.hpp` and its stable name in `src/ir/Operation.cpp::toString`.
2. Add the spelling to `BuiltinOperations` and the source-spanned frontend parser when syntax should expose it. Review whether it is legal inside functions. Keep the legacy compatibility parser exhaustive where its public tests require it.
3. Define and verify arity in `src/optimizer/GraphVerifier.cpp`.
4. Add semantic validation for ownership, attributes, constants, or other invariants in the verifier/parser layer that owns them.
5. Implement shape behavior in `src/analysis/ShapeInference.cpp`; report invalid ranks and dimensions deterministically.
6. Review canonicalization, constant folding, rewrite, fusion, and DCE implications under `src/optimizer/`. Do not add a rewrite without legality and regression tests.
7. Update classification exhaustiveness in `src/region/StrategyClassification.cpp`. Classification decides strategy; formation must remain strategy-agnostic.
8. Add the accepted whole-Graph PyTorch lowering in `src/backend/PythonExecutorEmitter.cpp` and any descriptive BackendIR handling actually required by normal mode.
9. Add exact Region lowering, arity, and metadata checks in `src/region/RegionPythonEmitter.cpp`.
10. Add C++ tests at every affected layer and CLI/integration coverage under `tests/integration/`.
11. Compare deterministic inputs numerically between whole-Graph and Region-controlled PyTorch, including keys, shapes, dtypes, and values.
12. Update operation lists, limitations, CLI examples, architecture notes, and project status where behavior changes.

## Defensive failure

`GraphVerifier` rejects unknown or malformed operations before planning. `RegionPythonEmitter` defensively returns `RPE002` for an unsupported operation and `RPE003` for invalid arity; it returns no partial artifact. Never emit a comment and continue.

## Validation

Run Debug, warnings, sanitizers with leak detection, Release, numerical integration, and normal-mode regression. Follow [TESTING.md](TESTING.md).
