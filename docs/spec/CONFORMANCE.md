# TH-003 exact-i64 conformance substrate

This is a language-owned developer oracle for the `v0-design` semantic contract, not an implemented Thiran parser, interpreter, compiler, or runtime. Normative meaning comes from `docs/language/SEMANTICS_V0.md` and decisions TH-LD-003 through TH-LD-011, with edition/stability boundaries in `COMPATIBILITY.md`. PyTorch, NumPy, Graph IR, and native libraries are not semantic authorities.

## Canonical result and case contract

`tests/spec/v0/cases.json` has schema `thiran-spec-cases-1`. Each case directly names one operation (`construct`, `add`, `index`, `slice`, `matmul`, `sum`) and contains an ID, `v0-design` edition, explicit i64 input value objects, manually frozen `expected`, and a rationale naming a frozen decision. This is a finite operation description, not a second source language. Tensor inputs/results have explicit rank-1/2 `shape` and flat logical row-major `values`; scalar values have distinct `kind: scalar` and one `value`. No rank-0 tensor is implied.

Success is `{"status":"ok","kind":"tensor","dtype":"i64","shape":[2,2],"values":[1,2,3,4]}` or `{"status":"ok","kind":"scalar","dtype":"i64","value":2}`. Failure is `{"status":"error","error_id":"TH-SPEC-BOUNDS"}`. The error ID, not host exception text, is the conformance key. Diagnostics may later carry extra context, but adapters must normalize to this exact core result before comparison. These IDs are stable within this fixture schema/edition and do not yet claim the future public compiler diagnostic-ID policy is settled.

Slice selectors are one per axis: an integer removes that axis; `{"start":0,"end":2,"step":1}` retains it. The current fixture requires explicit bounds, even though source notation permits omitted bounds. A successful slice canonicalizes its logical elements; this value observation does not model the view's storage identity or lifetime. Those remain source-level conformance obligations.

The oracle checks shape and extent validity, row-major offsets, i64 input bounds, and checked arithmetic. Matmul checks each product and every accumulation step. Sum checks every accumulation step. For integer arithmetic, this first exact subset admits no numerical tolerance.

Run `python3 -B tools/spec_oracle.py validate tests/spec/v0/cases.json`, `python3 -B tools/spec_oracle.py case C08-01`, and `python3 -B tools/spec_oracle.py --help`. A malformed fixture, duplicate ID, unknown operation, missing expected field, mismatch, or zero executed cases exits nonzero. Expected results are fixture constants; they are not computed from oracle output.

## Future adapter contract

Future adapters for a reference Thiran interpreter, native CPU, native GPU, AOT artifact, and JIT specialization must accept an applicable case or source equivalent, execute the implementation, and normalize its observed result to `status`, `kind`, `dtype`, `shape`/`values` or scalar `value`, or stable `error_id`. Compare each implementation's canonical observation independently with the frozen expected object; the oracle may also be run to audit the fixture. For checked failures, an adapter must map its public checked contract to the semantic ID, never Python/C++ exception names. An adapter must report unsupported cases separately from failures and cannot count unsupported cases as passes. Backend operation support and numeric profiles must be declared before broad equivalence claims. PyTorch may later be an additional differential test, never a replacement authority.

`tests/spec/v0/ownership_future.json` is deliberately non-executable and marked `SPEC_DEFINED` plus `COMPILER_NOT_IMPLEMENTED`. Future parser/checker work can turn those source-level expectations into separately reported tests; their presence does not count as compiler conformance today.
