# TH-009 internal native-library interop contract

TH-009 introduces internal compiler/runtime metadata for future BLAS, oneDNN, platform math, codec/service, and custom-kernel boundaries. It is not public arbitrary FFI, a stable ABI, or permission for safe source to manufacture pointers or lifetime facts. No `extern`, `unsafe`, raw-pointer, `dlopen`, or user shared-library path syntax is added.

## Contract metadata

A `NativeLibraryContract` records:

- stable-within-this-development-record contract ID, logical library name, and native symbol;
- semantic parameter types and result type;
- parameter access modes `Read`, `MutableBorrow`, or `Consume`;
- explicit TH-006 effects (`MayTrap`, `Mutates`, `RNG`, `IO`, `Transfer`, `Async`);
- a may-trap flag, bounded executable-support flag, and host/backend applicability;
- ordered structured link items.

Link item kinds are static archive path, shared library name, and search path. They become separate compiler argv entries. Raw linker command strings are not accepted. TH-009 qualification needs only a static archive.

## Independent validation

The verifier accepts manually built metadata and does not rely on a registration helper. It rejects duplicate/empty IDs or logical symbol identities, invalid/empty symbols, invalid semantic parameter or result types, scalar `MutableBorrow`, mutable access without `Mutates`, `Mutates` without a mutable/consuming resource parameter, may-trap/effect disagreement, unknown effect bits, malformed link items, and unsupported results falsely marked executable. The driver separately rejects contracts not applicable to its POSIX native-CPU host.

Effects are declarations, not deductions from C symbol spelling. An external operation described as mutation, I/O, RNG, transfer, async, or trapping cannot be treated as Pure. Access modes preserve TH-006 resource obligations; actual future calls will still require ownership/exclusivity proof and checked wrappers.

## Test-only linkage proof

The test-only static library exports deterministic C symbol `thiran_test_add_i64`. A valid pure scalar contract names it and carries its static-archive link item. A C++ integration stub is compiled through the same driver host-toolchain and structured link plan, executed as a new process, and required to print exactly `42`. This proves compiler/linker orchestration and symbol resolution, not source callability. No numerical library coverage, source FFI, stable C ABI, or performance claim follows.
