# TH-004 V0 source syntax (developer checkpoint)

This document describes the **implemented TH-004 syntax subset** under the frozen language constitution. Parsing is not type checking, ownership checking, execution, or semantic conformance. The existing R12 CLI still uses its legacy grammar. V0 is tested independently and is not a stable language edition yet.

## Lexical rules

Identifiers begin with an ASCII letter or `_` and continue with ASCII letters, digits, or `_`. Integer literals are nonempty decimal digit sequences; their value/range and contextual dtype are not checked here. Decimal real forms `digits.digits` are tokenized but rejected as expressions in TH-004, pending typing and literal conversion. Import paths use double-quoted strings; a backslash protects the next non-newline character lexically, but decoding/UTF-8/path validation is deferred. Unterminated strings are diagnosed. Only `//` line comments exist. Whitespace other than newlines is ignored.

Keywords used in grammar: `let`, `mut`, `fn`, `export`, `return`, `import`, `as`, `true`, `false`, `if`, `else`, `for`, `in`, `while`, `break`, `continue`. `struct` remains reserved but unsupported.

Operators/punctuation: `+ - * .* / ./ = : , ; . ( ) [ ] { } < > ->`. Lexing uses maximal munch: `.*`/`./` precede `.`, and `->` precedes `-`. Unsupported characters cause located lexical diagnostics; malformed sequences of individually valid tokens cause located parser diagnostics.

## Separators, statements, declarations

Newlines or explicit semicolons separate ordinary module items and function-body statements. Blank lines/comments are not AST statements. Within parentheses or brackets, newlines are insignificant and expressions may span lines. In a tensor literal, semicolon separates rows, not statements. A statement on one physical line must be separated from the next with `;` unless a newline intervenes.

Module items in this subset are imports, functions, and `let` bindings. Function bodies and nested braced blocks contain `let`, `return`, simple-identifier rebinding, `if`/`else`, range/iterable `for`, `while`, `break`, and `continue`. Expression statements are not supported. The parser represents rebinding (`x = expr`) as a distinct `RebindStmt`; the semantic analyzer validates mutability.

```text
binding        = "let" ["mut"] identifier "=" expression
rebind         = identifier "=" expression      // function body only
return         = "return" expression
if             = "if" expression block ["else" block]
for-range      = "for" identifier "in" expression ":" expression block
for-iterable   = "for" identifier "in" expression block // parser only; semantic stage diagnostic
while          = "while" expression block
loop-exit      = "break" | "continue"
block          = "{" statements "}"
import         = "import" string "as" identifier
function       = ["export"] "fn" identifier "(" parameters ")"
                 ["->" type] "{" statements "}"
parameters     = [parameter {"," parameter} [","]]
parameter      = identifier ":" type
```

Every parameter requires a type. An exported function requires a result type. A private function result may be omitted syntactically and inferred when acyclic. Function and nested blocks require a closing brace and ordinary newline/semicolon statement separation. A newline immediately before `else` belongs to its preceding `if`. `if` is a statement, not an expression. Every new structured AST node retains a `SourceSpan`.

## Type syntax

```text
type           = identifier
               | "Tensor" "<" type "," decimal_integer ">"
               | "Buffer" "<" type ">"
               | "(" type "," type {"," type} [","] ")"  // result position
```

Examples: `i64`, `f32`, `bool`, `Tensor<i64, 2>`, `Tensor<f32, 3>`, `Buffer<u8>`, `(Tensor<i64, 2>, State)`. The AST retains type spelling/structure; it does not instantiate R12 `TensorType` or determine whether a named type exists. Generic forms other than `Tensor`/`Buffer`, constraints, and generic function declarations are unsupported.

## Expressions and precedence

Primary expressions are identifier, integer, `true`/`false`, parenthesized expression, tuple expression, and tensor literal. A comma in parentheses constructs a tuple; plain parentheses group an expression. Unary minus is supported. Postfix call, index/slice, and member selection can chain, including `alias.name` and `A.T`.

| Binding strength | Operators | Associativity |
| --- | --- | --- |
| Highest | call `(...)`, index `[...]`, member `.name` | chained left to right |
| Next | unary `-` | prefix |
| Next | `*`, `.*`, `/`, `./` | equal precedence, left |
| Lowest | `+`, `-` | equal precedence, left |

`A + B * C` parses as `A + (B * C)`. `A * B .* C` parses as `(A * B) .* C`. This table says nothing about operand types, matmul legality, or division semantics; those are later checks. Comparisons, logical operators, casts, assignments-as-expressions, and control expressions are unsupported.

## Tensor literals and selectors

```text
tensor         = "[" expression {"," expression}
                 [";" expression {"," expression}] "]"
index          = expression "[" selector {"," selector} "]"
selector       = expression | [expression] ":" [expression] [":" [expression]]
```

`[1, 2, 3]` retains one row (rank-1 intent); `[1, 2; 3, 4]` retains two rows (rank-2 intent). A ragged two-row literal is rejected structurally. Empty rows, trailing commas, and more than two rows are rejected. Numeric element homogeneity and actual tensor rank/type semantics are deferred.

An index selector is `Index(expr)`. A slice selector is `Slice(start?, end?, step?)`, with absent fields stored explicitly. Examples: `A[0]`, `A[0, 1]`, `A[:, 1]`, `A[0:2]`, `A[0:2, 1:3]`, `A[0:4:2]`. A third colon is invalid. `0:4:-1` parses, but the frozen V0 positive-step semantic rule will reject it later. Bounds, rank, view lifetime, and storage semantics are not checked in TH-004.

## Examples and unsupported surface

```thiran
import "layers.th" as layers
let A = [1, 2; 3, 4]
let B = A
let pair = (A[0, :], A.T)
export fn add(a: Tensor<i64, 2>, b: Tensor<i64, 2>) -> Tensor<i64, 2> {
    return a + b
}
```

Imports are AST declarations only: no file loading, linking, or visibility resolution. Range-for uses a context-specific colon, distinct from type annotations and tensor slice selectors. The iterable-for form is parsed but diagnosed as `TH005C-ITERABLE-FOR-DEFERRED` during semantic analysis; borrowed leading-axis iteration awaits ownership/view work. No `struct` declarations, `const`, function generics, ownership modes, indexed assignment, rank > 2 literals, real-literal expressions, strings outside imports, comparison operators, or `if` expressions are implemented. R12 grammar is not a source of V0 rules.

## Conformance boundary

```text
source text -> TH-004/TH-005C spanned syntax AST
AST -> TH-005/TH-005C typed structured semantic IR -> execution
        -> canonical TH-003 result comparison
```

The TH-003 oracle's 33 operation fixtures remain semantic test infrastructure and are not parser implementation or evidence that these source examples execute.
