# Tensor functions

Thiran supports top-level compile-time tensor functions. A function is private unless declared with `export fn`:

```thiran
export fn linear_relu(x, weight, bias) {
    product = MatMul(x, weight)
    biased = Add(product, bias)
    result = ReLU(biased)
    return result
}
```

Imports precede functions, and all functions precede module tensor assignments. Calls are unqualified within a module or qualified through an explicit import alias. Function declarations are collected before bodies are resolved, so forward calls are valid.

Parameters and locals are tensor values. Locals are immutable, cannot shadow parameters or earlier locals, and each body has exactly one final `return name`. Direct return expressions, nested call expressions, literals as function-call arguments, and qualified tensor arguments are not supported. Zero-parameter functions and returning a parameter are supported.

Functions cannot capture module or imported tensor values: every dependency is an explicit parameter. `Constant` may appear in a body; `Input` and `Output` may not. Private functions cannot be called through imports. Recursion of every length is rejected with a deterministic call chain.

Calls are expanded deterministically into one ordinary Graph. Parameters, calls, returns, and identity aliases create no Graph Nodes. An identity return binds the caller name to the existing value. Materialized intermediate names use deterministic call, function, and local ordinals; provenance records the defining module, declarations, immediate call site, and full inline chain separately from Graph and Node.

Limits are 4096 functions per module, 256 parameters, 4096 body assignments, 256 expansion depth, 1,000,000 expanded Nodes, 100 diagnostics, and 1024 bytes for a generated internal name. These bounds do not claim complete safety against malicious input.

There are no closures, mutation, runtime function objects, overloads, default or variadic arguments, multiple returns, explicit tensor types, generics, classes, records, methods, native calls, or native AOT/JIT.
