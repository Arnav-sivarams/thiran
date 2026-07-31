# Graph IR

**IMPLEMENTED:** Each `Node` has an operation, stable ID, name, ordered inputs, outputs, shape, device, attributes, and optional scalar constant value. `Graph` owns Nodes, records explicit edges, and provides connect, disconnect, and remove operations.

**IMPLEMENTED:** Verification enforces unique names, Graph ownership and adjacency, known operations, operation arity, and acyclicity. Shape inference validates supported operation dimensions before lowering.

**ENGINEERING LIMITATION:** Nodes model one result each, the current language has no control-flow operations, and shape semantics cover only the accepted operation set.

**PLANNED / RESEARCH HYPOTHESIS:** Broader value and control-flow models require separate specifications and evidence; they are not current Graph behavior.
