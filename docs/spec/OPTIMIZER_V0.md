# TH-011 reference optimizer transitions

TH-011 optimizers are deterministic runtime/library algorithms applied after TH-010 backward completes. They neither modify generated backward IR nor introduce source operators. An optimizer consumes an ordered parameter state, ordered checked gradients, and explicit optimizer state and produces a complete new logical state transactionally.

## Supported values and arithmetic

The exact domain is f32 scalar, `Tensor<f32,1>`, and `Tensor<f32,2>`. Parameter, gradient, and velocity require identical semantic type, rank, and concrete shape. Small internal zero, scale, add, and subtract helpers operate elementwise in binary32 program order without implicit dtype conversion, reassociation, fast math, NumPy, or another framework.

IEEE NaN and infinity propagate through optimizer arithmetic; gradients are not automatically rejected or clipped. Hyperparameters are different: learning rate and momentum must be finite because they configure the transition. No numerical-stability guarantee is claimed.

## Plain SGD

For every parameter in plan order:

```text
p_next = p - learning_rate * gradient
```

`learning_rate` is f32, finite, and at least zero. Zero is legal. A zero gradient leaves the parameter unchanged. Plain SGD stores optimizer kind and step but no velocity vector.

## SGD with momentum

The exact convention is classical momentum without dampening, Nesterov, or weight decay:

```text
v_next = momentum * v + gradient
p_next = p - learning_rate * v_next
```

Initialization uses an explicit `ZeroLike(parameter)` velocity for every ordered ParameterId. `momentum` is finite with `0 <= momentum < 1`. Because initial velocity is zero, the first parameter update equals plain SGD at the same learning rate. A zero gradient does not necessarily leave a parameter unchanged: an existing velocity decays by momentum and still contributes.

## Explicit state and validation

`OptimizerState` contains optimizer kind, step, and—only for momentum—an ordered velocity descriptor/value per ParameterId. Correctness never uses addresses or unordered-map iteration. The independent audit rejects unknown kind, NaN/infinite/negative learning rate, invalid momentum, velocity count/type/shape mismatch, identity/order mismatch, and optimizer step or plan mismatch.

All gradients and all next velocities/parameters are validated or constructed in temporary values before a successful state is published. There is no partial update if any item fails. Input values are not mutated and semantic in-place reuse is not required. Physical reuse may be considered only by later verified storage lowering.

Adam, AdamW, RMSProp, Adagrad, LAMB, weight decay, dampening, Nesterov momentum, gradient clipping, loss scaling, and mixed precision are deliberately deferred. The bounded enum is an internal reference API, not a final source optimizer surface.
