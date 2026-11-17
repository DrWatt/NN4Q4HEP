import pennylane as pl
import jax
import jax.numpy as jnp

from variational_rotosolve import VariationalAdaptiveOptimizer

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------
# 1. Synthetic dataset
# --------------------------------------------------

X = jnp.array([
    [0.0, 0.0],
    [0.0, 0.5],
    [0.5, 0.0],
    [0.5, 0.5],
    [2.5, 2.5],
    [2.5, 3.0],
    [3.0, 2.5],
    [3.0, 3.0],
])

y = jnp.array([
    -1, -1, -1, -1,
     1,  1,  1,  1
], dtype=jnp.float64)


# --------------------------------------------------
# 2. Quantum device
# --------------------------------------------------

dev = pl.device(
    "default.qubit",
    wires=2,
    shots=None
)


# --------------------------------------------------
# 3. Initialize adaptive optimizer
# --------------------------------------------------

optimizer = VariationalAdaptiveOptimizer(
    algorithm = "roto",
    dev=dev,
    param_steps=5,
    num_layers=2,
    cz_edges=[(0, 1)],
readout_wire = 0
)


# --------------------------------------------------
# 4. Native gate pool
# --------------------------------------------------

operator_pool = [
    pl.RZ,
    pl.CZ,
    pl.RX
]


# --------------------------------------------------
# 5. Train
# --------------------------------------------------

circuit, theta_opt, history = optimizer.fit(
    operator_pool=operator_pool,
    X=X,
    y=y,
    optimize_each_gate=False
)


# --------------------------------------------------
# 6. Evaluate predictions
# --------------------------------------------------

scores = jax.vmap(
    circuit,
    in_axes=(None, 0)
)(theta_opt, X)

predictions = jnp.where(
    scores >= 0.0,
    1.0,
    -1.0
)

accuracy = jnp.mean(predictions == y)


# --------------------------------------------------
# 7. Print results
# --------------------------------------------------

print("\n===== RESULTS =====")

print("Optimized parameters:")
print(theta_opt)

print("\nLoss history:")
print(history)

print("\nPredictions:")
print(predictions)

print("\nTrue labels:")
print(y)

print(f"\nTraining accuracy: {float(accuracy):.2%}")


# --------------------------------------------------
# 8. Draw optimized circuit
# --------------------------------------------------

print("\nOptimized circuit:\n")

print(
    pl.draw(circuit, decimals=3)(
        theta_opt,
        X[0]
    )
)
