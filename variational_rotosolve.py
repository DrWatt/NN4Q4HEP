import copy

from pennylane import math
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn
from pennylane.workflow import construct_batch

from pennylane import draw
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

@transform
def append_gate(tape: QuantumScript, params, gates) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Append parametrized gates to an existing tape.

    Args:
        tape (QuantumTape or QNode or Callable): quantum circuit to transform by adding gates
        params (array[float]): parameters of the gates to be added
        gates (list[Operator]): list of the gates to be added

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qp.transform <pennylane.transform>`.

    """
    new_operations = []

    for i, g in enumerate(gates):
        g = copy.copy(g)
        new_params = (params[i], *g.data[1:])
        g.data = new_params
        new_operations.append(g)

    new_tape = tape.copy(operations=tape.operations + new_operations)


class VariationalAdaptiveOptimizer:
    def __init__(self, algorithm, dev, param_steps=20, num_layers = 10, optimizer = None, loss = None):
        self.algorithm = algorithm
        self.dev = dev
        self.param_steps = param_steps
        self.num_layers = num_layers
    #    if optimizer is None:
    #        raise ValueError("Missing training optimizer")
        if loss is None:
            print("Missing Loss function. Using MSE...")
#            self.loss = tf.keras.losses.MeanSquaredError()
    #    else:
    #        self.loss = loss
        self.circuit = circuit

    @pl.batch_input(argnum=0)
    @pl.qnode(dev, interface = 'jax')
    def circuit(dl):
        pl.AngleEmbedding(dl, wires = dev.wires, rotation ="X")
        return pl.expval(obs)

    @staticmethod
    def _circuit(circuit_args, gates,params, initial_circuit):
        """Append parametrized gates to an existing circuit.

        Args:
            params (array[float]): parameters of the gates to be added
            gates (list[Operator]): list of the gates to be added
            initial_circuit (function): user-defined circuit that returns an expectation value
            circuit_args (list): list of the circuit arguments for QML *****

        Returns:
            function: user-defined circuit with appended gates
        """
        final_circuit = append_gate(initial_circuit, params, gates)

        return final_circuit(circuit_args)

    @jax.jit
    def loss(theta, X, y):
        preds = jax.vmap(self.circuit, in_axes = (None,0))(theta, X)
        return jnp.mean(1.0 - y * preds)






    @jax.jit
    def rotosolve(self, rot, wire, tape):
        phi = jnp.random.rand() * 2*jnp.pi - jnp.pi
        h = qp.execute([tape.copy(operations=tape.operations + [rot(phi, wire)])], self.dev)[0]
        hp = qp.execute([tape.copy(operations=tape.operations + [rot(phi + jnp.pi/2, wire)])], self.dev)[0]
        hm = qp.execute([tape.copy(operations=tape.operations + [rot(phi - jnp.pi/2, wire)])], self.dev)[0]
        return phi - jnp.pi/2 - jnp.arctan2(2*h - hp - hm, hp - hm)
    
    @jax.jit
    def train_step_roto(self, circuit, operator_pool, circuit_args = None):

        def update_parameter(i, current_theta):

            def evaluate_at(angle):
                new_theta = current_theta.at[i].set(angle)

                return loss(new_theta, X, y)
        
            L0 = evaluate_at(0.0)
            L90 = evaluate_at(jnp.pi /2)
            L180 = evaluate_at(jnp.pi)

            a = (L0 - L180) / 2.0
            b = L90 - (L0 + L180) / 2.0

            optimal_angle = jnp.arctan2(-b, -a)
            # If the objective is effectively constant along
            # this coordinate, retain the current parameter.
            amplitude = jnp.hypot(a, b)

            optimal_angle = jnp.where(
                amplitude > 1e-12,
                optimal_angle,
                current_theta[i]
            )
            return current_theta.at[i].set(optimal_angle)
        
        theta = jax.lax.fori_loop(
            0,
            theta.shape[0],
            update_parameter,
            theta
        )
        
        return theta
    

    def fit(self, operator_pool, circuit_args, label):
        if self.algorithm = "roto":
            for n in self.num_layers:
                for i in range(len(operator_pool)):
                    for w in dev.wires:
                        self.circuit = append_gate(self.circuit, 0.0, operator_pool[i])
                        circuit, energy = self.train_step_roto(circuit, operator_pool, circuit_args = circuit_args, loss = self.loss)
                        if i % 2 == 0:
                            print("n = {:},  E = {:.8f} H, Largest Gradient = {:.3f}".format(i, energy, gradient))
                            print(draw(circuit, decimals=None)(circuit_args[:1]))
                            print()
                        #if i > 5:
                        #    break
                        if energy < 1e-1:
                            break
