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

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]  # pragma: no cover

    return [new_tape], null_postprocessing



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


    def step(self, circuit, operator_pool, params_zero=True):
        r"""Update the circuit with one step of the optimizer.

        Args:
            circuit (.QNode): user-defined circuit returning an expectation value
            operator_pool (list[Operator]): list of the gates to be used for adaptive optimization
            params_zero (bool): flag to initiate circuit parameters at zero

        Returns:
            .QNode: the optimized circuit
        """
        return self.step_and_cost(circuit, operator_pool, params_zero=params_zero)[0]

    def rotosolve(self, rot, wire, tape):
        phi = np.random.rand() * 2*np.pi - np.pi
        h = qp.execute([tape.copy(operations=tape.operations + [rot(phi, wire)])], self.dev)[0]
        hp = qp.execute([tape.copy(operations=tape.operations + [rot(phi + np.pi/2, wire)])], self.dev)[0]
        hm = qp.execute([tape.copy(operations=tape.operations + [rot(phi - np.pi/2, wire)])], self.dev)[0]
        return phi - np.pi/2 - np.arctan2(2*h - hp - hm, hp - hm)
    
    def arch_finder_roto(self, circuit, operators_pool, num_layers):
        num_qubits = self.dev.wires
        circuit_args = np.random.uniform(size = num_qubits)
        tape = construct_tape(circuit)(circuit_args)
        for _ in range(num_layers):
            for _ in range(num_qubits):
                best_cost = np.inf
                best_rot = None
                best_phi = None
                best_wire = None
                for rot in operators_pool:
                    for wire in range(num_qubits):
                        phi = rotosolve(rot, wire, tape, dev)
                        cost = qp.execute([tape.copy(operations=tape.operations + [rot(phi, wires=wire)])], self.dev)[0]
                        if cost < best_cost:
                            best_cost = cost
                            best_rot = rot
                            best_phi = phi
                            best_wire = wire
                tape = tape.copy(operations=tape.operations + [best_rot(best_phi, wires=best_wire)])
        
        assert np.isclose(best_cost, min_energy)
        return tape.circuit


    def train_step_grad(self, circuit, operator_pool, drain_pool=False, params_zero=True, circuit_args = None, circuit_target = None):
        r"""Update the circuit with one step of the optimizer, return the corresponding
        objective function value prior to the step, and return the maximum gradient

        Args:
            circuit (.QNode): user-defined circuit returning an expectation value
            operator_pool (list[Operator]): list of the gates to be used for adaptive optimization
            drain_pool (bool): flag to remove selected gates from the operator pool
            params_zero (bool): flag to initiate circuit parameters at zero

        Returns:
            tuple[.QNode, float, float]: the optimized circuit, the objective function output prior
            to the step, and the largest gradient
        """

        qnode = copy.copy(circuit)
        pl_tape, _ = construct_batch(qnode)(circuit_args)

        if drain_pool:
            operator_pool = [
                gate
                for gate in operator_pool
                if all(
                    gate.name != operation.name or gate.wires != operation.wires
                    for operation in pl_tape[0].operations
                )
            ]
        #weights = tf.Variable(tf.zeros(len(operator_pool), dtype = tf.float64))



        return qnode, loss, max(abs(math.toarray(grads)))

    def train_step_roto(self, circuit, operator_pool, drain_pool=False, params_zero=True, circuit_args = None, circuit_target = None):
# The optimization over data is done through the application of continous values correction to the best architecture extracted using rotosolove from a random noise input (e.g. uniform). This process could be boosted using different instances of random noise or different pdfs, less or more inferred from the data.
        qnode = copy.copy(circuit)                          
        pl_tape, _ = construct_batch(qnode)(circuit_args)

        circuit_seed = self.arch_finder_roto(circuit, operator_pool, self.num_layers)














        return qnode, loss, max(abs(math.toarray(grads)))

    def fit(self, circuit, operator_pool, drain_pool, circuit_args, circuit_target):
        if self.algorithm = "roto":
        num_qubits = self.dev.wires
        tape, _ = construct_batch(circuit)(circuit_args)

        else:
            for i in range(len(operator_pool)):
                circuit, energy, gradient = self.train_step(circuit, operator_pool, drain_pool=True, circuit_args = circuit_args, circuit_target = circuit_target)
                if i % 2 == 0:
                    print("n = {:},  E = {:.8f} H, Largest Gradient = {:.3f}".format(i, energy, gradient))
                    print(draw(circuit, decimals=None)(circuit_args[:1]))
                    print()
                #if i > 5:
                #    break
                if energy < 1e-1:
                    break
