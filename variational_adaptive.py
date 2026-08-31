import copy

from pennylane import math
from pennylane import numpy as pnp
from pennylane._grad import grad
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn
from pennylane.workflow import construct_batch

from pennylane import GradientDescentOptimizer
from pennylane import draw
import tensorflow as tf


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
    def __init__(self, param_steps=20, optimizer = None, loss = None):
        self.param_steps = param_steps
        if optimizer is None:
            raise ValueError("Missing training optimizer")
        if loss is None:
            print("Missing Loss function. Using RMSE...")
            self.loss = tf.keras.losses.MeanSquaredError()
        self._opt_config = tf.keras.optimizers.serialize(optimizer)


    def _create_optimizer(self):
        return tf.keras.optimizers.deserialize(self._opt_config)

    @staticmethod
    def _circuit(params, gates, initial_circuit, circuit_args):
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



    def train_step(self, circuit, operator_pool, drain_pool=False, params_zero=True, circuit_args = None, circuit_target = None):
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
        weights = tf.Variable([gate.parameters[0] for gate in operator_pool], trainable=True, dtype = tf.float64)
        qnode.func = self._circuit
        with tf.GradientTape() as tape:
            pred = qnode(weights, gates=operator_pool, initial_circuit=circuit.func, circuit_args = circuit_args)
            loss = self.loss(circuit_target, pred)

        
        
        grads = tape.gradient(loss, weights)

        selected_gates = [operator_pool[tf.argmax(tf.abs(grads))]]
        print("Selected gates: ", selected_gates)

        if params_zero:
            sel_weights = tf.Variable(tf.zeros(len(selected_gates), dtype = tf.float64))
        else:
            sel_weights = tf.Variable([gate.parameters[0] for gate in selected_gates], trainable = True, dtype = tf.float64)
        print(sel_weights)
        opt = self._create_optimizer()

        for _ in range(self.param_steps):
            with tf.GradientTape() as tape:
                pred = qnode(sel_weights, gates = selected_gates, initial_circuit=circuit.func, circuit_args = circuit_args)
                loss = self.loss(circuit_target,pred)

            gradients = tape.gradient(loss, sel_weights)
            opt.apply_gradients([(gradients, sel_weights)])


        qnode.func = append_gate(circuit.func, sel_weights, selected_gates)

        return qnode, loss, max(abs(math.toarray(grads)))

    def fit(self, circuit, operator_pool, drain_pool, circuit_args, circuit_target):
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
