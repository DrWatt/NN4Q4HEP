import copy

from pennylane import math
from pennylane import numpy as pnp
from pennylane._grad import grad
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn
from pennylane.workflow import construct_tape

from pennylane import GradientDescentOptimizer


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
    def __init__(self, param_steps=10, stepsize=0.5):
        self.param_steps = param_steps
        self.stepsize = stepsize

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



    def step_and_cost(self, circuit, operator_pool, drain_pool=False, params_zero=True, circuit_args = None):
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
        if True:
            cost = circuit(circuit_args)
        else:
            cost = circuit()
        qnode = copy.copy(circuit)
        tape = construct_tape(qnode)(circuit_args)

        if drain_pool:
            operator_pool = [
                gate
                for gate in operator_pool
                if all(
                    gate.name != operation.name or gate.wires != operation.wires
                    for operation in tape.operations
                )
            ]

        params = pnp.array([gate.parameters[0] for gate in operator_pool], requires_grad=True)
        circuit_args_np = pnp.array(circuit_args, requires_grad = False)
        qnode.func = self._circuit
        grads = grad(qnode)(params, gates=operator_pool, initial_circuit=circuit.func, circuit_args = circuit_args_np)

        selected_gates = [operator_pool[pnp.argmax(abs(grads))]]
        optimizer = GradientDescentOptimizer(stepsize=self.stepsize)

        if params_zero:
            params = pnp.zeros(len(selected_gates))
        else:
            params = pnp.array([gate.parameters[0] for gate in selected_gates], requires_grad=True)

        for _ in range(self.param_steps):
            params, _ = optimizer.step_and_cost(
                qnode, params, gates=selected_gates, initial_circuit=circuit.func, circuit_args = circuit_args_np
            )

        qnode.func = append_gate(circuit.func, params, selected_gates)

        return qnode, cost, max(abs(math.toarray(grads)))


