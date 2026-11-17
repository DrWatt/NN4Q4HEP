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
    def __init__(self, algorithm, dev, param_steps=20, num_layers = 10, optimizer = None, loss = None, readout_wire = None, cz_edges = None, new_gate_scale = 0.0, seed = 0):
        if algorithm.lower() not in ("roto", "rotosolve"):
            raise ValueError("Only 'roto'/'rotosolve' is implemented")
        if loss is not None:
            raise ValueError("This three-point Rotosolve implementation uses a linear-margin loss; MSE and cross-entropy need a different update.")
        if algorithm.lower() in ("roto", "rotosolve") and optimizer is not None:
            raise ValueError("No separate optimizer is needed for Rotosolve.")
        self.algorithm = algorithm
        self.dev = dev
        self.wires = tuple(dev.wires)
        self.readout_wire = self.wires[:-1] if readout_wire is None else readout_wire
        if self.readout_wire not in self.wires:
            raise ValueError("readout_wire must be among the device wires")
        # A line topology by default; pass hardware edges to override it.
        self.cz_edges = (
            tuple(zip(self.wires[:-1], self.wires[1:]))
            if cz_edges is None
            else tuple(tuple(edge) for edge in cz_edges)
        )
        for edge in self.cz_edges:
            if len(edge) != 2 or edge[0] == edge[1] or any(w not in self.wires for w in edge):
                raise ValueError(f"Invalid CZ edge: {edge}")
        self.new_gate_scale = float(new_gate_scale)
        self._key = jax.random.PRNGKey(seed)
        self.param_steps = param_steps
        self.num_layers = num_layers
        self.gates = []  # Each item: ('RX'|'RZ'|'CZ', tuple(wires))
        self.theta = jnp.zeros((0,), dtype = jnp.float64)
        self.history = []
        self.circuit = self.build_circuit()

    @staticmethod
    def _gate_name(gate):
        name = gate.upper() of isinstance(gate, str) else gate.__name__.upper()
        if name not in  ("RX","RZ","CZ"):
            raise ValueError(f"Non-native or unsupported operation: {name}")
        return name

    def build_circuit(self):
        architecture = tuple(self.gates)
        wires = self.wires
        readout_wire = self.readout_wire
        dev = self.dev

        @qml.qnode(dev, interface = "jax", diff_method = None)
        def circuit(theta, x):
            for feature_index,wire in enumerate(wires):
                qml.RX(x[feature_index], wires = wire)

            parameter_index = 0
            for gate_name, gate_wires in architecture:
                if gate_name == "RX":
                    qml.RX(theta[parameter_index], wires=gate_wires[0])
                    parameter_index += 1
                elif gate_name == "RZ":
                    qml.RZ(theta[parameter_index], wires=gate_wires[0])
                    parameter_index += 1
                elif gate_name == "CZ":
                    qml.CZ(wires=gate_wires)
            return qml.expval(qml.PauliZ(readout_wire))

        return circuit

    def append_gate(self, gate, target, angle = 0.0):
        gate = self._gate_name(gate)
        if gate == "CZ":
            target = tuple(target)

            if (target not in self.cz_edges and target [::-1] not in self.cz_edges):
                raise ValueError("Invalid CZ connection")
        else:
            if target not in self.wires:
                raise ValueError("Invalid target wire")

            self.theta = jnp.concatenate([self.theta, jnp.array([angle])])
        
        self.gates.append((gate, target))

        self.circuit = self.build_circuit()

















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
