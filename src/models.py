"""
Model definitions: the classical hard-IC PINN and the fixed-gate Qiskit
HPINN. Both share the same hard-constraint ansatz

    u_theta(x,t) = u0(x) + tau * N_theta(sin(2*pi*x/L), cos(2*pi*x/L), tau),   tau = t/Tfin

so u(x,0) = u0(x) exactly and periodicity in x is exact by construction.
The difference is what N_theta is: a plain MLP for the classical case, or a
classical pre-net -> fixed quantum circuit -> classical post-net for the
hybrid case. Qiskit imports are local to QuantumPINN so this module (and
the classical case) can be used without qiskit installed.
"""
import math

import numpy as np
import torch as tc
import torch.nn as nn

from functions import initial_condition


class ClassicalPINN(nn.Module):
    """Fully classical MLP ansatz for the 1D viscous Burgers equation."""

    def __init__(self, hidden_neurons=20, hidden_depth=2, L=1.0, Tfin=0.1):
        super().__init__()
        self.L, self.Tfin = float(L), float(Tfin)

        layers = [nn.Linear(3, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_depth - 1):
            layers += [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
        layers += [nn.Linear(hidden_neurons, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        xt = xt.to(dtype=next(self.parameters()).dtype)
        x, tt = xt[:, 0:1], xt[:, 1:2]
        tau = tt / self.Tfin

        features = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), tau],
            dim=1,
        )
        raw = self.net(features)
        return initial_condition(x, L=self.L) + tau * raw


# ----------------------------------------------------------------------
# Quantum-specific circuit builders (only used by QuantumPINN, kept local
# to it rather than in functions.py since nothing else needs them).
# ----------------------------------------------------------------------
FIXED_LAYERS = (
    "linear_cnot", "reverse_linear_cnot", "ring_cnot", "brickwork_cnot",
    "symmetric_cnot_ladder", "cz_ring", "ghz_fanout", "swap_network",
)


def _apply_fixed_gate_layer(qc, layer_kind):
    n = qc.num_qubits
    if n < 2:
        return

    if layer_kind == "linear_cnot":
        for c in range(n - 1):
            qc.cx(c, c + 1)
    elif layer_kind == "reverse_linear_cnot":
        for c in range(n - 1, 0, -1):
            qc.cx(c, c - 1)
    elif layer_kind == "ring_cnot":
        for c in range(n - 1):
            qc.cx(c, c + 1)
        if n > 2:
            qc.cx(n - 1, 0)
    elif layer_kind == "brickwork_cnot":
        for c in range(0, n - 1, 2):
            qc.cx(c, c + 1)
        for c in range(1, n - 1, 2):
            qc.cx(c, c + 1)
    elif layer_kind == "symmetric_cnot_ladder":
        for c in range(n - 1):
            qc.cx(c, c + 1)
        for c in range(n - 1, 0, -1):
            qc.cx(c, c - 1)
    elif layer_kind == "cz_ring":
        for q in range(n - 1):
            qc.cz(q, q + 1)
        if n > 2:
            qc.cz(n - 1, 0)
    elif layer_kind == "ghz_fanout":
        for target in range(1, n):
            qc.cx(0, target)
    elif layer_kind == "swap_network":
        for q in range(n - 1):
            qc.swap(q, q + 1)
    else:
        raise ValueError(f"Unknown fixed layer '{layer_kind}'. Options: {FIXED_LAYERS}.")


def _build_fixed_circuit(n_qubits, layer_kind, repeats, axis):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    input_params = ParameterVector("inputs", n_qubits)
    qc = QuantumCircuit(n_qubits, name=f"fixed_{layer_kind}")

    rot = {"rx": qc.rx, "ry": qc.ry, "rz": qc.rz}.get(axis.lower())
    if rot is None:
        raise ValueError("axis must be one of: 'rx', 'ry', 'rz'.")
    for q, p in enumerate(input_params):
        rot(p, q)

    for _ in range(repeats):
        _apply_fixed_gate_layer(qc, layer_kind)

    return qc, list(input_params)


def _z_observables(n_qubits):
    from qiskit.quantum_info import SparsePauliOp

    return [
        SparsePauliOp.from_list([("I" * (n_qubits - q - 1) + "Z" + "I" * q, 1.0)])
        for q in range(n_qubits)
    ]


class QuantumPINN(nn.Module):
    """
    Hybrid HPINN: classical pre-net -> bounded rotation angles -> fixed
    (parameter-free) quantum gate layer -> classical post-net. Only the
    classical layers are trainable; the quantum circuit has zero weights.
    """

    def __init__(self, n_qubits=5, pre_neurons=10, pre_depth=1, post_neurons=10,
                 post_depth=1, angle_scale=math.pi, fixed_layer="cz_ring",
                 fixed_layer_repeats=1, input_rotation_axis="rx", seed=42,
                 L=1.0, Tfin=0.1):
        super().__init__()
        from qiskit.primitives import StatevectorEstimator
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

        self.L, self.Tfin = float(L), float(Tfin)
        self.angle_scale = float(angle_scale)

        pre_layers = [nn.Linear(3, pre_neurons), nn.Tanh()]
        for _ in range(pre_depth - 1):
            pre_layers += [nn.Linear(pre_neurons, pre_neurons), nn.Tanh()]
        pre_layers += [nn.Linear(pre_neurons, n_qubits)]
        self.pre = nn.Sequential(*pre_layers)

        self.circuit, input_params = _build_fixed_circuit(
            n_qubits, fixed_layer, fixed_layer_repeats, input_rotation_axis
        )
        observables = _z_observables(n_qubits)

        estimator = StatevectorEstimator(default_precision=0.0, seed=seed)
        gradient = ParamShiftEstimatorGradient(estimator=estimator, precision=0.0)
        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=estimator,
            observables=observables,
            input_params=input_params,
            weight_params=[],
            gradient=gradient,
            input_gradients=True,
        )
        # No initial_weights beyond the empty array: the circuit is fixed, zero trainable weights.
        self.quantum = TorchConnector(qnn, initial_weights=np.zeros(qnn.num_weights, dtype=np.float32))

        post_layers = [nn.Linear(n_qubits, post_neurons), nn.Tanh()]
        for _ in range(post_depth - 1):
            post_layers += [nn.Linear(post_neurons, post_neurons), nn.Tanh()]
        post_layers += [nn.Linear(post_neurons, 1)]
        self.post = nn.Sequential(*post_layers)

    def forward(self, xt):
        xt = xt.to(dtype=self.pre[0].weight.dtype)
        x, tt = xt[:, 0:1], xt[:, 1:2]
        tau = tt / self.Tfin

        features = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), tau],
            dim=1,
        )
        angles = self.angle_scale * tc.tanh(self.pre(features))
        q = self.quantum(angles).to(dtype=self.post[0].weight.dtype)
        raw = self.post(q)
        return initial_condition(x, L=self.L) + tau * raw

    def show_circuit(self):
        return self.circuit.draw("mpl")
