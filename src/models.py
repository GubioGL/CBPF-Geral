import json
import math
import os

import numpy as np
import torch as tc
import torch.nn as nn

from .functions import initial_condition
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class PINNBase(nn.Module):
    """
    Mixin adding checkpoint save/load to PINN models. Subclasses must set
    `self.config` in `__init__` to the exact kwargs needed to reconstruct
    the model (i.e. `cls(**config)` must work).
    """

    def save(self, path):
        """Saves weights + config to `path` (state_dict + reconstruction
        kwargs), plus a human-readable `<path>_config.json` alongside it so
        the run configuration can be inspected without loading torch."""
        path = str(path)
        tc.save(
            {
                "class_name": type(self).__name__,
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

        config_path = os.path.splitext(path)[0] + "_config.json"
        with open(config_path, "w") as f:
            json.dump({"class_name": type(self).__name__, "config": self.config}, f, indent=2)

    @classmethod
    def load(cls, path, map_location="cpu"):
        """Reconstructs the model from a checkpoint saved with `save`, ready
        for further training or inference."""
        checkpoint = tc.load(path, map_location=map_location, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


class ClassicalPINN(PINNBase):
    """Fully classical MLP ansatz for the 1D viscous Burgers equation."""

    def __init__(self, hidden_neurons=20, hidden_depth=2, L=1.0, Tfin=0.1):
        super().__init__()
        self.config = dict(hidden_neurons=hidden_neurons, hidden_depth=hidden_depth, L=L, Tfin=Tfin)
        self.L, self.Tfin = float(L), float(Tfin)

        layers = [nn.Linear(3, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_depth - 1):
            layers += [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
        layers += [nn.Linear(hidden_neurons, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        input, x = self.Input_angular(xt)
        output = self.net(input)
        return self.Hard_constrain(x, input[:, 2:3], output)

    def Input_angular(self, xt):
        """Maps (x, t) to periodic (sin, cos, tau) features so the network sees x through its L-periodic image."""
        x, tt = xt[:, 0:1], xt[:, 1:2]
        input = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), tt / self.Tfin],
            dim=1,
        )
        return input, x

    def Hard_constrain(self, x, tau, output_nn):
        """Hard-imposes the initial condition at tau=0 by construction."""
        return initial_condition(x, L=self.L) + tau * output_nn


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
    return [
        SparsePauliOp.from_list([("I" * (n_qubits - q - 1) + "Z" + "I" * q, 1.0)])
        for q in range(n_qubits)
    ]


class QuantumPINN(PINNBase):
    """
    Hybrid HPINN: classical pre-net -> bounded rotation angles -> fixed
    (parameter-free) quantum gate layer -> classical post-net. Only the
    classical layers are trainable; the quantum circuit has zero weights.
    """

    def __init__(self, n_qubits=5, pre_neurons=10, pre_depth=1, post_neurons=10,
                 post_depth=1, angle_scale=math.pi, fixed_layer="linear_cnot",
                 fixed_layer_repeats=1, input_rotation_axis="rx", seed=42,
                 L=1.0, Tfin=0.1):
        super().__init__()

        self.config = dict(
            n_qubits=n_qubits, pre_neurons=pre_neurons, pre_depth=pre_depth,
            post_neurons=post_neurons, post_depth=post_depth, angle_scale=angle_scale,
            fixed_layer=fixed_layer, fixed_layer_repeats=fixed_layer_repeats,
            input_rotation_axis=input_rotation_axis, seed=seed, L=L, Tfin=Tfin,
        )
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
            # FIX: EstimatorQNN has its own default_precision (0.015625) that
            # overrides the estimator's default_precision=0.0 above if left
            # unset, silently making the "exact" StatevectorEstimator run
            # with shot-noise-like approximation instead. Pass it explicitly
            # to get the exact simulation the surrounding code intends.
            default_precision=0.0,
        )
        # No initial_weights beyond the empty array: the circuit is fixed, zero trainable weights.
        self.quantum = TorchConnector(qnn, initial_weights=np.zeros(qnn.num_weights, dtype=np.float32))

        post_layers = [nn.Linear(n_qubits, post_neurons), nn.Tanh()]
        for _ in range(post_depth - 1):
            post_layers += [nn.Linear(post_neurons, post_neurons), nn.Tanh()]
        post_layers += [nn.Linear(post_neurons, 1)]
        self.post = nn.Sequential(*post_layers)

    def forward(self, xt):
        input, x = self.Input_angular(xt)
        angles = self.angle_scale * tc.tanh(self.pre(input))
        # TorchConnector/EstimatorQNN returns its own dtype regardless of the input; cast back for self.post.
        q = self.quantum(angles).to(dtype=self.post[0].weight.dtype)
        output = self.post(q)
        return self.Hard_constrain(x, input[:, 2:3], output)

    def Input_angular(self, xt):
        """Maps (x, t) to periodic (sin, cos, tau) features so the network sees x through its L-periodic image."""
        x, tt = xt[:, 0:1], xt[:, 1:2]
        input = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), tt / self.Tfin],
            dim=1,
        )
        return input, x

    def Hard_constrain(self, x, tau, output_nn):
        """Hard-imposes the initial condition at tau=0 by construction."""
        return initial_condition(x, L=self.L) + tau * output_nn

    def show_circuit(self):
        return self.circuit.draw("mpl")


# ----------------------------------------------------------------------
# Generic checkpoint loading (use when the caller doesn't already know
# which PINN subclass produced the checkpoint).
# ----------------------------------------------------------------------
_MODEL_REGISTRY = {"ClassicalPINN": ClassicalPINN, "QuantumPINN": QuantumPINN}


def load_model(path, map_location="cpu"):
    """Loads a checkpoint saved with `PINNBase.save`, picking the right
    model class automatically. Returns a model ready for retraining or
    inference."""
    checkpoint = tc.load(path, map_location=map_location, weights_only=False)
    cls = _MODEL_REGISTRY[checkpoint["class_name"]]
    model = cls(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model
    
    
