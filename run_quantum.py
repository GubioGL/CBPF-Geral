"""
Trains the fixed-gate Qiskit HPINN on the 1D viscous Burgers equation and
compares it against the finite-difference reference. Requires qiskit,
qiskit-machine-learning, and qiskit-algorithms (see README.md).

Run with:
    python run_quantum.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import math

import numpy as np

from functions import (
    plot_loss,
    plot_solution,
    predict,
    reference_at_time,
    relative_l2_error,
    sample_collocation_points,
    set_seed,
    solve_burgers_fd,
    init_weights,
    train,
)
from models import QuantumPINN, FIXED_LAYERS

# ---- physical parameters ----------------------------------------------
L, Tfin, nu = 1.0, 0.1, 0.01
HX, HT = L / 64.0, 2.5e-3

# ---- quantum architecture -----------------------------------------------
N_QUBITS = 5
PRE_NEURONS, PRE_DEPTH = 10, 1
POST_NEURONS, POST_DEPTH = 10, 1
ANGLE_SCALE = math.pi
FIXED_LAYER = "cz_ring"          # one of FIXED_LAYERS, see models.py
FIXED_LAYER_REPEATS = 1
INPUT_ROTATION_AXIS = "rx"

# ---- training -------------------------------------------------------------
N_COLLOCATION = 64
EPOCHS, LR = 400, 2.0e-4
SEED = 42

if __name__ == "__main__":
    print("Available fixed quantum layers:", FIXED_LAYERS)
    print("Selected layer:", FIXED_LAYER)

    set_seed(SEED)

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

    model = QuantumPINN(
        n_qubits=N_QUBITS, pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
        post_neurons=POST_NEURONS, post_depth=POST_DEPTH, angle_scale=ANGLE_SCALE,
        fixed_layer=FIXED_LAYER, fixed_layer_repeats=FIXED_LAYER_REPEATS,
        input_rotation_axis=INPUT_ROTATION_AXIS, seed=SEED, L=L, Tfin=Tfin,
    )
    init_weights(model, seed=SEED)
    n_quantum_params = sum(p.numel() for p in model.quantum.parameters() if p.requires_grad)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters inside the quantum circuit (should be 0):", n_quantum_params)

    model.show_circuit().savefig("quantum_circuit.png", dpi=150)
    print("Saved circuit diagram to quantum_circuit.png")

    x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)
    model, hist = train(
        model, x_f, t_f,
        loss_kwargs=dict(L=L, Tfin=Tfin, nu=nu, hx=HX, ht=HT),
        epochs=EPOCHS, lr=LR, print_every=50,
    )

    plot_loss({"Quantum HPINN": hist["loss"]}, title="Quantum HPINN training loss",
               save_path="quantum_loss.png")

    u_pred = predict(model, x_plot, Tfin)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    plot_solution(
        x_plot, {"Quantum HPINN": u_pred}, ref=u_ref,
        title=f"Quantum HPINN vs FDM reference at T={Tfin}",
        save_path="quantum_solution.png",
    )

    print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
