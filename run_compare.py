"""
Trains both the classical PINN and the quantum HPINN and compares them
side by side against the finite-difference reference: solution overlay,
loss curves, and final relative-L2 error table.

Run with:
    python run_compare.py
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
from models import ClassicalPINN, QuantumPINN

# ---- physical parameters ----------------------------------------------
L, Tfin, nu = 1.0, 0.1, 0.01
HX, HT = L / 64.0, 2.5e-3
N_COLLOCATION = 64
EPOCHS, LR = 400, 2.0e-4
SEED = 42

# ---- architectures ---------------------------------------------------
HIDDEN_NEURONS, HIDDEN_DEPTH = 20, 2
N_QUBITS = 5
PRE_NEURONS, PRE_DEPTH = 10, 1
POST_NEURONS, POST_DEPTH = 10, 1
FIXED_LAYER = "cz_ring"


def run(model, label):
    init_weights(model, seed=SEED)
    x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)
    print(f"\n[{label}] training...")
    model, hist = train(
        model, x_f, t_f,
        loss_kwargs=dict(L=L, Tfin=Tfin, nu=nu, hx=HX, ht=HT),
        epochs=EPOCHS, lr=LR, print_every=100,
    )
    return model, hist


if __name__ == "__main__":
    set_seed(SEED)
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    classical = ClassicalPINN(hidden_neurons=HIDDEN_NEURONS, hidden_depth=HIDDEN_DEPTH, L=L, Tfin=Tfin)
    classical, hist_classical = run(classical, "Classical PINN")

    quantum = QuantumPINN(
        n_qubits=N_QUBITS, pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
        post_neurons=POST_NEURONS, post_depth=POST_DEPTH, angle_scale=math.pi,
        fixed_layer=FIXED_LAYER, seed=SEED, L=L, Tfin=Tfin,
    )
    quantum, hist_quantum = run(quantum, "Quantum HPINN")

    u_classical = predict(classical, x_plot, Tfin)
    u_quantum = predict(quantum, x_plot, Tfin)

    plot_solution(
        x_plot,
        {"Classical PINN": u_classical, "Quantum HPINN": u_quantum},
        ref=u_ref,
        title=f"Classical PINN vs Quantum HPINN at T={Tfin}",
        save_path="compare_solution.png",
    )
    plot_loss(
        {"Classical PINN": hist_classical["loss"], "Quantum HPINN": hist_quantum["loss"]},
        title="Classical PINN vs Quantum HPINN training loss",
        save_path="compare_loss.png",
    )

    print("\nFinal-time relative L2 errors against the FDM reference:")
    print(f"  Classical PINN  | {relative_l2_error(u_classical, u_ref):.6e}")
    print(f"  Quantum HPINN   | {relative_l2_error(u_quantum, u_ref):.6e}")
