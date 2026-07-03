"""
Trains the classical hard-IC PINN on the 1D viscous Burgers equation and
compares it against the finite-difference reference.

Run with:
    python run_classical.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

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
from models import ClassicalPINN

# ---- physical parameters ----------------------------------------------
L, Tfin, nu = 1.0, 0.1, 0.01
HX, HT = L / 64.0, 2.5e-3

# ---- architecture and training ------------------------------------------
HIDDEN_NEURONS, HIDDEN_DEPTH = 20, 2
N_COLLOCATION = 64
EPOCHS, LR = 400, 2.0e-4
SEED = 42

if __name__ == "__main__":
    set_seed(SEED)

    # Reference solution, used only for comparison/plots, never for training.
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

    model = ClassicalPINN(hidden_neurons=HIDDEN_NEURONS, hidden_depth=HIDDEN_DEPTH, L=L, Tfin=Tfin)
    init_weights(model, seed=SEED)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)
    model, hist = train(
        model, x_f, t_f,
        loss_kwargs=dict(L=L, Tfin=Tfin, nu=nu, hx=HX, ht=HT),
        epochs=EPOCHS, lr=LR, print_every=50,
    )

    plot_loss({"Classical PINN": hist["loss"]}, title="Classical PINN training loss",
               save_path="classical_loss.png")

    u_pred = predict(model, x_plot, Tfin)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    plot_solution(
        x_plot, {"Classical PINN": u_pred}, ref=u_ref,
        title=f"Classical PINN vs FDM reference at T={Tfin}",
        save_path="classical_solution.png",
    )

    print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
