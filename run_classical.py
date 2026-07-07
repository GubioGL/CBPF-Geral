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
    train,
)
from models import ClassicalPINN
from config import L, TFIN as Tfin, NU as nu, HX, HT, N_COLLOCATION, SEED, EPOCHS, LR, HIDDEN_NEURONS, HIDDEN_DEPTH


if __name__ == "__main__":
    
    for N_COLLOCATION in range(20, 101,20):
        set_seed(SEED)

        # Reference solution, used only for comparison/plots, never for training.
        x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
        x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

        model = ClassicalPINN(hidden_neurons=HIDDEN_NEURONS, hidden_depth=HIDDEN_DEPTH, L=L, Tfin=Tfin)
        #print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

        x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)
        model, hist = train(
            model, x_f, t_f,
            loss_kwargs=dict(L=L, Tfin=Tfin, nu=nu, hx=HX, ht=HT),
            epochs=EPOCHS, lr=LR, print_every=50,
        )

        model.save(f"data/classical/classical_pinn_N{N_COLLOCATION}.pt")
       # print(f"Saved model checkpoint to data/classical/classical_pinn_N{N_COLLOCATION}.pt (config: data/classical/classical_pinn_config.json)")

        plot_loss({"Classical PINN": hist["loss"]}, title="Classical PINN training loss",
                save_path=f"data/classical/classical_loss_N{N_COLLOCATION}.png"
                )

        u_pred = predict(model, x_plot, Tfin)
        u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
        plot_solution(
            x_plot, {"Classical PINN": u_pred}, ref=u_ref,
            title=f"Classical PINN vs FDM reference at T={Tfin}",
            save_path=f"data/classical/classical_solution_N{N_COLLOCATION}.png",
            show_cond=False
        )

        # print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
