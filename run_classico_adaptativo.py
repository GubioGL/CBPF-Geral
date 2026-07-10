import matplotlib.pyplot as plt
import numpy as np

from src.functions import (
    plot_loss,
    plot_solution,
    predict,
    reference_at_time,
    relative_l2_error,
    sample_collocation_points,
    set_seed,
    solve_burgers_fd,
    train_adaptive,
)
from src.models import ClassicalPINN
from src.config import L, TFIN as Tfin, NU as nu, HX, HT, N_COLLOCATION, SEED, EPOCHS, LR, HIDDEN_NEURONS, HIDDEN_DEPTH


def plot_step_sizes(hist, title, save_path):
    """hx_mean/ht_mean evolution -- shows the adaptive stencil shrinking/growing over training."""
    plt.figure(figsize=(9, 4.5))
    plt.plot(hist["hx_mean"], linewidth=2.0, label=r"mean $h_x$ (spatial)")
    plt.plot(hist["ht_mean"], linewidth=2.0, label=r"mean $h_t$ (temporal)")
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Step size", fontsize=14)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":

    for N_COLLOCATION in range(20, 101, 20):
        set_seed(SEED)

        # Reference solution, used only for comparison/plots, never for training.
        x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
        x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

        model = ClassicalPINN(hidden_neurons=HIDDEN_NEURONS, hidden_depth=HIDDEN_DEPTH, L=L, Tfin=Tfin)

        x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)
        model, hist = train_adaptive(
            model, x_f, t_f,
            loss_kwargs=dict(L=L, Tfin=Tfin, nu=nu, hx=HX, ht=HT),
            epochs=EPOCHS, lr=LR, print_every=50,
        )

        model.save(f"data/classical_adaptive/classical_pinn_adaptive_N{N_COLLOCATION}.pt")

        plot_loss({"Classical PINN (adaptive mesh)": hist["loss"]}, title="Classical PINN training loss (adaptive mesh)",
                  save_path=f"data/classical_adaptive/classical_loss_adaptive_N{N_COLLOCATION}.png")

        plot_step_sizes(
            hist, title="Adaptive stencil widths (Classical PINN)",
            save_path=f"data/classical_adaptive/classical_stepsize_adaptive_N{N_COLLOCATION}.png",
        )

        u_pred = predict(model, x_plot, Tfin)
        u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
        plot_solution(
            x_plot, {"Classical PINN (adaptive mesh)": u_pred}, ref=u_ref,
            title=f"Classical PINN (adaptive mesh) vs FDM reference at T={Tfin}",
            save_path=f"data/classical_adaptive/classical_solution_adaptive_N{N_COLLOCATION}.png",
            show_cond=False
        )

        print(f"N_COLLOCATION={N_COLLOCATION} | Relative L2 error at T={Tfin}: "
              f"{relative_l2_error(u_pred, u_ref):.6e}")
