"""Plots mean +/- std MSE/MAE of the trained classical PINN models against
INTRA_NEURO, with the standard deviation across the run_id samples shown as
error bars. One figure is produced for T_0.1 and one for T_1, using the
legacy checkpoints in data/classical_old."""
import os

import matplotlib.pyplot as plt
import numpy as np

from plote_classico import find_classical_configs
from src.config import NU
from src.functions import predict, reference_at_time, solve_burgers_fd
from src.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical_old"
T_LABELS = ["0.1", "1"]
COLOR_MSE = "#2a78d6"
COLOR_MAE = "#1baf7a"


def compute_error_stats(t_dir):
    """Loads every classical model in `t_dir`, evaluates each against the FDM
    reference at t=Tfin, and returns per-INTRA_NEURO mean/std MSE and MAE
    across the run_id samples. Returns (intra_neurons, mse_mean, mse_std,
    mae_mean, mae_std, n_samples, Tfin), one point per INTRA_NEURO."""
    entries = find_classical_configs(t_dir)
    if not entries:
        return [], [], [], [], [], [], None

    first_model = ClassicalPINN2.load(entries[0][2])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_neuron = {}
    for intra_neuron, _run_id, pt_path in entries:
        model = ClassicalPINN2.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_neuron.setdefault(intra_neuron, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    intra_neurons = sorted(errors_by_neuron)
    mse_mean = [float(np.mean(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_std = [float(np.std(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mae_mean = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_std = [float(np.std(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
    return intra_neurons, mse_mean, mse_std, mae_mean, mae_std, n_samples, Tfin


def plot_errors_std(intra_neurons, mse_mean, mse_std, mae_mean, mae_std, Tfin, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(intra_neurons, mse_mean, yerr=mse_std, fmt="o-", color=COLOR_MSE,
                linewidth=2.0, markersize=6, capsize=4, label="MSE (mean ± std)")
    ax.errorbar(intra_neurons, mae_mean, yerr=mae_std, fmt="o-", color=COLOR_MAE,
                linewidth=2.0, markersize=6, capsize=4, label="MAE (mean ± std)")
    ax.set_xlabel("INTRA_NEURO", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(intra_neurons)
    ax.set_title(f"Classical PINN error vs INTRA_NEURO (T = {Tfin}, legacy data)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")
        intra_neurons, mse_mean, mse_std, mae_mean, mae_std, n_samples, Tfin = compute_error_stats(t_dir)
        if not intra_neurons:
            print(f"No classical models found in {t_dir}, skipping.")
            continue

        save_path = f"Error_std_vs_IntraNeuro_T_{t_label}_old.png"
        plot_errors_std(intra_neurons, mse_mean, mse_std, mae_mean, mae_std, Tfin, save_path)

        print(f"T = {Tfin}")
        for n, m_mean, m_std, a_mean, a_std, n_s in zip(
            intra_neurons, mse_mean, mse_std, mae_mean, mae_std, n_samples
        ):
            print(f"  INTRA_NEURO={n:2d}  MSE={m_mean:.6e}+/-{m_std:.6e}  MAE={a_mean:.6e}+/-{a_std:.6e}  (n={n_s})")
        print(f"  -> saved {save_path}")
