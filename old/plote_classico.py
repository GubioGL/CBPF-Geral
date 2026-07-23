import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import NU
from src.functions import predict, reference_at_time, solve_burgers_fd
from src.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
COLOR_MSE = "#2a78d6"
COLOR_MAE = "#1baf7a"


def find_classical_configs(t_dir):
    """Returns [(intra_neuron, run_id, pt_path), ...] for every classical
    Cmodel_NEURO* checkpoint in `t_dir`, reading intra_neuron from its
    companion `_config.json` (the run_id/sample index comes from the
    trailing `_<run_id>` in the filename)."""
    entries = []
    for config_path in glob.glob(os.path.join(t_dir, "Cmodel_NEURO*_config.json")):
        pt_path = config_path[: -len("_config.json")] + ".pt"
        if not os.path.exists(pt_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        intra_neuron = cfg["config"]["intra_neuron"]
        run_id = int(os.path.basename(pt_path)[: -len(".pt")].rsplit("_", 1)[1])
        entries.append((intra_neuron, run_id, pt_path))
    return sorted(entries)


def compute_mean_errors(t_dir):
    """Loads every classical model in `t_dir`, evaluates each against the FDM
    reference at t=Tfin, and averages the MSE/MAE across the run_id samples
    (one per amostragem) for each INTRA_NEURO value. Returns (intra_neurons,
    mean_mse, mean_mae, n_samples, Tfin), one point per INTRA_NEURO."""
    entries = find_classical_configs(t_dir)
    if not entries:
        return [], [], [], [], None

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
    mse = [float(np.mean(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mae = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
    return intra_neurons, mse, mae, n_samples, Tfin


def plot_errors(intra_neurons, mse, mae, Tfin, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(intra_neurons, mse, "o-", color=COLOR_MSE, linewidth=2.0, markersize=6, label="MSE")
    ax.plot(intra_neurons, mae, "o-", color=COLOR_MAE, linewidth=2.0, markersize=6, label="MAE")
    ax.set_xlabel("INTRA_NEURO", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(intra_neurons)
    ax.set_title(f"Classical PINN mean error vs INTRA_NEURO (T = {Tfin})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_dir in sorted(glob.glob(os.path.join(CLASSICAL_DIR, "T_*"))):
        intra_neurons, mse, mae, n_samples, Tfin = compute_mean_errors(t_dir)
        if not intra_neurons:
            continue

        t_label = os.path.basename(t_dir).replace("T_", "")

        plot_errors(intra_neurons, mse, mae, Tfin, f"Error_vs_IntraNeuro_T_{t_label}.png")

        print(f"T = {Tfin}")
        for n, e_mse, e_mae, n_s in zip(intra_neurons, mse, mae, n_samples):
            print(f"  INTRA_NEURO={n:2d}  MSE={e_mse:.6e}  MAE={e_mae:.6e}  (mean over {n_s} samples)")
