"""Plots MSE and MAE of the trained quantum PINN models against the number of
qubits, comparing each model's prediction at t=Tfin to the FDM reference
solution. One figure is produced per T_* directory in data/quantum."""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from src.config import NU
from src.functions import predict, reference_at_time, solve_burgers_fd
from src.models import ClassicalPINN
from src.models_pennylane import QuantumPINNPennyLane

DATA_DIR = "data/quantum"
CLASSICAL_DIR = "data/classical"
COLOR_MSE = "#2a78d6"
COLOR_MAE = "#1baf7a"
COLOR_CLASSICAL = "#e34948"

MODEL_RE = re.compile(r"model_NQUBITS(\d+)_p_\d+\.pt$")


def find_models(t_dir):
    """Returns [(n_qubits, path), ...] sorted by n_qubits for one T_* directory."""
    models = []
    for path in glob.glob(os.path.join(t_dir, "model_NQUBITS*_p_*.pt")):
        m = MODEL_RE.search(path)
        if m:
            models.append((int(m.group(1)), path))
    return sorted(models)


def compute_errors(t_dir):
    """Loads every model in `t_dir`, evaluates each against the FDM reference
    at t=Tfin, and averages the MSE/MAE across all available samples (p_*)
    for each number of qubits -- since the number of samples generally
    differs per qubit count. Returns (n_qubits, mean_mse, mean_mae,
    n_samples, Tfin), one point per qubit count."""
    models = find_models(t_dir)
    if not models:
        return [], [], [], [], None

    loaded = [(n, QuantumPINNPennyLane.load(path)) for n, path in models]
    L, Tfin = loaded[0][1].L, loaded[0][1].Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_n = {}
    for n, model in loaded:
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_n.setdefault(n, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    n_qubits = sorted(errors_by_n)
    mse = [float(np.mean(errors_by_n[n]["mse"])) for n in n_qubits]
    mae = [float(np.mean(errors_by_n[n]["mae"])) for n in n_qubits]
    n_samples = [len(errors_by_n[n]["mse"]) for n in n_qubits]
    return n_qubits, mse, mae, n_samples, Tfin


def compute_classical_mean_errors(t_dir):
    """Loads every classical model (amostragem_*.pt) in `t_dir`, evaluates each
    against the FDM reference at t=Tfin, and returns the mean MSE and mean MAE
    across all of them (None, None if no classical models are found)."""
    paths = sorted(glob.glob(os.path.join(t_dir, "amostragem_*.pt")))
    if not paths:
        return None, None

    loaded = [ClassicalPINN.load(path) for path in paths]
    L, Tfin = loaded[0].L, loaded[0].Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    mse_vals, mae_vals = [], []
    for model in loaded:
        u_pred = predict(model, x_plot, Tfin)
        mse_vals.append(float(np.mean((u_pred - u_ref) ** 2)))
        mae_vals.append(float(np.mean(np.abs(u_pred - u_ref))))
    return float(np.mean(mse_vals)), float(np.mean(mae_vals))


def plot_errors(n_qubits, mse, mae, Tfin, classical_mse, classical_mae, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(n_qubits, mse, "o-", color=COLOR_MSE, linewidth=2.0, markersize=6, label="MSE")
    ax.plot(n_qubits, mae, "o-", color=COLOR_MAE, linewidth=2.0, markersize=6, label="MAE")
    if classical_mse is not None:
        ax.axhline(classical_mse, color=COLOR_CLASSICAL, linestyle="--", linewidth=2.0,
                    label="Classical mean MSE")
    if classical_mae is not None:
        ax.axhline(classical_mae, color=COLOR_CLASSICAL, linestyle=":", linewidth=2.0,
                    label="Classical mean MAE")
    ax.set_xlabel("Number of qubits", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(n_qubits)
    ax.set_title(f"Quantum PINN mean error vs number of qubits (T = {Tfin})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_dir in sorted(glob.glob(os.path.join(DATA_DIR, "T_*"))):
        n_qubits, mse, mae, n_samples, Tfin = compute_errors(t_dir)
        if not n_qubits:
            continue

        t_label = os.path.basename(t_dir).replace("T_", "")

        classical_t_dir = os.path.join(CLASSICAL_DIR, os.path.basename(t_dir))
        classical_mse, classical_mae = compute_classical_mean_errors(classical_t_dir)

        save_path = os.path.join("data/quantum", f"Error_vs_Qubits_T_{t_label}.png")
        plot_errors(n_qubits, mse, mae, Tfin, classical_mse, classical_mae, save_path)

        print(f"T = {Tfin}")
        for n, e_mse, e_mae, n_s in zip(n_qubits, mse, mae, n_samples):
            print(f"  n_qubits={n:2d}  MSE={e_mse:.6e}  MAE={e_mae:.6e}  (mean over {n_s} samples)")
        if classical_mse is not None:
            print(f"  classical  MSE={classical_mse:.6e}  MAE={classical_mae:.6e}")
        print(f"  -> saved {save_path}")
