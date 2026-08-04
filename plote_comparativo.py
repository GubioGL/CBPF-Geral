"""Plots MSE and MAE of the trained quantum PINN models against the number of
qubits, comparing each model's prediction at t=Tfin to the FDM reference
solution. One figure is produced per T_* directory in data/quantum."""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch as tc

from src.config import NU
from src.functions import predict, reference_at_time, solve_burgers_fd
from src.models import ClassicalPINN, ClassicalPINN2
from src.models_pennylane import QuantumPINNPennyLane

DATA_DIR = "data/quantum_old"
CLASSICAL_DIR = "data/classical_old"
COLOR_MSE = "#2a78d6"
COLOR_MAE = "#1baf7a"
COLOR_CLASSICAL = "#e34948"

# Mid-layer (bottleneck) widths of the classical Cmodel_NEURO110{n}_* family,
# i.e. ClassicalPINN2 checkpoints with intra_depth=1, intra_neurons=10 -- one
# reference line per value, colored in the palette's fixed categorical order.
MID_LAYER_NEURONS = [1, 5, 10]
COLOR_MID_LAYER = {1: "#eb6834", 5: "#e87ba4", 10: "#4a3aa7"}

MODEL_RE = re.compile(r"model_NQUBITS(\d+)_p_\d+\.pt$")


def find_models(t_dir):
    """Returns [(n_qubits, path), ...] sorted by n_qubits for one T_* directory."""
    models = []
    for path in glob.glob(os.path.join(t_dir, "model_NQUBITS*_p_*.pt")):
        m = MODEL_RE.search(path)
        if m:
            models.append((int(m.group(1)), path))
    return sorted(models)


def load_legacy_quantum(path):
    """Loads a QuantumPINNPennyLane checkpoint from `data/quantum_old`. Those
    checkpoints predate the `estrutura` config field and were all trained
    with the pre -> quantum -> post architecture, i.e. `estrutura=1` --
    `QuantumPINNPennyLane.load` would otherwise fall back to the current
    default `estrutura=0`, which requires n_qubits=3 and breaks on every
    other qubit count."""
    checkpoint = tc.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config.setdefault("estrutura", 1)
    model = QuantumPINNPennyLane(**config)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def compute_errors(t_dir):
    """Loads every model in `t_dir`, evaluates each against the FDM reference
    at t=Tfin, and averages the MSE/MAE across all available samples (p_*)
    for each number of qubits -- since the number of samples generally
    differs per qubit count. Returns (n_qubits, mean_mse, mean_mae,
    n_samples, Tfin), one point per qubit count."""
    models = find_models(t_dir)
    if not models:
        return [], [], [], [], None

    loaded = [(n, load_legacy_quantum(path)) for n, path in models]
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


def compute_mid_layer_mean_errors(t_dir, mid_neurons=MID_LAYER_NEURONS):
    """Loads the classical Cmodel_NEURO110{n}_* checkpoints in `t_dir` -- i.e.
    ClassicalPINN2 models with intra_depth=1, intra_neurons=10 and bottleneck
    width `intra_neuron` in `mid_neurons` -- and averages MSE/MAE across
    run_id samples for each n. The bottleneck width is read from each model's
    `_config.json` rather than parsed from the filename. Returns {n: (mean_mse,
    mean_mae)}, one entry per n found."""
    paths_by_n = {n: [] for n in mid_neurons}
    for config_path in glob.glob(os.path.join(t_dir, "Cmodel_NEURO*_config.json")):
        pt_path = config_path[: -len("_config.json")] + ".pt"
        if not os.path.exists(pt_path):
            continue
        with open(config_path) as f:
            model_cfg = json.load(f)["config"]
        n = model_cfg.get("intra_neuron")
        if n in paths_by_n and model_cfg.get("intra_neurons") == 10 and model_cfg.get("intra_depth") == 1:
            paths_by_n[n].append(pt_path)

    paths_by_n = {n: paths for n, paths in paths_by_n.items() if paths}
    if not paths_by_n:
        return {}

    first_model = ClassicalPINN2.load(next(iter(paths_by_n.values()))[0])
    L, Tfin = first_model.L, first_model.Tfin
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors = {}
    for n, paths in paths_by_n.items():
        mse_vals, mae_vals = [], []
        for pt_path in paths:
            model = ClassicalPINN2.load(pt_path)
            u_pred = predict(model, x_plot, Tfin)
            mse_vals.append(float(np.mean((u_pred - u_ref) ** 2)))
            mae_vals.append(float(np.mean(np.abs(u_pred - u_ref))))
        errors[n] = (float(np.mean(mse_vals)), float(np.mean(mae_vals)))
    return errors


def plot_errors(n_qubits, mse, Tfin, classical_mse, mid_layer_mse, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(n_qubits, mse, "o-", color=COLOR_MSE, linewidth=2.0, markersize=6, label="MSE")
    if classical_mse is not None:
        ax.axhline(classical_mse, color=COLOR_CLASSICAL, linestyle="--", linewidth=2.0,
                    label="Classical mean MSE (20-20)")
    for n in sorted(mid_layer_mse):
        ax.axhline(mid_layer_mse[n], color=COLOR_MID_LAYER[n], linestyle=":", linewidth=2.0,
                    label=f"Classical mean MSE (10-{n}-10)")
    ax.set_xlabel("Number of qubits", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(n_qubits)
    ax.set_title(f"Quantum PINN mean MSE vs number of qubits (T = {Tfin})", fontsize=13)
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
        mid_layer_errors = compute_mid_layer_mean_errors(classical_t_dir)
        mid_layer_mse = {n: e_mse for n, (e_mse, _e_mae) in mid_layer_errors.items()}

        save_path = os.path.join(f"Error_vs_Qubits_T_{t_label}.png")
        plot_errors(n_qubits, mse, Tfin, classical_mse, mid_layer_mse, save_path)

        print(f"T = {Tfin}")
        for n, e_mse, e_mae, n_s in zip(n_qubits, mse, mae, n_samples):
            print(f"  n_qubits={n:2d}  MSE={e_mse:.6e}  MAE={e_mae:.6e}  (mean over {n_s} samples)")
        if classical_mse is not None:
            print(f"  classical  MSE={classical_mse:.6e}  MAE={classical_mae:.6e}")
        for n in sorted(mid_layer_errors):
            e_mse, e_mae = mid_layer_errors[n]
            print(f"  mid layer={n:2d}  MSE={e_mse:.6e}  MAE={e_mae:.6e}")
        print(f"  -> saved {save_path}")
