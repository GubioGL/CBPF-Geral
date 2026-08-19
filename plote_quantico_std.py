"""Plots mean MAE of the trained quantum PINN models against N_QUBITS, one
curve per FIXED_LAYER, with the [min, max] range across the SEED samples
shown as error bars -- the quantum counterpart of
`plote_classico_std copy.py`. One figure is produced per T_* directory in
data/quantum.

Error bars show [min, max] across seeds rather than mean +/- std: MAE is
always positive and, with only a handful of seeds, its std is frequently
larger than its mean, which would push a symmetric mean +/- std bar below
zero -- something the log-scale y-axis below cannot render, so matplotlib
clips that whisker all the way to the bottom of the plot. [min, max] is
built from samples that were actually observed, so it is always positive
and never overruns the axis. mean +/- std is still printed to the console
for reference."""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import NU
from src.functions import asymmetric_yerr, predict, reference_at_time, solve_burgers_fd
from src.models_pennylane import QuantumPINNPennyLane

QUANTUM_DIR = "data/quantum"
INPUT_ROTATION_AXIS = "ry"
COLORS = ["#2a78d6", "#e34948", "#eb6834", "#4a3aa7", "#1baf7a", "#e87ba4"]


def find_quantum_configs(t_dir, input_rotation_axis=INPUT_ROTATION_AXIS, collocation_distribution="random"):
    """Returns [(fixed_layer, n_qubits, seed, pt_path), ...] for every
    Qmodel_<collocation_distribution>_NQUBITS* checkpoint in `t_dir` whose
    input_rotation_axis matches `input_rotation_axis`
    (`collocation_distribution` is "uniform" or "random", matching how
    run_quantum_autograd_pennylane.py names its checkpoints), reading
    fixed_layer/n_qubits/seed from each checkpoint's companion
    `_config.json` rather than parsing the filename (fixed_layer names like
    "cz_ring" and "ring_cnot" contain underscores themselves, which makes
    filename parsing ambiguous)."""
    entries = []
    pattern = f"Qmodel_{collocation_distribution}_NQUBITS*_config.json"
    for config_path in glob.glob(os.path.join(t_dir, pattern)):
        pt_path = config_path[: -len("_config.json")] + ".pt"
        if not os.path.exists(pt_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)["config"]
        if cfg.get("input_rotation_axis") != input_rotation_axis:
            continue
        entries.append((cfg["fixed_layer"], cfg["n_qubits"], cfg["seed"], pt_path))
    return sorted(entries)


def compute_error_stats(t_dir):
    """Loads every quantum model in `t_dir`, evaluates each against the FDM
    reference at t=Tfin, and returns per-(FIXED_LAYER, N_QUBITS) mean/std/
    min/max MAE across the SEED samples. Returns (stats, Tfin), where stats
    is {fixed_layer: (n_qubits, mae_mean, mae_std, mae_min, mae_max,
    n_samples)}, one entry per FIXED_LAYER found. mae_min/mae_max (the
    extremes actually observed across seeds) are what the plot uses for its
    error bars -- unlike mean +/- std, they can never go negative on the
    plot's log-scale y-axis."""
    entries = find_quantum_configs(t_dir)
    if not entries:
        return {}, None

    first_model = QuantumPINNPennyLane.load(entries[0][3])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_layer = {}
    for fixed_layer, n_qubits, _seed, pt_path in entries:
        model = QuantumPINNPennyLane.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        by_n_qubits = errors_by_layer.setdefault(fixed_layer, {})
        by_n_qubits.setdefault(n_qubits, []).append(float(np.mean(np.abs(u_pred - u_ref))))

    stats = {}
    for fixed_layer, by_n_qubits in errors_by_layer.items():
        n_qubits_list = sorted(by_n_qubits)
        mae_mean = [float(np.mean(by_n_qubits[n])) for n in n_qubits_list]
        mae_std = [float(np.std(by_n_qubits[n])) for n in n_qubits_list]
        mae_min = [float(np.min(by_n_qubits[n])) for n in n_qubits_list]
        mae_max = [float(np.max(by_n_qubits[n])) for n in n_qubits_list]
        n_samples = [len(by_n_qubits[n]) for n in n_qubits_list]
        stats[fixed_layer] = (n_qubits_list, mae_mean, mae_std, mae_min, mae_max, n_samples)
    return stats, Tfin


def plot_errors_std(stats, Tfin, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_n_qubits = set()
    for (fixed_layer, (n_qubits, mae_mean, _mae_std, mae_min, mae_max, _n_samples)), color in zip(
            sorted(stats.items()), COLORS):
        all_n_qubits.update(n_qubits)
        yerr = asymmetric_yerr(mae_mean, mae_min, mae_max)
        ax.errorbar(n_qubits, mae_mean, yerr=yerr, fmt="o-", color=color,
                    linewidth=2.0, markersize=6, capsize=4, label=fixed_layer)
    ax.set_xlabel("Number of qubits", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_n_qubits))
    ax.set_title(f"Quantum PINN MAE vs number of qubits (T = {Tfin})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_dir in sorted(glob.glob(os.path.join(QUANTUM_DIR, "T_*"))):
        stats, Tfin = compute_error_stats(t_dir)
        if not stats:
            print(f"No quantum models found in {t_dir}, skipping.")
            continue

        t_label = os.path.basename(t_dir).replace("T_", "")
        save_path = f"Error_std_vs_Qubits_T_{t_label}.png"
        plot_errors_std(stats, Tfin, save_path)

        print(f"T = {Tfin}")
        for fixed_layer, (n_qubits, mae_mean, mae_std, mae_min, mae_max, n_samples) in sorted(stats.items()):
            print(f"  FIXED_LAYER={fixed_layer}")
            for n, a_mean, a_std, a_min, a_max, n_s in zip(
                n_qubits, mae_mean, mae_std, mae_min, mae_max, n_samples
            ):
                flag = "  <<< mean-std<0, plot uses [min,max] instead" if a_mean - a_std < 0 else ""
                print(f"    N_QUBITS={n:2d}  MAE={a_mean:.6e}+/-{a_std:.6e}  "
                      f"range=[{a_min:.6e}, {a_max:.6e}]  (n={n_s}){flag}")
        print(f"  -> saved {save_path}")
