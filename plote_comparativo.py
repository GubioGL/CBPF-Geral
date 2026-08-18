"""Plots mean MSE (error bars show the [min, max] range across SEED/run_id
samples) of the trained quantum PINN models (one line per FIXED_LAYER,
averaged across SEED) against N_QUBITS, overlaid with the mean MSE of the
trained classical PINN models (one line, averaged across run_id) against
INTRA_NEURO on the same "bottleneck width" axis, comparing each model's
prediction at t=Tfin to the FDM reference solution. One figure is produced
per T_* directory present in data/quantum (skipped if there is no matching
T_* directory in data/classical).

Error bars show [min, max] across seeds rather than mean +/- std: MSE is
always positive and, with only a handful of seeds, its std is frequently
larger than its mean, which would push a symmetric mean +/- std bar below
zero -- something the log-scale y-axis below cannot render, so matplotlib
clips that whisker all the way to the bottom of the plot. [min, max] is
built from samples that were actually observed, so it is always positive
and never overruns the axis. mean +/- std is still printed to the console
for reference."""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from old.plote_classico import find_classical_configs
from plote_quantico_std import find_quantum_configs
from src.config import NU
from src.functions import asymmetric_yerr, predict, reference_at_time, solve_burgers_fd
from src.models import ClassicalPINN2
from src.models_pennylane import QuantumPINNPennyLane

QUANTUM_DIR = "data/quantum"
CLASSICAL_DIR = "data/classical"
INPUT_ROTATION_AXIS = "ry"
COLOR_CLASSICAL = "#e34948"
COLORS_QUANTUM = {"cz_ring": "#2a78d6", "ring_cnot": "#1baf7a"}
FALLBACK_COLORS = ["#eb6834", "#4a3aa7", "#e87ba4"]


def compute_quantum_errors(t_dir, x_plot, u_ref, Tfin):
    """Loads every Qmodel_random_NQUBITS* checkpoint in `t_dir` matching
    INPUT_ROTATION_AXIS, evaluates each against `u_ref`, and averages
    MSE/MAE across the SEED samples for each (fixed_layer, n_qubits).
    Returns {fixed_layer: (n_qubits_list, mse_mean, mse_std, mse_min,
    mse_max, mae_mean, mae_std, n_samples)}. mse_min/mse_max (the extremes
    actually observed across seeds) are what the plot uses for its error
    bars -- unlike mean +/- std, they can never go negative on the plot's
    log-scale y-axis."""
    entries = find_quantum_configs(t_dir, input_rotation_axis=INPUT_ROTATION_AXIS)

    errors_by_layer = {}
    for fixed_layer, n_qubits, _seed, pt_path in entries:
        model = QuantumPINNPennyLane.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        by_n = errors_by_layer.setdefault(fixed_layer, {})
        entry = by_n.setdefault(n_qubits, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    stats = {}
    for fixed_layer, by_n in errors_by_layer.items():
        n_list = sorted(by_n)
        mse_mean = [float(np.mean(by_n[n]["mse"])) for n in n_list]
        mse_std = [float(np.std(by_n[n]["mse"])) for n in n_list]
        mse_min = [float(np.min(by_n[n]["mse"])) for n in n_list]
        mse_max = [float(np.max(by_n[n]["mse"])) for n in n_list]
        mae_mean = [float(np.mean(by_n[n]["mae"])) for n in n_list]
        mae_std = [float(np.std(by_n[n]["mae"])) for n in n_list]
        n_samples = [len(by_n[n]["mse"]) for n in n_list]
        stats[fixed_layer] = (n_list, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples)
    return stats


def compute_classical_errors(t_dir, x_plot, u_ref, Tfin):
    """Loads every Cmodel_random_NEURO* checkpoint in `t_dir`, evaluates each
    against `u_ref`, and averages MSE/MAE across the run_id samples for each
    INTRA_NEURO. Returns (intra_neurons, mse_mean, mse_std, mse_min, mse_max,
    mae_mean, mae_std, n_samples), or None if `t_dir` has no classical
    checkpoints. mse_min/mse_max (the extremes actually observed across
    run_ids) are what the plot uses for its error bars -- unlike mean +/-
    std, they can never go negative on the plot's log-scale y-axis."""
    entries = find_classical_configs(t_dir)
    if not entries:
        return None

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
    mse_min = [float(np.min(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_max = [float(np.max(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mae_mean = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_std = [float(np.std(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
    return intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples


def plot_comparison(quantum_stats, classical_data, Tfin, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_widths = set()

    fallback = iter(FALLBACK_COLORS)
    for fixed_layer, (n_qubits, mse_mean, _mse_std, mse_min, mse_max, _mae_mean, _mae_std, _n_samples) in sorted(
        quantum_stats.items()
    ):
        color = COLORS_QUANTUM.get(fixed_layer, next(fallback, None))
        all_widths.update(n_qubits)
        yerr = asymmetric_yerr(mse_mean, mse_min, mse_max)
        ax.errorbar(n_qubits, mse_mean, yerr=yerr, fmt="o-", color=color, linewidth=2.0,
                    markersize=6, capsize=4, label=f"Quantum PINN ({fixed_layer})")

    if classical_data is not None:
        intra_neurons, mse_mean, _mse_std, mse_min, mse_max, _mae_mean, _mae_std, _n_samples = classical_data
        all_widths.update(intra_neurons)
        yerr = asymmetric_yerr(mse_mean, mse_min, mse_max)
        ax.errorbar(intra_neurons, mse_mean, yerr=yerr, fmt="s--", color=COLOR_CLASSICAL,
                    linewidth=2.0, markersize=6, capsize=4, label="Classical PINN")

    ax.set_xlabel("Width (N_QUBITS / INTRA_NEURO)", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_widths))
    ax.set_title(f"Quantum vs classical PINN mean MSE vs width (T = {Tfin})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_dir in sorted(glob.glob(os.path.join(QUANTUM_DIR, "T_*"))):
        entries = find_quantum_configs(t_dir, input_rotation_axis=INPUT_ROTATION_AXIS)
        if not entries:
            continue

        first_model = QuantumPINNPennyLane.load(entries[0][3])
        L, Tfin = first_model.L, first_model.Tfin

        x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
        x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
        u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

        quantum_stats = compute_quantum_errors(t_dir, x_plot, u_ref, Tfin)

        classical_t_dir = os.path.join(CLASSICAL_DIR, os.path.basename(t_dir))
        classical_data = compute_classical_errors(classical_t_dir, x_plot, u_ref, Tfin)

        t_label = os.path.basename(t_dir).replace("T_", "")
        save_path = f"Error_vs_Width_T_{t_label}.png"
        plot_comparison(quantum_stats, classical_data, Tfin, save_path)

        print(f"T = {Tfin}")
        for fixed_layer, (n_qubits, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples) in sorted(
            quantum_stats.items()
        ):
            print(f"  quantum fixed_layer={fixed_layer}")
            for n, e_mse, s_mse, lo_mse, hi_mse, e_mae, s_mae, n_s in zip(
                n_qubits, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples
            ):
                flag = "  <<< mean-std<0, plot uses [min,max] instead" if e_mse - s_mse < 0 else ""
                print(f"    N_QUBITS={n:2d}  MSE={e_mse:.6e}+/-{s_mse:.6e}  range=[{lo_mse:.6e}, {hi_mse:.6e}]  "
                      f"MAE={e_mae:.6e}+/-{s_mae:.6e}  (mean over {n_s} seeds){flag}")
        if classical_data is not None:
            intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples = classical_data
            print("  classical")
            for n, e_mse, s_mse, lo_mse, hi_mse, e_mae, s_mae, n_s in zip(
                intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples
            ):
                flag = "  <<< mean-std<0, plot uses [min,max] instead" if e_mse - s_mse < 0 else ""
                print(f"    INTRA_NEURO={n:2d}  MSE={e_mse:.6e}+/-{s_mse:.6e}  range=[{lo_mse:.6e}, {hi_mse:.6e}]  "
                      f"MAE={e_mae:.6e}+/-{s_mae:.6e}  (mean over {n_s} runs){flag}")
        else:
            print(f"  no classical models found in {classical_t_dir}")
        print(f"  -> saved {save_path}")
