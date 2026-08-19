
import os

import matplotlib.pyplot as plt
import numpy as np

from old.plote_classico import find_classical_configs
from src.config import NU
from src.functions import asymmetric_yerr, predict, reference_at_time, solve_burgers_fd
from src.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
T_LABELS = ["0.1", "1"]
COLLOCATION_DISTRIBUTIONS = ["uniform", "random"]
COLOR_MSE = "#2a78d6"
# Line style / marker per COLLOCATION_DISTRIBUTION, so both variants can share
# one set of axes: solid filled markers for "uniform", dashed hollow markers
# for "random".
DIST_STYLES = {
    "uniform": dict(fmt="o-", markerfacecolor=None),
    "random": dict(fmt="s--", markerfacecolor="none"),
}


def compute_error_stats(t_dir, collocation_distribution):
    """Loads every classical model in `t_dir` whose COLLOCATION_DISTRIBUTION
    matches `collocation_distribution`, evaluates each against the FDM
    reference at t=Tfin, and returns per-INTRA_NEURO mean/std/min/max MSE and
    MAE across the run_id samples. Returns (intra_neurons, mse_mean, mse_std,
    mse_min, mse_max, mae_mean, mae_std, mae_min, mae_max, n_samples, Tfin),
    one point per INTRA_NEURO. The _min/_max values (the extremes actually
    observed across run_ids) are what the plot uses for its error bars --
    unlike mean +/- std, they can never go negative on the plot's log-scale
    y-axis."""
    entries = find_classical_configs(t_dir, collocation_distribution=collocation_distribution)
    if not entries:
        return [], [], [], [], [], [], [], [], [], [], None

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
    mse_min = [float(np.min(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_max = [float(np.max(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mae_mean = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_std = [float(np.std(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_min = [float(np.min(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_max = [float(np.max(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
    return (intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, mae_min,
            mae_max, n_samples, Tfin)


def plot_errors_std(stats_by_distribution, Tfin, save_path):
    """`stats_by_distribution` maps COLLOCATION_DISTRIBUTION -> (intra_neurons,
    mse_mean, mse_min, mse_max) for that distribution; distributions with no
    models are omitted by the caller."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_neurons = set()
    for distribution, (intra_neurons, mse_mean, mse_min,
                        mse_max) in stats_by_distribution.items():
        all_neurons.update(intra_neurons)
        style = DIST_STYLES.get(distribution, dict(fmt="o-", markerfacecolor=None))
        mse_yerr = asymmetric_yerr(mse_mean, mse_min, mse_max)
        ax.errorbar(intra_neurons, mse_mean, yerr=mse_yerr, fmt=style["fmt"], color=COLOR_MSE,
                    markerfacecolor=style["markerfacecolor"], linewidth=2.0, markersize=6,
                    capsize=4, label=f"MSE ({distribution})")
    ax.set_xlabel("INTRA_NEURO", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_neurons))
    ax.set_title(f"Classical PINN error vs INTRA_NEURO (T = {Tfin})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")

        stats_by_distribution = {}
        Tfin = None
        for distribution in COLLOCATION_DISTRIBUTIONS:
            (intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, mae_min,
             mae_max, n_samples, dist_Tfin) = compute_error_stats(t_dir, distribution)
            if not intra_neurons:
                continue
            Tfin = dist_Tfin
            stats_by_distribution[distribution] = (intra_neurons, mse_mean, mse_min, mse_max)


        if not stats_by_distribution:
            print(f"No classical models found in {t_dir}, skipping.")
            continue

        save_path = f"Error_std_vs_IntraNeuro_T_{t_label}.png"
        plot_errors_std(stats_by_distribution, Tfin, save_path)
        print(f"  -> saved {save_path}")
