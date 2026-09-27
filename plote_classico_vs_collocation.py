import os
import matplotlib.pyplot as plt

from main.config import NU
from main.functions import (
    compute_classical_error_stats_vs_collocation,
    compute_classical_stopping_stats_vs_collocation,
    quantum_exact_stats_vs_collocation,
)
from main.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
T_LABELS = ["0.1", "1"]
COLLOCATION_DISTRIBUTION = "random"  # must match the one used in run_classical.py
INTRA_NEURO_LIST = [3, 5, 7, 10, 15]  # architectures compared on the same plot
INTRA_NEURO_COLORS = {
    3: "#e34948",
    5: "#e39a1f",
    7: "#3aa15c",
    10: "#2a78d6",
    15: "#8a4fd1",
}
MAX_N_COLLOCATION = 100  # cap the sweep points shown (quantum goes up to 500)

# Exact quantum results (run_quantumExact.ipynb), one directory per N_QUBITS
QUANTUM_EXACT_DIR_BY_N_QUBITS = {
    3: "data/CBPF_quantum_exact_results_Brickwork3",
    5: "data/CBPF_quantum_exact_results__Brickwork5",
    7: "data/CBPF_quantum_exact_results_fixed_BrickWork7",
    10: "data/CBPF_quantum_exact_results_fixed_BrickWork10",
    15: "data/CBPF_quantum_exact_results_fixed_BrickWork15",
}
QUANTUM_NETWORK = "brickwork_cnot"
QUANTUM_N_CNOT_LAYERS = 1
QUANTUM_COLOR = "black"
QUANTUM_LINESTYLES = {3: "--", 5: "-.", 7: ":", 10: (0, (3, 1, 1, 1)), 15: (0, (5, 1))}

# Fixed marker per architecture (INTRA_NEURO / N_QUBITS), shared between
# classical and quantum -- only color/linestyle distinguish the two.
MARKER_BY_PARAM = {3: "o", 5: "^", 7: "D", 10: "s", 15: "v"}


def _limit_n_collocation(n_collocation, *value_lists, max_n=MAX_N_COLLOCATION):
    """Keeps only the points with N_COLLOCATION <= max_n, applying the same
    mask to n_collocation and to each list in value_lists (same indexing)."""
    keep = [i for i, n in enumerate(n_collocation) if n <= max_n]
    filtered_n = [n_collocation[i] for i in keep]
    filtered_values = tuple([values[i] for i in keep] for values in value_lists)
    return (filtered_n, *filtered_values)


def plot_error_comparison(series, Tfin, save_path):
    """series: list of (kind, param, n_collocation, mse_mean, mse_var), kind
    in {"Classical", "Quantum"} and param = INTRA_NEURO (classical) or
    N_QUBITS (quantum). One curve per (kind, param): fixed marker per param
    (MARKER_BY_PARAM), classical colored by param (INTRA_NEURO_COLORS) with
    a solid line, quantum always black with its own dashed/dotted linestyle
    (QUANTUM_LINESTYLES). X-axis positions are evenly spaced (union of the
    N_COLLOCATION values across all series), same convention as
    plot_classical_error_vs_collocation."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_n = sorted({n for _, _, n_list, _, _ in series for n in n_list})
    positions = {n: i for i, n in enumerate(all_n)}

    for kind, param, n_collocation, mse_mean, mse_var in series:
        pos = [positions[n] for n in n_collocation]
        marker = MARKER_BY_PARAM.get(param, "o")
        if kind == "Quantum":
            linestyle = QUANTUM_LINESTYLES.get(param, "--")
            color = QUANTUM_COLOR
            label = f"{param} - Qubits"
        else:
            linestyle = "-"
            color = INTRA_NEURO_COLORS.get(param)
            label = f"{param} - Neurons "
        ax.errorbar(pos, mse_mean, yerr=mse_var, color=color, linestyle=linestyle,
                     marker=marker, linewidth=2.0, markersize=6, capsize=4, label=label)

    ax.set_xlabel("N", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([str(n) for n in all_n])
    ax.set_title(f"T = {Tfin}", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_steps_comparison(series, Tfin, save_path):
    """series: list of (kind, param, n_collocation, steps_mean), same
    kind/param/style convention as plot_error_comparison. One curve
    (average steps until stopping) per (kind, param)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_n = sorted({n for _, _, n_list, _ in series for n in n_list})
    positions = {n: i for i, n in enumerate(all_n)}

    for kind, param, n_collocation, steps_mean in series:
        pos = [positions[n] for n in n_collocation]
        marker = MARKER_BY_PARAM.get(param, "o")
        if kind == "Quantum":
            linestyle = QUANTUM_LINESTYLES.get(param, "--")
            color = QUANTUM_COLOR
            label = f"{param} - Qubits)"
        else:
            linestyle = "-"
            color = INTRA_NEURO_COLORS.get(param)
            label = f"{param} - Neurons"
        ax.plot(pos, steps_mean, color=color, linestyle=linestyle, marker=marker,
                 linewidth=2.0, markersize=6, label=label)

    ax.set_xlabel("N", fontsize=14)
    ax.set_ylabel("Steps of train", fontsize=14)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([str(n) for n in all_n])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")
        Tfin_target = float(t_label)

        error_series = []
        steps_series = []
        Tfin_error = None
        Tfin_steps = None

        for intra_neuron in INTRA_NEURO_LIST:
            (n_collocation, mse_mean, mse_std, mse_var, mae_mean, mae_std, mae_var,
             n_samples, Tfin) = compute_classical_error_stats_vs_collocation(
                t_dir, ClassicalPINN2, NU, COLLOCATION_DISTRIBUTION, intra_neuron=intra_neuron
            )

            (n_stop, steps_mean, steps_std, steps_var, target_reached_frac,
             n_samples_stop, Tfin_stop) = compute_classical_stopping_stats_vs_collocation(
                t_dir, COLLOCATION_DISTRIBUTION, intra_neuron=intra_neuron
            )

            if not n_collocation and not n_stop:
                print(f"No classical models found in {t_dir} for INTRA_NEURO={intra_neuron}, skipping.")
            else:
                if n_collocation:
                    error_series.append(("Classical", intra_neuron, n_collocation, mse_mean, mse_var))
                    Tfin_error = Tfin

                if n_stop:
                    steps_series.append(("Classical", intra_neuron, n_stop, steps_mean))
                    Tfin_steps = Tfin_stop

            # Exact quantum curve (run_quantumExact.ipynb) with the same
            # network width (N_QUBITS == intra_neuron), overlaid on the
            # classical curve above.
            quantum_dir = QUANTUM_EXACT_DIR_BY_N_QUBITS.get(intra_neuron)
            if quantum_dir is None:
                print(f"No quantum data directory configured for N_QUBITS={intra_neuron}, skipping quantum curve.")
                continue
            quantum_csv = os.path.join(quantum_dir, "quantum_exact_summary.csv")
            (q_n_collocation, q_mse_mean, q_mse_std, q_mse_var, q_steps_mean, q_steps_std,
             q_steps_var, q_target_reached_frac, q_n_samples) = quantum_exact_stats_vs_collocation(
                quantum_csv, Tfin_target, network=QUANTUM_NETWORK, n_cnot_layers=QUANTUM_N_CNOT_LAYERS
            )

            if not q_n_collocation:
                print(f"No quantum results found in {quantum_csv} for Tfin={Tfin_target}, skipping.")
                continue

            (q_n_collocation, q_mse_mean, q_mse_var, q_steps_mean, q_steps_std,
             q_target_reached_frac, q_n_samples) = _limit_n_collocation(
                q_n_collocation, q_mse_mean, q_mse_var, q_steps_mean, q_steps_std,
                q_target_reached_frac, q_n_samples)

            error_series.append(("Quantum", intra_neuron, q_n_collocation, q_mse_mean, q_mse_var))
            steps_series.append(("Quantum", intra_neuron, q_n_collocation, q_steps_mean))
            Tfin_error = Tfin_error or Tfin_target
            Tfin_steps = Tfin_steps or Tfin_target

        neuro_suffix = "-".join(str(n) for n in INTRA_NEURO_LIST)

        if error_series:
            save_path = f"Error_vs_N_COLLOCATION_T_{t_label}_NEURO{neuro_suffix}.png"
            plot_error_comparison(error_series, Tfin_error, save_path)
            print(f"  -> saved {save_path}")

        if steps_series:
            steps_save_path = f"Steps_vs_N_COLLOCATION_T_{t_label}_NEURO{neuro_suffix}.png"
            plot_steps_comparison(steps_series, Tfin_steps, steps_save_path)
            print(f"  -> saved {steps_save_path}")
