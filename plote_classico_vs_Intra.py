import os
from main.config import NU
from main.functions import (
    compute_classical_error_stats_std,
    plot_classical_vs_quantum_std,
    quantum_stats_from_sweep_summary,
)
from main.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
T_LABELS = ["0.1", "1"]
COLLOCATION_DISTRIBUTIONS = ["random"]

CLASSICAL_N_COLLOCATION = 100

# Quantum N_QUBITS sweep, read from qubit_sweep_summary.csv (already aggregated
# mean/min/max/std per network/n_cnot_layers/n_qubits/Tfin/n_collocation -- see
# `qubit_sweep_runs.csv` for the raw per-seed rows it was built from).
QUBIT_SWEEP_SUMMARY_CSV = "qubit_sweep_summary.csv"
QUANTUM_NETWORK = "brickwork_cnot"
QUANTUM_N_CNOT_LAYERS = 1
QUANTUM_N_COLLOCATION = CLASSICAL_N_COLLOCATION


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")

        stats_by_distribution = {}
        Tfin = None
        for distribution in COLLOCATION_DISTRIBUTIONS:
            (intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, mae_min,
             mae_max, n_samples, dist_Tfin) = compute_classical_error_stats_std(
                t_dir, ClassicalPINN2, NU, distribution, n_collocation=CLASSICAL_N_COLLOCATION
            )
            if not intra_neurons:
                continue
            Tfin = dist_Tfin
            stats_by_distribution[distribution] = (intra_neurons, mse_mean, mse_std)

        if not stats_by_distribution:
            print(f"No classical models found in {t_dir}, skipping.")
            continue

        quantum_stats = quantum_stats_from_sweep_summary(
            QUBIT_SWEEP_SUMMARY_CSV,
            network=QUANTUM_NETWORK,
            n_cnot_layers=QUANTUM_N_CNOT_LAYERS,
            n_collocation=QUANTUM_N_COLLOCATION,
            Tfin=float(t_label),
        )
        if not quantum_stats[0]:
            print(f"  no quantum sweep rows found for T={t_label} in {QUBIT_SWEEP_SUMMARY_CSV}")

        save_path = f"Error_std_vs_IntraNeuro_T_{t_label}_NCOL{CLASSICAL_N_COLLOCATION}.png"
        plot_classical_vs_quantum_std(stats_by_distribution, quantum_stats, Tfin, save_path)
        print(f"  -> saved {save_path}")
