import os
from main.config import NU
from main.functions import (
    compute_classical_error_stats_std,
    compute_classical_stopping_stats_std,
    plot_classical_vs_quantum_std,
    plot_classical_vs_quantum_steps_std,
    quantum_stats_from_sweep_summary,
    quantum_steps_stats_from_sweep_summary,
)
from main.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical2"
T_LABELS = ["0.1", "1"]
COLLOCATION_DISTRIBUTIONS = ["random"]

CLASSICAL_N_COLLOCATION = 10

# One of ERROR_BAR_TYPES:
# "minimo_maximo" (observed min/max),
# "desvio_padrao" (+/-std), or
# "variancia" (+/-variance).
ERROR_BAR_TYPE = "variancia"

# Passos ate a parada sao uma contagem (unidade "passos"), entao sua variancia
# vive em passos^2 -- bem maior que a media e estoura a escala do eixo y.
# Desvio padrao tem a mesma unidade da media, entao mantem o eixo legivel.
STEPS_ERROR_BAR_TYPE = "desvio_padrao"

# Quantum N_QUBITS sweep, read from qubit_sweep_summary.csv (already aggregated
# mean/min/max/std per network/n_cnot_layers/n_qubits/Tfin/n_collocation -- see
# `qubit_sweep_runs.csv` for the raw per-seed rows it was built from).
QUBIT_SWEEP_SUMMARY_CSV = "qubit_sweep_summary.csv"
QUANTUM_NETWORK = "brickwork_cnot"
QUANTUM_N_QUBITS = 4
QUANTUM_N_CNOT_LAYERS = 1
QUANTUM_N_COLLOCATION = CLASSICAL_N_COLLOCATION


if __name__ == "__main__":
    
    for t_label in T_LABELS:
            t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")

            stats_by_distribution = {}
            Tfin = None
            for distribution in COLLOCATION_DISTRIBUTIONS:
                (intra_neurons, mse_mean, mse_std, mse_var, mse_min, mse_max, mae_mean, mae_std,
                mae_var, mae_min, mae_max, n_samples, dist_Tfin) = compute_classical_error_stats_std(
                    t_dir, ClassicalPINN2, NU, distribution, n_collocation=CLASSICAL_N_COLLOCATION
                )
                if not intra_neurons:
                    continue
                Tfin = dist_Tfin
                stats_by_distribution[distribution] = (
                    intra_neurons, mse_mean, mse_std, mse_var, mse_min, mse_max
                )

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
            plot_classical_vs_quantum_std(
                stats_by_distribution, quantum_stats, Tfin, save_path, error_type=ERROR_BAR_TYPE
            )
            print(f"  -> saved {save_path}")

            steps_stats_by_distribution = {}
            Tfin_steps = None
            for distribution in COLLOCATION_DISTRIBUTIONS:
                (intra_neurons, steps_mean, steps_std, steps_var, steps_min, steps_max,
                target_reached_frac, n_samples, dist_Tfin) = compute_classical_stopping_stats_std(
                    t_dir, distribution, n_collocation=CLASSICAL_N_COLLOCATION
                )
                if not intra_neurons:
                    continue
                Tfin_steps = dist_Tfin
                steps_stats_by_distribution[distribution] = (
                    intra_neurons, steps_mean, steps_std, steps_var, steps_min, steps_max
                )

            if not steps_stats_by_distribution:
                print(f"No classical stopping-step data found in {t_dir}, skipping steps plot.")
                continue

            quantum_steps_stats = quantum_steps_stats_from_sweep_summary(
                QUBIT_SWEEP_SUMMARY_CSV,
                network=QUANTUM_NETWORK,
                n_cnot_layers=QUANTUM_N_CNOT_LAYERS,
                n_collocation=QUANTUM_N_COLLOCATION,
                Tfin=float(t_label),
            )
            if not quantum_steps_stats[0]:
                print(f"  no quantum sweep rows found for T={t_label} in {QUBIT_SWEEP_SUMMARY_CSV}")

            steps_save_path = f"Steps_vs_IntraNeuro_T_{t_label}_NCOL{CLASSICAL_N_COLLOCATION}.png"
            plot_classical_vs_quantum_steps_std(
                steps_stats_by_distribution, quantum_steps_stats, Tfin_steps, steps_save_path,
                error_type=STEPS_ERROR_BAR_TYPE
            )
            print(f"  -> saved {steps_save_path}")
