"""Folds the N_QUBITS=1 raw runs in data/Dados1Qubit.csv (50 seeds x
Tfin in {0.1, 1.0} x n_collocation in {10, 100}, quantum_exact/brickwork_cnot)
into data/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/
qubit_sweep_summary.csv, the aggregated file `plote_classico_vs_Intra.py`
reads for its quantum N_QUBITS-sweep curve. Run this once whenever
Dados1Qubit.csv is refreshed; it overwrites just the n_qubits=1 rows in the
summary and leaves every other row untouched.
"""
from main.functions import merge_raw_runs_into_sweep_summary

RAW_CSV = "data/Dados1Qubit.csv"
SUMMARY_CSV = "data/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/qubit_sweep_summary.csv"

if __name__ == "__main__":
    merged = merge_raw_runs_into_sweep_summary(RAW_CSV, SUMMARY_CSV)
    new_rows = [row for row in merged if int(float(row["n_qubits"])) == 1]
    print(f"{SUMMARY_CSV}: {len(merged)} rows total, {len(new_rows)} for n_qubits=1")
    for row in new_rows:
        print(
            f"  Tfin={row['Tfin']} n_collocation={row['n_collocation']} "
            f"mse_mean={row['mse_mean']:.3e} target_reached_fraction={row['target_reached_fraction']}"
        )
