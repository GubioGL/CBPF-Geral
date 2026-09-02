import glob
import os

import numpy as np

from main.config import NU
from main.functions import (
    compute_classical_errors_comparativo,
    compute_quantum_errors_comparativo,
    find_quantum_configs,
    plot_comparison,
    reference_at_time,
    solve_burgers_fd,
)
from main.models import ClassicalPINN2
from main.models_pennylane import QuantumPINNPennyLane

QUANTUM_DIR = "data/quantum"
CLASSICAL_DIR = "data/classical"
INPUT_ROTATION_AXIS = "ry"
COLLOCATION_DISTRIBUTION = "uniform"  # "uniform" ou "random" -- filtra os checkpoints de ambos os lados
if COLLOCATION_DISTRIBUTION not in ("uniform", "random"):
    raise ValueError(
        f"COLLOCATION_DISTRIBUTION must be 'uniform' or 'random', got {COLLOCATION_DISTRIBUTION!r}."
    )


if __name__ == "__main__":
    for t_dir in sorted(glob.glob(os.path.join(QUANTUM_DIR, "T_*"))):
        entries = find_quantum_configs(
            t_dir, input_rotation_axis=INPUT_ROTATION_AXIS, collocation_distribution=COLLOCATION_DISTRIBUTION
        )
        if not entries:
            continue

        first_model = QuantumPINNPennyLane.load(entries[0][3])
        L, Tfin = first_model.L, first_model.Tfin

        x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
        x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
        u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

        quantum_stats = compute_quantum_errors_comparativo(
            t_dir, QuantumPINNPennyLane, x_plot, u_ref, Tfin,
            INPUT_ROTATION_AXIS, COLLOCATION_DISTRIBUTION
        )

        classical_t_dir = os.path.join(CLASSICAL_DIR, os.path.basename(t_dir))
        classical_data = compute_classical_errors_comparativo(
            classical_t_dir, ClassicalPINN2, x_plot, u_ref, Tfin, COLLOCATION_DISTRIBUTION
        )

        t_label = os.path.basename(t_dir).replace("T_", "")
        save_path = f"Error_vs_Width_{COLLOCATION_DISTRIBUTION}_T_{t_label}.png"
        plot_comparison(quantum_stats, classical_data, Tfin, save_path)

        # for fixed_layer, (n_qubits, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples) in sorted(
        #     quantum_stats.items()
        # ):
        #     print(f"  quantum fixed_layer={fixed_layer}")
        #     for n, e_mse, s_mse, lo_mse, hi_mse, e_mae, s_mae, n_s in zip(
        #         n_qubits, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples
        #     ):
        #         flag = "  <<< mean-std<0, plot uses [min,max] instead" if e_mse - s_mse < 0 else ""
        #         print(f"    N_QUBITS={n:2d}  MSE={e_mse:.6e}+/-{s_mse:.6e}  range=[{lo_mse:.6e}, {hi_mse:.6e}]  "
        #               f"MAE={e_mae:.6e}+/-{s_mae:.6e}  (mean over {n_s} seeds){flag}")
        # if classical_data is not None:
        #     intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples = classical_data
        #     print("  classical")
        #     for n, e_mse, s_mse, lo_mse, hi_mse, e_mae, s_mae, n_s in zip(
        #         intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples
        #     ):
        #         flag = "  <<< mean-std<0, plot uses [min,max] instead" if e_mse - s_mse < 0 else ""
        #         print(f"    INTRA_NEURO={n:2d}  MSE={e_mse:.6e}+/-{s_mse:.6e}  range=[{lo_mse:.6e}, {hi_mse:.6e}]  "
        #               f"MAE={e_mae:.6e}+/-{s_mae:.6e}  (mean over {n_s} runs){flag}")
        # else:
        #     print(f"  no classical models found in {classical_t_dir}")
        print(f"  -> saved {save_path}")
