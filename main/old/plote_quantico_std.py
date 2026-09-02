import glob
import os

from main.config import NU
from main.functions import compute_quantum_error_stats_std, plot_quantum_errors_std
from main.models_pennylane import QuantumPINNPennyLane

QUANTUM_DIR = "data/quantum"
INPUT_ROTATION_AXIS = "ry"


if __name__ == "__main__":
    for t_dir in sorted(glob.glob(os.path.join(QUANTUM_DIR, "T_*"))):
        stats, Tfin = compute_quantum_error_stats_std(
            t_dir, QuantumPINNPennyLane, NU, input_rotation_axis=INPUT_ROTATION_AXIS
        )
        if not stats:
            print(f"No quantum models found in {t_dir}, skipping.")
            continue

        t_label = os.path.basename(t_dir).replace("T_", "")
        save_path = f"Error_std_vs_Qubits_T_{t_label}.png"
        plot_quantum_errors_std(stats, Tfin, save_path)

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
