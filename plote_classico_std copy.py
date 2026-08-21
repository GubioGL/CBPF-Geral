import os

from main.config import NU
from main.functions import compute_classical_mae_stats, plot_classical_mae_std
from main.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
T_LABELS = ["0.1", "1"]


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")
        intra_neurons, mae_mean, mae_std, mae_min, mae_max, n_samples, Tfin = compute_classical_mae_stats(
            t_dir, ClassicalPINN2, NU
        )
        if not intra_neurons:
            print(f"No classical models found in {t_dir}, skipping.")
            continue

        save_path = f"Error_std_vs_IntraNeuro_T_{t_label}_old.png"
        plot_classical_mae_std(intra_neurons, mae_mean, mae_min, mae_max, Tfin, save_path)

        print(f"T = {Tfin}")
        for n, a_mean, a_std, a_min, a_max, n_s in zip(
            intra_neurons, mae_mean, mae_std, mae_min, mae_max, n_samples
        ):
            flag = "  <<< mean-std<0, plot uses [min,max] instead" if a_mean - a_std < 0 else ""
            print(f"  INTRA_NEURO={n:2d}  MAE={a_mean:.6e}+/-{a_std:.6e}  range=[{a_min:.6e}, {a_max:.6e}]  (n={n_s}){flag}")
        print(f"  -> saved {save_path}")
