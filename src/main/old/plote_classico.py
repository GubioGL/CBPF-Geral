import glob
import os

from main.config import NU
from main.functions import compute_classical_mean_errors, plot_classical_mean_errors
from main.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"


if __name__ == "__main__":
    for t_dir in sorted(glob.glob(os.path.join(CLASSICAL_DIR, "T_*"))):
        intra_neurons, mse, mae, n_samples, Tfin = compute_classical_mean_errors(t_dir, ClassicalPINN2, NU)
        if not intra_neurons:
            continue

        t_label = os.path.basename(t_dir).replace("T_", "")

        plot_classical_mean_errors(intra_neurons, mse, mae, Tfin, f"Error_vs_IntraNeuro_T_{t_label}.png")

        print(f"T = {Tfin}")
        for n, e_mse, e_mae, n_s in zip(intra_neurons, mse, mae, n_samples):
            print(f"  INTRA_NEURO={n:2d}  MSE={e_mse:.6e}  MAE={e_mae:.6e}  (mean over {n_s} samples)")
