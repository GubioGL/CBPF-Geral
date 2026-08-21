import os

from main.config import NU
from main.functions import compute_classical_error_stats_std, plot_classical_errors_std
from main.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
T_LABELS = ["0.1", "1"]
COLLOCATION_DISTRIBUTIONS = ["uniform", "random"]


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")

        stats_by_distribution = {}
        Tfin = None
        for distribution in COLLOCATION_DISTRIBUTIONS:
            (intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, mae_min,
             mae_max, n_samples, dist_Tfin) = compute_classical_error_stats_std(
                t_dir, ClassicalPINN2, NU, distribution
            )
            if not intra_neurons:
                continue
            Tfin = dist_Tfin
            stats_by_distribution[distribution] = (intra_neurons, mse_mean, mse_min, mse_max)

        if not stats_by_distribution:
            print(f"No classical models found in {t_dir}, skipping.")
            continue

        save_path = f"Error_std_vs_IntraNeuro_T_{t_label}.png"
        plot_classical_errors_std(stats_by_distribution, Tfin, save_path)
        print(f"  -> saved {save_path}")
