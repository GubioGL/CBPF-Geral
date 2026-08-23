import os
from main.config import NU
from main.functions import (
    compute_classical_error_stats_vs_collocation,
    compute_classical_stopping_stats_vs_collocation,
    plot_classical_error_vs_collocation,
    plot_classical_stopping_steps_vs_collocation,
)
from main.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
T_LABELS = ["0.1", "1"]
COLLOCATION_DISTRIBUTION = "uniform"  # deve bater com o usado em run_classical.py

if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")

        (n_collocation, mse_mean, mse_std, mse_var, mae_mean, mae_std, mae_var, n_samples, Tfin) = compute_classical_error_stats_vs_collocation(
            t_dir, ClassicalPINN2, NU, COLLOCATION_DISTRIBUTION
        )

        if not n_collocation:
            print(f"No classical models found in {t_dir}, skipping.")
            continue

        save_path = f"Error_vs_N_COLLOCATION_T_{t_label}.png"
        plot_classical_error_vs_collocation(n_collocation, mse_mean, mse_var, Tfin, save_path)

        (_n_stop, steps_mean, steps_std, steps_var, target_reached_frac,_n_samples_stop, _Tfin_stop) = compute_classical_stopping_stats_vs_collocation(
            t_dir, COLLOCATION_DISTRIBUTION )

        steps_save_path = f"Steps_vs_N_COLLOCATION_T_{t_label}.png"
        plot_classical_stopping_steps_vs_collocation(n_collocation, steps_mean, steps_var, Tfin, steps_save_path)

        # print(f"T={Tfin}")
        # for n, e_mse, v_mse, n_s, s_mean, s_std, reached in zip(
        #     n_collocation, mse_mean, mse_var, n_samples, steps_mean, steps_std, target_reached_frac
        # ):
        #     print(f"  N_COLLOCATION={n:4d}  MSE={e_mse:.6e} +/- var={v_mse:.6e}  (n={n_s} seeds)  "
        #           f"passos_ate_parar={s_mean:8.1f} +/- {s_std:6.1f}  target_reached={reached:.0%}")
        print(f"  -> saved {save_path}")
        print(f"  -> saved {steps_save_path}")
