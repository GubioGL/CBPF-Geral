import os

import matplotlib.pyplot as plt

from main.config import NU
from main.functions import (
    compute_classical_error_stats_vs_collocation,
    compute_classical_stopping_stats_vs_collocation,
)
from main.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
T_LABELS = ["0.1", "1"]
COLLOCATION_DISTRIBUTION = "random"  # deve bater com o usado em run_classical.py
INTRA_NEURO_LIST = [3, 10]  # arquiteturas comparadas no mesmo grafico
INTRA_NEURO_COLORS = {3: "#e34948", 10: "#2a78d6"}


def plot_error_comparison(series, Tfin, save_path):
    """series: lista de (intra_neuron, n_collocation, mse_mean, mse_var).
    Uma curva por INTRA_NEURO, cores distintas, legenda com o valor de
    INTRA_NEURO. Posicoes no eixo x sao igualmente espacadas (uniao dos
    N_COLLOCATION de todas as series), mesma convencao de
    plot_classical_error_vs_collocation."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_n = sorted({n for _, n_list, _, _ in series for n in n_list})
    positions = {n: i for i, n in enumerate(all_n)}

    for intra_neuron, n_collocation, mse_mean, mse_var in series:
        pos = [positions[n] for n in n_collocation]
        color = INTRA_NEURO_COLORS.get(intra_neuron)
        ax.errorbar(pos, mse_mean, yerr=mse_var, fmt="o-", color=color,
                     linewidth=2.0, markersize=6, capsize=4, label=str(intra_neuron))

    ax.set_xlabel("N_COLLOCATION", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([str(n) for n in all_n])
    ax.set_title(f"Classical PINN: MSE vs N_COLLOCATION (T = {Tfin})", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(title="INTRA_NEURO")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_steps_comparison(series, Tfin, save_path):
    """series: lista de (intra_neuron, n_collocation, steps_mean). Uma curva
    (media de passos ate a parada) por INTRA_NEURO, cores distintas, legenda
    com o valor de INTRA_NEURO."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_n = sorted({n for _, n_list, _ in series for n in n_list})
    positions = {n: i for i, n in enumerate(all_n)}

    for intra_neuron, n_collocation, steps_mean in series:
        pos = [positions[n] for n in n_collocation]
        color = INTRA_NEURO_COLORS.get(intra_neuron)
        ax.plot(pos, steps_mean, "o-", color=color, linewidth=2.0,
                 markersize=6, label=str(intra_neuron))

    ax.set_xlabel("N_COLLOCATION", fontsize=14)
    ax.set_ylabel("Passos ate a parada (media)", fontsize=14)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([str(n) for n in all_n])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="INTRA_NEURO")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")

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
                continue

            if n_collocation:
                error_series.append((intra_neuron, n_collocation, mse_mean, mse_var))
                Tfin_error = Tfin
                print(f"T={Tfin}  INTRA_NEURO={intra_neuron}")
                for n, e_mse, v_mse, n_s in zip(n_collocation, mse_mean, mse_var, n_samples):
                    print(f"  N_COLLOCATION={n:4d}  MSE={e_mse:.6e} +/- var={v_mse:.6e}  (n={n_s} seeds)")

            if n_stop:
                steps_series.append((intra_neuron, n_stop, steps_mean))
                Tfin_steps = Tfin_stop
                for n, s_mean, s_std, reached in zip(n_stop, steps_mean, steps_std, target_reached_frac):
                    print(f"  N_COLLOCATION={n:4d}  passos_ate_parar={s_mean:8.1f} +/- {s_std:6.1f}  "
                          f"target_reached={reached:.0%}")

        if error_series:
            save_path = f"Error_vs_N_COLLOCATION_T_{t_label}_NEURO3-10.png"
            plot_error_comparison(error_series, Tfin_error, save_path)
            print(f"  -> saved {save_path}")

        if steps_series:
            steps_save_path = f"Steps_vs_N_COLLOCATION_T_{t_label}_NEURO3-10.png"
            plot_steps_comparison(steps_series, Tfin_steps, steps_save_path)
            print(f"  -> saved {steps_save_path}")
