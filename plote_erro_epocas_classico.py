"""Erro de predicao x epocas de treino, caso classico.

Junta num so grafico o erro (MSE contra a referencia FDM) de cada modelo
amostrado com o numero de passos (epocas AdamW + LBFGS) que ele precisou
para treinar, marcando quais amostras esgotaram o orcamento de epocas
(TARGET_REACHED=False, ou seja "usaram todas as epocas") e quais pararam
mais cedo por terem atingido o criterio de parada (TARGET_REACHED=True).

Grafico 3D: (INTRA_NEURO, epocas ate parar, MSE) -- com um plano marcando
o teto de epocas (EPOCHS + LBFGS_EPOCHS), pra deixar visualmente claro quem
bateu no teto.

So o caso classico (sem curva quantica sobreposta).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra a projecao 3d)

from main.config import NU
from main.functions import compute_classical_error_and_stopping_per_run
from main.models import ClassicalPINN2

CLASSICAL_DIR = "data/classical"
T_LABELS = ["0.1", "1"]
COLLOCATION_DISTRIBUTION = "random"
N_COLLOCATION_LIST = [10, 100]  # mesmos casos dos Error_std_vs_IntraNeuro_T_*_NCOL*.png

COLOR_REACHED = "#1baf7a"      # convergiu (atingiu TARGET_LOSS antes do teto)
COLOR_NOT_REACHED = "#e34948"  # usou todo o orcamento de epocas (nao convergiu)

LABEL_REACHED = "Convergiu (atingiu o criterio de parada)"
LABEL_NOT_REACHED = "Usou todas as epocas (nao convergiu)"


def _jitter_offsets(n, width=0.30):
    """Deslocamentos simetricos em torno de 0 para espalhar `n` pontos que
    caem no mesmo INTRA_NEURO, sem sobrepor -- um "beeswarm" simples e
    deterministico (mesma amostra sempre cai no mesmo lugar)."""
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def plot_3d_error_epochs(rows, Tfin, save_path, n_collocation):
    """(INTRA_NEURO, epocas ate parar, MSE) em 3D, colorido por
    TARGET_REACHED, com um plano tracejado no teto de epocas
    (EPOCHS + LBFGS_EPOCHS) para mostrar quem bateu no limite.

    O eixo z usa log10(MSE) plotado em escala linear (em vez de
    `ax.set_zscale("log")`) porque o eixo z log nativo do mplot3d tem bugs
    conhecidos de renderizacao (grade/ticks quebram); log10 manual da o
    mesmo resultado visual sem o bug."""
    fig = plt.figure(figsize=(9.5, 7))
    ax = fig.add_subplot(111, projection="3d")

    by_neuron = {}
    for row in rows:
        by_neuron.setdefault(row["intra_neuron"], []).append(row)

    epoch_budgets = {row["epoch_budget"] for row in rows if row["epoch_budget"] is not None}
    log_mse_all = [np.log10(r["mse"]) for r in rows]
    log_mse_min, log_mse_max = min(log_mse_all), max(log_mse_all)

    for n, group in by_neuron.items():
        group = sorted(group, key=lambda r: r["run_id"])
        offsets = _jitter_offsets(len(group))
        for row, dx in zip(group, offsets):
            if row["stopped_at_step"] is None:
                continue
            color = COLOR_REACHED if row["target_reached"] else COLOR_NOT_REACHED
            marker = "o" if row["target_reached"] else "^"
            ax.scatter(n + dx, row["stopped_at_step"], np.log10(row["mse"]), color=color,
                       marker=marker, s=45, alpha=0.9, depthshade=True)

    # Um plano por teto de epocas distinto encontrado nos dados -- diferentes
    # runs podem ter usado EPOCHS/LBFGS_EPOCHS diferentes (ex.: sweeps feitos
    # em momentos diferentes), entao um unico teto seria enganoso.
    if epoch_budgets:
        all_neurons = sorted(by_neuron)
        xx, zz = np.meshgrid(
            np.linspace(min(all_neurons) - 0.5, max(all_neurons) + 0.5, 2),
            np.linspace(log_mse_min, log_mse_max, 2),
        )
        for budget in sorted(epoch_budgets):
            yy = np.full_like(xx, budget)
            ax.plot_surface(xx, yy, zz, color="gray", alpha=0.12, linewidth=0)
            ax.text(max(all_neurons) + 0.5, budget, log_mse_max,
                    f"teto = {budget:g} epocas", color="dimgray", fontsize=8)

    ax.set_xlabel("INTRA_NEURO", fontsize=11, labelpad=8)
    ax.set_ylabel("Epocas ate parar", fontsize=11, labelpad=8)
    ax.set_zlabel("MSE (log10)", fontsize=11, labelpad=8)
    ax.set_title(f"Classical PINN: MSE x epocas x INTRA_NEURO (T = {Tfin}, N_COLLOCATION = {n_collocation})",
                 fontsize=11)

    handles = [
        plt.Line2D([0], [0], color=COLOR_REACHED, marker="o", linestyle="", label=LABEL_REACHED),
        plt.Line2D([0], [0], color=COLOR_NOT_REACHED, marker="^", linestyle="", label=LABEL_NOT_REACHED),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for t_label in T_LABELS:
        t_dir = os.path.join(CLASSICAL_DIR, f"T_{t_label}")
        for n_collocation in N_COLLOCATION_LIST:
            rows, Tfin = compute_classical_error_and_stopping_per_run(
                t_dir, ClassicalPINN2, NU, COLLOCATION_DISTRIBUTION, n_collocation=n_collocation
            )
            if not rows:
                print(f"No classical models found in {t_dir} for N_COLLOCATION={n_collocation}, skipping.")
                continue

            n_reached = sum(1 for r in rows if r["target_reached"])
            print(f"T={Tfin}  N_COLLOCATION={n_collocation}  "
                  f"{n_reached}/{len(rows)} amostras convergiram antes do teto de epocas")

            save_3d = f"ErroEpocas_3D_T_{t_label}_NCOL{n_collocation}.png"
            plot_3d_error_epochs(rows, Tfin, save_3d, n_collocation)
            print(f"  -> saved {save_3d}")
