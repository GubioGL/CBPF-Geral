"""Fig. 2 -- noiseless simulations, classical PINN vs HQPINN under the SAME protocol.

Data: data/Forward/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/qubit_sweep_runs.csv
(50 seeds per point, width n = 2..15, N_col in {10, 100}, T in {0.1, 1}; 20 000 AdamW + up to
1000 L-BFGS steps, stopping at the FDM residual floor monitored on a fixed validation set).

Rows: T = 0.1 (top) and T = 1 (bottom).
  (a,d) median MSE of u(x,T) vs width n (band: interquartile range);
  (b,e) empirical CDF of the MSE over the 50 seeds, n = 10;
  (c,f) cumulative fraction of runs that reached the physical stopping target vs optimizer step, n = 10.
"""
import numpy as np
from matplotlib.lines import Line2D

from paper_common import (C_CLASSICAL, C_EXACT, COL2, INK2, MUTED, load_sweep, panel_label, save,
                          set_style)

import matplotlib.pyplot as plt

MODELS = {"classical_mlp": ("Classical PINN", C_CLASSICAL), "quantum_exact": ("HQPINN (ideal circuit)", C_EXACT)}
NCOL_STYLE = {100: dict(ls="-", lw=1.3, marker="o", mfc=None), 10: dict(ls=(0, (3, 1.5)), lw=1.0, marker="o", mfc="white")}
N_SHOW = 10
ADAM_LIMIT = 20000


def mse_vs_width(ax, df, T):
    for fam, (_, color) in MODELS.items():
        for ncol, st in NCOL_STYLE.items():
            g = df[(df.model_family == fam) & (df.Tfin == T) & (df.n_collocation == ncol)]
            q = g.groupby("n_qubits").mse.quantile([0.25, 0.5, 0.75]).unstack()
            n = q.index.values
            ax.fill_between(n, q[0.25], q[0.75], color=color, alpha=0.10 if ncol == 100 else 0.06, lw=0)
            ax.plot(n, q[0.5], color=color, ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=3.2,
                    mfc=st["mfc"] or color, mec=color, mew=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"width $n$ (qubits / bottleneck neurons)")
    ax.set_ylabel(r"MSE of $u(x,T)$")
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14])


def ecdf(ax, df, T):
    for fam, (_, color) in MODELS.items():
        for ncol, st in NCOL_STYLE.items():
            v = np.sort(df[(df.model_family == fam) & (df.Tfin == T) & (df.n_collocation == ncol)
                           & (df.n_qubits == N_SHOW)].mse.values)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, ls=st["ls"], lw=st["lw"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(rf"MSE of $u(x,T)$, $n={N_SHOW}$")
    ax.set_ylabel("fraction of runs")


def time_to_target(ax, df, T):
    grid = np.linspace(0, 20050, 600)
    for fam, (_, color) in MODELS.items():
        for ncol, st in NCOL_STYLE.items():
            g = df[(df.model_family == fam) & (df.Tfin == T) & (df.n_collocation == ncol) & (df.n_qubits == N_SHOW)]
            steps = g.stopped_at_step.values[g.target_reached.values.astype(bool)]
            frac = np.array([(steps <= s).sum() for s in grid]) / len(g)
            ax.plot(grid / 1e3, frac, color=color, ls=st["ls"], lw=st["lw"])
    ax.axvline(ADAM_LIMIT / 1e3, color=MUTED, lw=0.6)
    ax.text(ADAM_LIMIT / 1e3, 1.03, "AdamW limit", ha="center", va="bottom", fontsize=5.5, color=INK2,
            transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 20.5)
    ax.set_xlabel(r"optimizer steps ($\times10^3$)")
    ax.set_ylabel("reached physical target")


def main():
    set_style()
    df = load_sweep()
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.9), constrained_layout=True)
    labels = iter("abcdef")
    for row, T in enumerate((0.1, 1.0)):
        mse_vs_width(axes[row, 0], df, T)
        ecdf(axes[row, 1], df, T)
        time_to_target(axes[row, 2], df, T)
        for ax in axes[row]:
            panel_label(ax, next(labels), x=-0.25)
        axes[row, 0].text(0.97, 0.95, rf"$T={T:g}$", transform=axes[row, 0].transAxes, ha="right", va="top",
                          fontsize=7.5, color=INK2)
    handles = [Line2D([], [], color=c, lw=1.3, label=lbl) for lbl, c in MODELS.values()]
    handles += [Line2D([], [], color=INK2, ls=st["ls"], lw=st["lw"], marker="o", ms=3.2,
                       mfc=st["mfc"] or INK2, mec=INK2, label=rf"$N_{{\rm col}}={n}$") for n, st in NCOL_STYLE.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, columnspacing=1.6)
    save(fig, "fig2_simulations")


if __name__ == "__main__":
    main()
