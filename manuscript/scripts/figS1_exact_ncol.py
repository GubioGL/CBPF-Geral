"""Fig. S1 (appendix) -- noiseless HQPINN vs number of collocation points, and parameter efficiency.

  (a,b) median MSE of u(x,T) vs N_col (10..500) for n = 3, 5, 7, 10, 15 qubits, T = 0.1 and 1
        (data/Forward/CBPF_quantum_exact_results_*, 50 seeds, same protocol as the sweep);
        squares: classical PINN, n = 10, same protocol (sweep, N_col = 10 and 100);
  (c)   median MSE vs number of trainable parameters for n = 2..15 (sweep), T = 1, N_col = 100;
        band: interquartile range.
"""
import numpy as np

from paper_common import (C_CLASSICAL, C_EXACT, COL2, INK2, load_exact_ncol, load_sweep, panel_label,
                          save, set_style)

import matplotlib.pyplot as plt

# ordinal steps of the orange series hue, light -> dark with the number of qubits
QUBIT_STEPS = {3: "#f39a6e", 5: "#ee7d4b", 7: "#eb6834", 10: "#c2501f", 15: "#8a3510"}


def main():
    set_style()
    ex, sw = load_exact_ncol(), load_sweep()
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2), constrained_layout=True)
    for ax, T, lab in zip(axes[:2], (0.1, 1.0), "ab"):
        for n, color in QUBIT_STEPS.items():
            g = ex[(ex.n_qubits == n) & (ex.Tfin == T)].groupby("n_collocation").mse
            q = g.quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(q.index, q[0.5], color=color, marker="o", ms=2.8, lw=1.0, mec=color, label=rf"$n={n}$")
        c = sw[(sw.model_family == "classical_mlp") & (sw.Tfin == T) & (sw.n_qubits == 10)]
        med = c.groupby("n_collocation").mse.median()
        ax.plot(med.index, med.values, ls="none", marker="s", ms=4, color=C_CLASSICAL, mec="white", mew=0.5,
                label=r"classical, $n=10$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"collocation points per epoch $N_{\rm col}$")
        ax.set_ylabel(r"median MSE of $u(x,T)$")
        ax.text(0.97, 0.95, rf"$T={T:g}$", transform=ax.transAxes, ha="right", va="top", color=INK2)
        panel_label(ax, lab, x=-0.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=6, fontsize=6,
               columnspacing=1.4)

    ax = axes[2]
    for fam, color, label in (("classical_mlp", C_CLASSICAL, "classical PINN"), ("quantum_exact", C_EXACT, "HQPINN (ideal)")):
        g = sw[(sw.model_family == fam) & (sw.Tfin == 1.0) & (sw.n_collocation == 100)]
        q = g.groupby("trainable_parameters").mse.quantile([0.25, 0.5, 0.75]).unstack()
        n_of = g.groupby("trainable_parameters").n_qubits.first()
        ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.10, lw=0)
        ax.plot(q.index, q[0.5], color=color, marker="o", ms=2.8, lw=1.1, label=label)
        dy = 5 if fam == "classical_mlp" else -9
        for P in (q.index[0], q.index[-1]):
            ax.annotate(rf"$n={n_of[P]}$", (P, q[0.5][P]), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=5.4, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([100, 200, 400, 800], ["100", "200", "400", "800"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel(r"MSE of $u(x,T)$")
    ax.text(0.97, 0.95, r"$T=1$, $N_{\rm col}=100$", transform=ax.transAxes, ha="right", va="top", color=INK2)
    ax.legend(loc="lower left", fontsize=5.6)
    panel_label(ax, "c", x=-0.25)
    save(fig, "figS1_exact_ncol")


if __name__ == "__main__":
    main()
