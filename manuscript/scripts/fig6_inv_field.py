"""Inverse problem, simulation: reconstruction of the field at the final time, one row per final time.

Data: manuscript/data/inverse/ (field.csv, runs.csv).

  (a,c) u(x,T) at 500 epochs for the paired run n = k = 10, seed 0: exact solution, classical PINN, HQPINN on the emulator;
  (b,d) relative L2 error of u(x,T) of every run at both budgets (33 runs per column; ticks, medians): classical PINN,
        HQPINN with the circuit on the emulator, and the same HQPINN weights with the ideal circuit.
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _ROOT / "src" / "quantum_hardware", _HERE):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------

import numpy as np

from paper_common import C_CLASSICAL, C_EMU, COL2, INK, INK2, MUTED, SURFACE, panel_label, read_table, save, set_style

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BUDGETS = (110, 500)
COLUMNS = [("l2", "classical", C_CLASSICAL, "o"), ("l2", "hqpinn", C_EMU, "^"), ("l2_ideal", "hqpinn", MUTED, "s")]


def profile(ax, df, T):
    g = df[(df.epochs == 500) & (df.Tfin == T)]
    ax.plot(g.x, g.u_exact, color=INK, lw=1.2, zorder=3)
    ax.plot(g.x, g.u_classical, color=C_CLASSICAL, lw=1.0, zorder=4)
    ax.plot(g.x[1::4], g.u_device[1::4], ls="none", marker="^", ms=3.0, color=C_EMU, mec=SURFACE, mew=0.3, zorder=5)
    ax.set(xlim=(0, 1), xlabel=r"$x$", ylabel=rf"$u(x,T={T:g})$")


def errors(ax, runs, T):
    rng = np.random.default_rng(1)
    for b, epochs in enumerate(BUDGETS):
        for c, (column, model, color, marker) in enumerate(COLUMNS):
            v = runs[(runs.epochs == epochs) & (runs.Tfin == T) & (runs.model == model)][column].values
            x = b * 3.6 + c
            ax.scatter(x + rng.uniform(-0.2, 0.2, len(v)), v, s=8, marker=marker, color=color, edgecolor=SURFACE, lw=0.3,
                       alpha=0.85, zorder=3)
            ax.plot([x - 0.33, x + 0.33], [np.median(v)] * 2, color=INK, lw=1.0, zorder=4)
    ax.set_xticks([1, 4.6], [f"{e} epochs" for e in BUDGETS])
    ax.tick_params(axis="x", length=0)
    ax.grid(False, axis="x")
    ax.set(yscale="log", ylabel=r"$\epsilon_{L_2}$ of $u(x,T)$", xlim=(-0.7, 6.3))


def main():
    set_style()
    df, runs = read_table("inverse/field.csv"), read_table("inverse/runs.csv")
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 3.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.15]))
    labels = iter("abcd")
    for row, T in enumerate((0.1, 1.0)):
        profile(axes[row][0], df, T)
        errors(axes[row][1], runs, T)
        for ax in axes[row]:
            panel_label(ax, next(labels), x=-0.13)
    handles = [Line2D([], [], color=INK, lw=1.2, label="exact (Cole-Hopf)"),
               Line2D([], [], color=C_CLASSICAL, lw=1.0, marker="o", ms=3, mec=SURFACE, mew=0.3, label="classical PINN"),
               Line2D([], [], color=C_EMU, ls="none", marker="^", ms=3.4, mec=SURFACE, mew=0.3, label="HQPINN, circuit on the emulator"),
               Line2D([], [], color=MUTED, ls="none", marker="s", ms=3, mec=SURFACE, mew=0.3, label="HQPINN, same weights, ideal circuit")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, columnspacing=1.4)
    save(fig, "fig6_inv_field")


if __name__ == "__main__":
    main()
