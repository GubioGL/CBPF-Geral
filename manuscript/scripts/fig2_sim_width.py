"""Fig. 2 -- simulations: growing the number of qubits / neurons, classical PINN vs noiseless HQPINN, same setup.

Data: manuscript/data/forward/simulation/width_sweep.csv (n = 2..15, N_col = 10 and 100, 50 seeds per point). Colour: model.
Marker and line: N_col.

Rows: T = 0.1 (top) and T = 1 (bottom).
  (a,d) median MSE of u(x,T) vs width n (band: interquartile range);
  (b,e) mean optimizer steps of the classical PINN divided by those of the HQPINN, vs n (log axis; above 1, the HQPINN needs fewer
        steps). A run that misses the target counts at the budget, so both means are lower bounds of the steps needed;
  (c,f) median MSE vs number of trainable parameters.
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
from matplotlib.lines import Line2D

from paper_common import (C_CLASSICAL, C_EXACT, COL2, INK, INK2, MUTED, panel_label, read_table, save, set_style)

import matplotlib.pyplot as plt

MODELS = {"classical": ("Classical PINN", C_CLASSICAL), "hqpinn": ("HQPINN (noiseless)", C_EXACT)}
NCOL_STYLE = {100: dict(ls="-", lw=1.3, marker="o", ms=3.4, alpha=0.14), 10: dict(ls=(0, (3, 1.5)), lw=1.0, marker="^", ms=3.4, alpha=0.08)}


def series(df, model, T, ncol):
    return df[(df.model == model) & (df.Tfin == T) & (df.n_collocation == ncol)].sort_values("n")


def line(ax, x, y, color, st):
    ax.plot(x, y, color=color, ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=st["ms"], mec=color)


def mse_panel(ax, df, T, x_col):
    for model, (_, color) in MODELS.items():
        for ncol, st in NCOL_STYLE.items():
            g = series(df, model, T, ncol)
            ax.fill_between(g[x_col], g.mse_q25, g.mse_q75, color=color, alpha=st["alpha"], lw=0)
            line(ax, g[x_col], g.mse_median, color, st)
    ax.set_yscale("log")
    ax.set_ylabel(r"median MSE of $u(x,T)$")


RATIO_TICKS = [1, 1.5, 2]
RATIO_LIMITS = (0.9, 2.4)


def ratio_axis(ax):
    """Classical / HQPINN steps on a log axis, with the line of equal cost."""
    ax.axhline(1.0, color=MUTED, lw=0.7, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(*RATIO_LIMITS)
    ax.set_yticks(RATIO_TICKS, [f"{t:g}" for t in RATIO_TICKS])
    ax.set_yticks([], minor=True)
    ax.set_ylabel("steps, classical / HQPINN")


def steps_panel(ax, df, T):
    """One curve per N_col: mean steps of the classical PINN over those of the HQPINN at the same n (above 1: the
    HQPINN needs fewer steps). Runs that miss the target count at the budget, as in the means themselves."""
    ax.text(15.3, 0.985, "same number of steps", ha="right", va="top", fontsize=5.6, color=INK2)
    for ncol, st in NCOL_STYLE.items():
        c, q = series(df, "classical", T, ncol).set_index("n"), series(df, "hqpinn", T, ncol).set_index("n")
        line(ax, c.index, c.steps_mean / q.steps_mean, INK, st)
    ratio_axis(ax)
    ax.set_xlabel(r"width $n$ (qubits / neurons)")


def main():
    set_style()
    df = read_table("forward/simulation/width_sweep.csv")
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.9), constrained_layout=True)
    labels = iter("abcdef")
    for row, T in enumerate((0.1, 1.0)):
        ax_n, ax_steps, ax_params = axes[row]
        mse_panel(ax_n, df, T, "n")
        ax_n.set_xlabel(r"width $n$ (qubits / neurons)")
        steps_panel(ax_steps, df, T)
        mse_panel(ax_params, df, T, "params")
        ax_params.set_xscale("log")
        ax_params.set_xticks([100, 200, 400, 800], ["100", "200", "400", "800"])
        ax_params.set_xticks([], minor=True)
        ax_params.set_xlabel("trainable parameters")
        for ax in (ax_n, ax_steps):
            ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
        for ax in axes[row]:
            panel_label(ax, next(labels), x=-0.25)
        ax_n.text(0.97, 0.95, rf"$T={T:g}$", transform=ax_n.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    handles = [Line2D([], [], color=c, lw=1.3, label=lbl) for lbl, c in MODELS.values()]
    handles += [Line2D([], [], color=INK2, ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=st["ms"], mec=INK2,
                       label=rf"$N_{{\rm col}}={n}$") for n, st in NCOL_STYLE.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, columnspacing=1.6)
    save(fig, "fig2_sim_width")


if __name__ == "__main__":
    main()
