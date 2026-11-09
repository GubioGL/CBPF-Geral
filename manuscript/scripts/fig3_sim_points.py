"""Fig. 3 -- simulations: growing the number of collocation points, noiseless HQPINN vs classical PINN.

Data: manuscript/data/forward/simulation/points_sweep.csv (same protocol, 50 seeds per point). Both models cover N_col = 10..100 at the five
widths; a width that ever had only the two ends of the range would be drawn as isolated points, without a line.

Encoding: black = classical PINN, one colour per width = HQPINN, one marker per width in both models. Blue is left out on
purpose: it means "classical" in the other figures. The five colours pass validate_palette.js on all pairs (CVD and normal
vision); yellow and aqua are below 3:1 on white, which the markers and the legend make up for.

Rows: T = 0.1 (top) and T = 1 (bottom).
  (a,c) median MSE of u(x,T) vs N_col;
  (b,d) mean optimizer steps of the classical PINN divided by those of the HQPINN, vs N_col, one curve per width, all in one colour,
        told apart by the marker (log axis; above 1, the HQPINN needs fewer steps); a run that misses the target counts
        at the budget.
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

from matplotlib.lines import Line2D

from fig2_sim_width import ratio_axis
from paper_common import COL2, INK, INK2, MUTED, panel_label, read_table, save, set_style

import matplotlib.pyplot as plt

WIDTHS = {3: "#eda100", 5: "#dc5620", 7: "#ad2c6c", 10: "#6250d6", 15: "#1baf7a"}   # HQPINN colour of each width
MARKERS = {3: "o", 5: "s", 7: "D", 10: "^", 15: "v"}
MARKER_SIZE = {3: 4.6, 5: 4.1, 7: 3.8, 10: 4.6, 15: 4.6}   # same visual area across shapes
C_RATIO = "#7b4b2a"   # (b,d): one colour, none of the colours above, so the ratios are not read as HQPINN curves; the marker gives n
XTICKS = list(range(10, 101, 10))


def has_curve(df, model, n):
    """A width gets a line only if it has more N_col values than the two ends of the range."""
    return df[(df.model == model) & (df.n == n)].n_collocation.nunique() > 2


def style(df, model, n):
    """Line and marker of one curve, shared by the panels and the legend."""
    return dict(color=INK if model == "classical" else WIDTHS[n], lw=1.0 if has_curve(df, model, n) else 0,
                marker=MARKERS[n], ms=MARKER_SIZE[n], mec="white", mew=0.4)


def draw(ax, df, T, column, scale=1.0):
    """One curve per width and model; a width with two N_col values is left as isolated points."""
    for model, zorder in (("hqpinn", 3), ("classical", 4)):
        for n in WIDTHS:
            g = df[(df.model == model) & (df.Tfin == T) & (df.n == n)].sort_values("n_collocation")
            ax.plot(g.n_collocation, g[column] / scale, zorder=zorder, **style(df, model, n))


def draw_ratio(ax, df, T):
    """One curve per width, all in C_RATIO, told apart by the marker: classical over HQPINN mean steps vs N_col."""
    for n in WIDTHS:
        c = df[(df.model == "classical") & (df.Tfin == T) & (df.n == n)].set_index("n_collocation").steps_mean
        q = df[(df.model == "hqpinn") & (df.Tfin == T) & (df.n == n)].set_index("n_collocation").steps_mean
        r = (c / q).dropna().sort_index()
        ax.plot(r.index, r.values, color=C_RATIO, lw=1.0, marker=MARKERS[n], ms=MARKER_SIZE[n] * 1.15, mec="white", mew=0.5,
                label=rf"$n={n}$")


def main():
    set_style()
    df = read_table("forward/simulation/points_sweep.csv")
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 3.9), constrained_layout=True)
    labels = iter("abcd")
    for row, T in enumerate((0.1, 1.0)):
        ax_mse, ax_steps = axes[row]
        draw(ax_mse, df, T, "mse_median")
        ax_mse.set_yscale("log")
        ax_mse.set_ylabel(r"median MSE of $u(x,T)$")
        ax_steps.text(100, 0.985, "same number of steps", ha="right", va="top", fontsize=5.6, color=INK2)
        draw_ratio(ax_steps, df, T)
        ratio_axis(ax_steps)
        for ax in axes[row]:
            ax.set_xticks(XTICKS)
            ax.set_xlabel(r"collocation points $N_{\rm col}$")
            panel_label(ax, next(labels), x=-0.14)
        ax_mse.text(0.97, 0.95, rf"$T={T:g}$", transform=ax_mse.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    handles = []       # the legend fills column by column: one column per width, HQPINN above classical
    for n in WIDTHS:
        handles.append(Line2D([], [], label=rf"HQPINN $n={n}$", **style(df, "hqpinn", n)))
        handles.append(Line2D([], [], label=rf"classical $n={n}$", **style(df, "classical", n)))
    fig.legend(handles=handles, loc="outside upper center", ncol=5, columnspacing=1.4)
    axes[1][1].legend(loc="upper left", ncol=5, columnspacing=0.9, handlelength=1.4, fontsize=6,
                      title="classical / HQPINN steps", title_fontsize=6)
    save(fig, "fig3_sim_points")


if __name__ == "__main__":
    main()
