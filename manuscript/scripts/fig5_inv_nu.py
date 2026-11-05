"""Inverse problem, simulation: identification of nu over the 33 paired runs of each model, one row per final time.

Data: manuscript/data/inverse/ (nu_curves.csv, runs.csv). Writes two figures with the same layout:
  fig5_inv_nu_E500.pdf   500 epochs, HQPINN on FakeAachen (main text)
  figC1_inv_nu_E110.pdf  110 epochs, HQPINN on FakeBerlin (Appendix C)

  (a,d) learned nu vs epoch: median, band 10-90 %;
  (b,e) relative error of nu vs epoch: median, band interquartile range;
  (c,f) final relative error of every run, per size; horizontal ticks, medians per size.

Colour: blue = classical PINN, aqua = HQPINN on a noisy emulator. Aqua is below 3:1 on white (validate_palette.js), so
the models also differ by marker in panels (c,f) and the legend names them.
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

from paper_common import (C_CLASSICAL, C_EMU, COL2, INK, INK2, INV_NU, INV_SIZES, INVERSE_BUDGETS, MUTED, SURFACE, panel_label,
                          read_table, save, set_style)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODELS = {"classical": (C_CLASSICAL, "o", "Classical PINN"), "hqpinn": (C_EMU, "^", "HQPINN on {backend}")}


def curves(ax_nu, ax_eps, df, T, epochs):
    for model, (color, _, _) in MODELS.items():
        g = df[(df.epochs == epochs) & (df.Tfin == T) & (df.model == model)]
        ax_nu.fill_between(g.epoch, g.nu_p10, g.nu_p90, color=color, alpha=0.16, lw=0)
        ax_nu.plot(g.epoch, g.nu_median, color=color, lw=1.1)
        ax_eps.fill_between(g.epoch, g.eps_q25, g.eps_q75, color=color, alpha=0.18, lw=0)
        ax_eps.plot(g.epoch, g.eps_median, color=color, lw=1.1)
    ax_nu.axhline(INV_NU, color=INK, ls=(0, (3, 1.5)), lw=0.8, zorder=1)
    ax_nu.text(epochs, INV_NU * 1.25, r"$\nu_{\rm true}$", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax_nu.set(yscale="log", xlim=(1, epochs), ylim=(INV_NU / 20, 1.2), xlabel=r"epoch $k$", ylabel=r"learned $\nu_k$")
    ax_eps.axhline(0.05, color=MUTED, ls=":", lw=0.8, zorder=1)
    ax_eps.text(epochs, 0.055, "5%", ha="right", va="bottom", fontsize=5.8, color=INK2)
    ax_eps.set(yscale="log", xlim=(1, epochs), ylim=(3e-3, 120), xlabel=r"epoch $k$",
               ylabel=r"$|\nu_k-\nu_{\rm true}|/\nu_{\rm true}$")


def final_errors(ax, runs, T, epochs):
    rng = np.random.default_rng(0)
    for (model, (color, marker, _)), dx in zip(MODELS.items(), (-0.17, 0.17)):
        g = runs[(runs.epochs == epochs) & (runs.Tfin == T) & (runs.model == model)]
        for i, size in enumerate(INV_SIZES):
            e = 100 * g[g["size"] == size].eps_nu.values
            ax.scatter(i + dx + rng.uniform(-0.07, 0.07, len(e)), e, s=9, marker=marker, color=color, edgecolor=SURFACE,
                       lw=0.3, alpha=0.85, zorder=3)
            ax.plot([i + dx - 0.14, i + dx + 0.14], [np.median(e)] * 2, color=INK, lw=1.0, zorder=4)
    ax.set_xticks(range(len(INV_SIZES)), [str(s) for s in INV_SIZES])
    ax.set(xlabel=r"qubits $n$ / neurons $k$", ylabel=r"final $\epsilon_\nu$ (%)", ylim=(0, None))


def figure(curves_df, runs, epochs, name):
    backend = INVERSE_BUDGETS[epochs]["backend"]
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
    labels = iter("abcdef")
    for row, T in enumerate((0.1, 1.0)):
        ax_nu, ax_eps, ax_final = axes[row]
        curves(ax_nu, ax_eps, curves_df, T, epochs)
        final_errors(ax_final, runs, T, epochs)
        ax_nu.text(0.97, 0.95, rf"$T={T:g}$", transform=ax_nu.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
        for ax in axes[row]:
            panel_label(ax, next(labels), x=-0.23)
    handles = [Line2D([], [], color=c, lw=1.2, marker=m, ms=3.2, mec=SURFACE, mew=0.3, label=lbl.format(backend=backend))
               for c, m, lbl in MODELS.values()]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, columnspacing=1.6)
    save(fig, name)


def main():
    set_style()
    df, runs = read_table("inverse/nu_curves.csv"), read_table("inverse/runs.csv")
    figure(df, runs, 500, "fig5_inv_nu_E500")
    figure(df, runs, 110, "figC1_inv_nu_E110")


if __name__ == "__main__":
    main()
