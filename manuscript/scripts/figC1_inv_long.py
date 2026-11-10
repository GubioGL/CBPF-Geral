"""Fig. 8 (Appendix C) -- inverse problem, error versus the size n, in the form of Figs. 2(a,d).

Data: manuscript/data/inverse/size_errors.csv. Rows: T = 0.1 (top) and T = 1 (bottom).
  (a,c) MSE of nu, the mean over the seeds of (nu - nu_true)^2;
  (b,d) median MSE of u(x,T) over the seeds (band: interquartile range), as in Figs. 2(a,d).
Only the 20 000-epoch runs, classical PINN and HQPINN with the ideal circuit (11 seeds per size; the sizes on the axis are
those found in the data). Colour and marker: model; legend with the two models only. There is no N_col counterpart of
Fig. 3: every inverse run uses N_col = 10.
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

from paper_common import C_CLASSICAL, C_EXACT, COL2, INK2, SURFACE, panel_label, read_table, save, set_style

import matplotlib.pyplot as plt

MODELS = {"classical": (C_CLASSICAL, "o", "Classical PINN"), "hqpinn": (C_EXACT, "^", "HQPINN")}
LONG = 20000


def decades(values):
    return 10 ** np.floor(np.log10(np.min(values))), 10 ** np.ceil(np.log10(np.max(values)))


def main():
    set_style()
    df = read_table("inverse/size_errors.csv")
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 3.9), constrained_layout=True)
    labels = iter("abcd")
    for row, T in enumerate((0.1, 1.0)):
        ax_nu, ax_u = axes[row]
        d = df[(df.Tfin == T) & (df.epochs == LONG)]
        for model, (color, marker, _) in MODELS.items():
            g = d[d.model == model].sort_values("size")
            ax_nu.plot(g["size"], g.mse_nu, color=color, marker=marker, ms=5, mec=SURFACE, mew=0.5, lw=1.1)
            ax_u.fill_between(g["size"], g.mse_u_q25, g.mse_u_q75, color=color, alpha=0.15, lw=0)
            ax_u.plot(g["size"], g.mse_u_median, color=color, marker=marker, ms=5, mec=SURFACE, mew=0.5, lw=1.1)
        ax_nu.set(yscale="log", ylim=decades(d.mse_nu), ylabel=r"MSE of $\nu$")
        ax_u.set(yscale="log", ylim=decades(np.r_[d.mse_u_q25, d.mse_u_q75]), ylabel=r"median MSE of $u(x,T)$")
        for ax in (ax_nu, ax_u):
            sizes = sorted(d["size"].unique())
            ax.set(xticks=sizes, xlim=(sizes[0] - 1, sizes[-1] + 1), xlabel=r"width $n$ (qubits / neurons)")
            ax.text(0.03, 0.95, rf"$T={T:g}$", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK2)
            panel_label(ax, next(labels), x=-0.14)
    handles = [Line2D([], [], color=c, marker=m, ms=5, mec=SURFACE, lw=1.2, label=lbl) for c, m, lbl in MODELS.values()]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, columnspacing=1.8)
    save(fig, "figC1_inv_long")


if __name__ == "__main__":
    main()
