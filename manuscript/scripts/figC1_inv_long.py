"""Appendix C -- inverse problem, 20 000 epochs with the ideal circuit: mean squared error of nu versus the size.

Data: manuscript/data/inverse/mse_size.csv (data/Inverso). MSE of nu = mean over the 11 seeds of (nu - nu_true)^2, per
model, final time and size n = k = 3, 7, 10. Same training, CONFIGs and adjustments as the 110/500-epoch runs; the
HQPINN with the noiseless layer. Legend: the two models only.
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

from paper_common import C_CLASSICAL, C_EXACT, COL2, INK2, INV_SIZES, SURFACE, panel_label, read_table, save, set_style

import matplotlib.pyplot as plt

MODELS = {"classical": (C_CLASSICAL, "o", "Classical PINN"), "hqpinn": (C_EXACT, "^", "HQPINN")}


def main():
    set_style()
    df = read_table("inverse/mse_size.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), constrained_layout=True)
    for ax, T, lab in zip(axes, (0.1, 1.0), "ab"):
        for model, (color, marker, _) in MODELS.items():
            g = df[(df.model == model) & (df.Tfin == T)].sort_values("size")
            ax.plot(g["size"], g.mse_nu, color=color, marker=marker, ms=5, mec=SURFACE, mew=0.5, lw=1.1)
        span = df[df.Tfin == T].mse_nu
        ax.set(yscale="log", ylim=(10 ** np.floor(np.log10(span.min())), 10 ** np.ceil(np.log10(span.max()))),
               xticks=INV_SIZES, xlim=(2, 11), xlabel=r"neurons or qubits $n$", ylabel=r"MSE of $\nu$")
        ax.text(0.97, 0.95, rf"$T={T:g}$", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
        panel_label(ax, lab, x=-0.14)
    handles = [Line2D([], [], color=c, marker=m, ms=5, mec=SURFACE, lw=1.2, label=lbl) for c, m, lbl in MODELS.values()]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, columnspacing=1.8)
    save(fig, "figC1_inv_long")


if __name__ == "__main__":
    main()
