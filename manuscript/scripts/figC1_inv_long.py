"""Appendix C -- older long-budget sweeps of the inverse problem: mean squared error of nu vs collocation points.

Data: manuscript/data/inverse/long_budget_ncol.csv. Widths up to 10: n = 3, 7 (N_col = 10..100) and n = 10 (N_col = 10 and
100 only, drawn as isolated points). Colour: model (legend); marker: width (caption). The two models follow different
protocols (the older HQPINN has a linear CNOT circuit, a stopping target and 10 seeds): they are drawn together, not tested.
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

from paper_common import C_CLASSICAL, C_EXACT, COL2, INK2, SURFACE, panel_label, read_table, save, set_style

import matplotlib.pyplot as plt

MODELS = {"classical": (C_CLASSICAL, "Classical PINN"), "hqpinn_older": (C_EXACT, "HQPINN")}
MARKERS = {3: "o", 7: "D", 10: "^"}


def main():
    set_style()
    df = read_table("inverse/long_budget_ncol.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), constrained_layout=True)
    for ax, T, lab in zip(axes, (0.1, 1.0), "ab"):
        for model, (color, _) in MODELS.items():
            for n, marker in MARKERS.items():
                g = df[(df.model == model) & (df.Tfin == T) & (df["size"] == n)].sort_values("n_collocation")
                ax.plot(g.n_collocation, g.mse_nu, color=color, marker=marker, ms=5, mec=SURFACE, mew=0.5,
                        lw=1.1 if len(g) > 2 else 0)
        ax.set(yscale="log", xticks=range(10, 101, 10), xlabel=r"collocation points $N_{\rm col}$", ylabel=r"MSE of $\nu$")
        ax.text(0.97, 0.95, rf"$T={T:g}$", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
        panel_label(ax, lab, x=-0.14)
    handles = [Line2D([], [], color=c, lw=1.2, label=lbl) for c, lbl in MODELS.values()]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, columnspacing=1.8)
    save(fig, "figC1_inv_long_ncol")


if __name__ == "__main__":
    main()
