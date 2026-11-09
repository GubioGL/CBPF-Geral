"""Fig. 5 -- inverse problem, 20 000 epochs with the ideal circuit: the best run of each model and final time.

Data: manuscript/data/inverse/best_profile.csv and best_history.csv (data/Inverso).
  (a,d) u(x,T) and the exact solution; (b,e) learned nu per epoch (dashed: nu_true); (c,f) the two MAE terms of the
  training loss, residual (solid) and data (dashed). Histories: medians over logarithmic bins of epochs, log epoch axis.
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

from paper_common import C_CLASSICAL, C_EXACT, COL2, INK, INV_NU, SURFACE, panel_label, read_table, save, set_style

import matplotlib.pyplot as plt

MODELS = {"classical": (C_CLASSICAL, "o", "Classical PINN"), "hqpinn": (C_EXACT, "^", "HQPINN")}


def main():
    set_style()
    prof, hist = read_table("inverse/best_profile.csv"), read_table("inverse/best_history.csv")
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.9), constrained_layout=True)
    epochs = int(prof.epochs.max())
    labels = iter("abcdef")
    for row, T in enumerate((0.1, 1.0)):
        ax_u, ax_nu, ax_l = axes[row]
        exact = prof[(prof.Tfin == T) & (prof.model == "classical")]
        ax_u.plot(exact.x, exact.u_exact, color=INK, lw=1.3)
        for model, (color, marker, _) in MODELS.items():
            p = prof[(prof.Tfin == T) & (prof.model == model)]
            h = hist[(hist.Tfin == T) & (hist.model == model)]
            r = p.iloc[0]
            ax_u.plot(p.x[1::3], p.u[1::3], ls="none", marker=marker, ms=3.2, color=color, mec=SURFACE, mew=0.3,
                      label=f"n = {int(r['size'])}, seed {int(r.seed)}: " + rf"$\epsilon_{{L_2}}={r.l2:.3f}$")
            ax_nu.plot(h.epoch, h.nu, color=color, lw=1.2, label=rf"$\hat\nu={r.nu:.5f}$ ({100 * r.eps_nu:.2f}%)")
            ax_l.plot(h.epoch, h.loss_residual, color=color, lw=1.1, label="residual" if model == "classical" else None)
            ax_l.plot(h.epoch, h.loss_data, color=color, lw=1.1, ls=(0, (3, 1.5)), label="data" if model == "classical" else None)
        ax_nu.axhline(INV_NU, color=INK, ls=(0, (3, 1.5)), lw=0.8)
        lo, hi = exact.u_exact.min(), exact.u_exact.max()     # head room for the legend above the profile
        ax_u.set(xlim=(0, 1), ylim=(lo - 0.06 * (hi - lo), hi + 0.42 * (hi - lo)), xlabel=r"$x$", ylabel=rf"$u(x,T={T:g})$")
        ax_nu.set(xscale="log", yscale="log", ylim=(1e-4, 30), xlim=(1, epochs), xlabel="epoch", ylabel=r"learned $\nu$")
        ax_l.set(xscale="log", yscale="log", xlim=(1, epochs), xlabel="epoch", ylabel="training loss (MAE)")
        ax_u.legend(loc="upper right", fontsize=5.3, handlelength=1.0, frameon=True, framealpha=0.9, edgecolor="none")
        ax_nu.legend(loc="upper right", fontsize=5.3, handlelength=1.2, frameon=True, framealpha=0.9, edgecolor="none")
        ax_l.legend(loc="upper right", fontsize=5.3)
        for ax in axes[row]:
            panel_label(ax, next(labels), x=-0.22)
    handles = [Line2D([], [], color=c, lw=1.2, marker=m, ms=3.5, mec=SURFACE, label=lbl) for c, m, lbl in MODELS.values()]
    handles.append(Line2D([], [], color=INK, lw=1.3, label="exact solution"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3, columnspacing=1.8)
    save(fig, "fig6_inv_training")


if __name__ == "__main__":
    main()
