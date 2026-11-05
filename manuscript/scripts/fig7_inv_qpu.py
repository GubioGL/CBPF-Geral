"""Inverse problem, real QPU: learned nu and u(x,T), one case per final time.

Data: manuscript/data/inverse/ (qpu_cases.csv, nu_curves.csv, field.csv and qpu/T_<T>/ once a run exists). NO real-QPU run
of the inverse problem exists yet: the panels show only the emulator baselines the run will be compared with, and a
"pending" label. They fill in by themselves when build_inverse_data.py finds the raw files of a case in INVERSE_QPU_CASES.

  (a,c) learned nu vs epoch: classical PINN and HQPINN on FakeBerlin at 110 epochs (median over 33 runs); real QPU, if present;
  (b,d) u(x,T): exact solution; real QPU, if present.
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

from paper_common import (C_CLASSICAL, C_EMU, C_QPU, COL2, INK, INK2, INV_NU, MUTED, PAPER_DATA, SURFACE, panel_label,
                          read_table, save, set_style)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EPOCHS = 110


def pending_note(ax, text, x, y):
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center", fontsize=7, color=INK2,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=MUTED, lw=0.5), zorder=10)


def main():
    set_style()
    cases, curves, field = (read_table(f"inverse/{name}.csv") for name in ("qpu_cases", "nu_curves", "field"))
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 3.5), constrained_layout=True)
    labels = iter("abcd")
    for row, case in enumerate(cases.itertuples()):
        T, folder = case.Tfin, f"T_{case.Tfin:g}"
        ax_nu, ax_u = axes[row]
        for model, color in (("classical", C_CLASSICAL), ("hqpinn", C_EMU)):
            g = curves[(curves.epochs == EPOCHS) & (curves.Tfin == T) & (curves.model == model)]
            ax_nu.plot(g.epoch, g.nu_median, color=color, lw=1.0)
        ax_nu.axhline(INV_NU, color=INK, ls=(0, (3, 1.5)), lw=0.8, zorder=1)
        ax_nu.set(yscale="log", xlim=(1, EPOCHS), ylim=(INV_NU / 20, 1.2), xlabel=r"epoch $k$", ylabel=r"learned $\nu_k$")
        g = field[(field.epochs == EPOCHS) & (field.Tfin == T)]
        ax_u.plot(g.x, g.u_exact, color=INK, lw=1.2, zorder=3)
        ax_u.set(xlim=(0, 1), xlabel=r"$x$", ylabel=rf"$u(x,T={T:g})$")
        if case.status == "done":
            n, p = read_table(f"inverse/qpu/{folder}/nu.csv"), read_table(f"inverse/qpu/{folder}/profile.csv")
            ax_nu.plot(n.epoch, n.nu, color=C_QPU, lw=1.2, zorder=5)
            ax_u.plot(p.x[1::4], p.u_qpu[1::4], ls="none", marker="o", ms=3.0, color=C_QPU, mec=SURFACE, mew=0.3, zorder=5)
        else:
            pending_note(ax_nu, rf"QPU run at $T={T:g}$ pending", 0.55, 0.82)
            pending_note(ax_u, rf"QPU run at $T={T:g}$ pending", 0.74, 0.82)
        panel_label(ax_nu, next(labels), x=-0.13)
        panel_label(ax_u, next(labels), x=-0.13)
    handles = [Line2D([], [], color=C_CLASSICAL, lw=1.0, label="classical PINN (median)"),
               Line2D([], [], color=C_EMU, lw=1.0, label="HQPINN on FakeBerlin (median)"),
               Line2D([], [], color=C_QPU, lw=1.2, marker="o", ms=3, mec=SURFACE, mew=0.3, label="HQPINN on the QPU"),
               Line2D([], [], color=INK, lw=1.2, label=r"exact solution; $\nu_{\rm true}$ dashed")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, columnspacing=1.4)
    save(fig, "fig7_inv_qpu")


if __name__ == "__main__":
    main()
