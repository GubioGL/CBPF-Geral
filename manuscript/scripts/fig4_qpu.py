"""Fig. 4 -- the HQPINN trained with the real QPU in the loop, one case per horizon.

Data: manuscript/data/forward/qpu/ (cases.csv and one folder per T). A case whose run does not exist yet is drawn as
empty, labelled panels (with the FDM reference where available), and fills in by itself once
build_paper_data.py finds its raw files.

  (a,c) u(x,T): FDM reference, initial network, classical PINN, HQPINN on the device;
  (b,d) MSE of u(x,T) vs epoch: HQPINN on the device (star: final network evaluated on the device),
        classical PINN and noiseless HQPINN with the same protocol.
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

from paper_common import (C_CLASSICAL, C_QPU, COL2, INK, INK2, MUTED, PAPER_DATA, SURFACE, TRIVIAL_MSE, panel_label,
                          read_table, save, set_style)

import matplotlib.pyplot as plt


def profile_panel(ax, folder, T):
    p = read_table(f"forward/qpu/{folder}/profile.csv")
    if "u_initial" in p:
        ax.plot(p.x, p.u_initial, color=MUTED, lw=0.8, ls=(0, (1, 1.2)), label="initial network")
    classical = [c for c in p.columns if c.startswith("u_classical")]
    if classical:
        ax.plot(p.x, p[classical[0]], color=C_CLASSICAL, lw=1.1, label="classical PINN")
    ax.plot(p.x, p.u_fdm, color=INK, lw=1.2, label="FDM reference", zorder=3)
    if "u_qpu" in p:
        ax.plot(p.x[2::5], p.u_qpu[2::5], ls="none", marker="o", ms=3.2, color=C_QPU, mec=SURFACE, mew=0.4,
                label="HQPINN on the QPU", zorder=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"$u(x,T={T:g})$")


def curve_panel(ax, folder, case, T):
    ax.axhline(TRIVIAL_MSE[T], color=MUTED, lw=0.7)
    ax.text(1, TRIVIAL_MSE[T] * 1.12, r"no evolution ($u=\sin 2\pi x$)", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=5.6, color=INK2)
    ax.set_yscale("log")
    ax.set_xlim(0, 112)
    ax.set_ylim(5e-5, 2)
    ax.set_xlabel("epoch (one QPU job per epoch)")
    ax.set_ylabel(r"MSE of $u(x,T)$")
    if not (PAPER_DATA / "forward" / "qpu" / folder / "learning_curve.csv").exists():
        return
    lc = read_table(f"forward/qpu/{folder}/learning_curve.csv")
    ax.fill_between(lc.epoch, lc.mse_classical_q25, lc.mse_classical_q75, color=C_CLASSICAL, alpha=0.10, lw=0)
    ax.plot(lc.epoch, lc.mse_classical_median, color=C_CLASSICAL, lw=1.1, label="classical PINN (median, IQR)")
    ax.plot(lc.epoch, lc.mse_noiseless_hqpinn_median, color=C_QPU, lw=0.9, ls=(0, (3, 1.5)),
            label="HQPINN, noiseless simulation (median)")
    ax.plot(lc.epoch, lc.mse_qpu, color=C_QPU, lw=1.3, label="HQPINN on the QPU")
    ax.plot([lc.epoch.iloc[-1]], [case.mse], marker="*", ms=8, color=C_QPU, mec=SURFACE, mew=0.6, ls="none", zorder=6,
            label="final network, evaluated on the QPU")


def pending_note(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=7, color=INK2,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=MUTED, lw=0.5))


def main():
    set_style()
    cases = read_table("forward/qpu/cases.csv")
    fig, axes = plt.subplots(len(cases), 2, figsize=(COL2, 2.0 * len(cases) + 0.2), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
    labels = iter("abcd")
    for row, case in enumerate(cases.itertuples()):
        folder, T = f"T_{case.Tfin:g}", case.Tfin
        ax_u, ax_c = axes[row]
        profile_panel(ax_u, folder, T)
        curve_panel(ax_c, folder, case, T)
        ax_u.text(0.03, 0.05, rf"$T={T:g}$", transform=ax_u.transAxes, ha="left", va="bottom", fontsize=7.5, color=INK2)
        panel_label(ax_u, next(labels), x=-0.19)
        panel_label(ax_c, next(labels), x=-0.15)
        if case.status == "pending":
            pending_note(ax_c, rf"QPU run at $T={T:g}$ pending")
        else:
            ax_u.legend(loc="upper right", fontsize=5.6)
            ax_c.legend(loc="lower left", fontsize=5.6, ncol=2, columnspacing=1.2)
    save(fig, "fig4_qpu")


if __name__ == "__main__":
    main()
