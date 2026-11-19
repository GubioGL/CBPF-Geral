"""Inverse problem on the real QPU: the HQPINN trained with ibm_aachen in the loop against the same run (seed 1, n = 10,
T = 0.1, 200 epochs, same training) with the quantum layer on FakeAachen, with the ideal (noiseless) circuit, and the
classical PINN with the same seed. Colours as in the forward problem (Fig. 4): orange = HQPINN on the QPU (solid,
markers) and HQPINN with the ideal circuit (dashed), aqua = HQPINN on the noisy emulator, blue = classical PINN,
black = exact solution.

  (a) u(x, T): on the QPU the trained network evaluated with the QPU in the loop (validation job, 30 points);
      FakeAachen the same on its emulator (100 points); ideal and classical their own prediction;
  (b) learned nu per epoch (dashed black: nu_true = 0.01);
  (c) residual loss and (d) data loss per epoch (MAE; on the QPU and on FakeAachen computed with the measured values).

    python run/plot_inverse_qpu_real.py        # -> data/Inverso_real/ibm_aachen/inverse_qpu_T0.1_seed1.pdf (and .png)
Data: data/Inverso_real/ibm_aachen (QPU), data/Inverso_200/quantico (FakeAachen), data/Inverso_200/quantico_ideal (ideal,
run on 18/11/2026 with run/Inverser_runs_paralelo.run_job, the training of the 200-epoch runs), data/Inverso_200/classico.
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _ROOT / "src" / "quantum_hardware", _HERE, _ROOT / "manuscript" / "scripts"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper_common import C_CLASSICAL, C_EMU, C_QPU, COL2, INK, INK2, SURFACE, panel_label, set_style

SEED, T, EPOCHS, NU = 1, 0.1, 200, 0.01
CASE = f"T_{T}_nu_{NU:g}"
OUT = os.path.join("data", "Inverso_real", "ibm_aachen")
# key: (folder, file prefix, field in validation_*.npz, colour, line style, marker, label)
RUNS = {
    "qpu": (os.path.join(OUT, "quantico", CASE), "NQ", "u_qpu", C_QPU, "-", "o", "HQPINN on the QPU (ibm_aachen)"),
    "fake": (os.path.join("data", "Inverso_200", "quantico", CASE), "NQ", "u_qpu", C_EMU, "-", "^", "HQPINN, FakeAachen"),
    "ideal": (os.path.join("data", "Inverso_200", "quantico_ideal", CASE), "NQ", "u_pred", C_QPU, (0, (3, 1.5)), None,
              "HQPINN, ideal circuit"),
    "classical": (os.path.join("data", "Inverso_200", "classico", CASE), "NE", "u_pred", C_CLASSICAL, "-", "s",
                  "classical PINN"),
}


def load(key):
    folder, prefix, field = RUNS[key][:3]
    tag = f"10_{prefix}10_{SEED}_E{EPOCHS}"
    run = json.load(open(os.path.join(folder, f"run_{tag}.json")))
    hist = np.load(os.path.join(folder, f"losshist_{tag}.npz"))
    val = np.load(os.path.join(folder, f"validation_{tag}.npz"))
    l2 = run.get("rel_l2_T_qpu", run.get("rel_l2_T"))
    return dict(run=run, hist=hist, x=val["x"].ravel(), u=val[field].ravel(), u_exact=val["u_exact"].ravel(), l2=l2)


def main():
    set_style()
    data = {k: load(k) for k in RUNS}
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.6), constrained_layout=True)
    ax_u, ax_nu, ax_r, ax_d = axes.ravel()
    epoch = np.arange(1, EPOCHS + 1)

    ref = data["classical"]
    ax_u.plot(ref["x"], ref["u_exact"], color=INK, lw=1.2, zorder=1)
    for key, (_, _, _, color, ls, marker, label) in RUNS.items():
        d = data[key]
        if marker is None:
            ax_u.plot(d["x"], d["u"], color=color, lw=1.1, ls=ls, zorder=2)
        else:
            step = 1 if key == "qpu" else 4
            ax_u.plot(d["x"][::step], d["u"][::step], ls="none", marker=marker, ms=3.6 if key == "qpu" else 3.0,
                      color=color, mec=SURFACE, mew=0.4, zorder=5 if key == "qpu" else 3)
        h = d["hist"]
        width = 1.3 if key == "qpu" else 0.9
        ax_nu.plot(epoch, h["nu"], color=color, ls=ls, lw=width, zorder=4 if key == "qpu" else 2)
        ax_r.plot(epoch, h["loss_r"], color=color, ls=ls, lw=width, zorder=4 if key == "qpu" else 2)
        ax_d.plot(epoch, h["loss_d"], color=color, ls=ls, lw=width, zorder=4 if key == "qpu" else 2)
    ax_nu.axhline(NU, color=INK, ls=(0, (3, 1.5)), lw=0.8)

    lo, hi = ref["u_exact"].min(), ref["u_exact"].max()
    ax_u.set(xlim=(0, 1), ylim=(lo - 0.06 * (hi - lo), hi + 0.70 * (hi - lo)), xlabel=r"$x$", ylabel=rf"$u(x,T={T:g})$")
    ax_nu.set(yscale="log", ylim=(1e-4, 1), xlim=(1, EPOCHS), xlabel="epoch", ylabel=r"learned $\nu$")
    ax_r.set(yscale="log", xlim=(1, EPOCHS), xlabel="epoch", ylabel="residual loss (MAE)")
    ax_d.set(yscale="log", xlim=(1, EPOCHS), xlabel="epoch", ylabel="data loss (MAE)")
    ax_nu.text(0.97, 0.95, r"$\nu_{\rm true}=0.01$", transform=ax_nu.transAxes, ha="right", va="top", fontsize=6.5,
               color=INK2)
    for ax, label in zip(axes.ravel(), "abcd"):
        panel_label(ax, label, x=-0.17)

    handles = [Line2D([], [], color=c, ls=ls, lw=1.2, marker=m, ms=3.5, mec=SURFACE, label=lbl)
               for _, _, _, c, ls, m, lbl in RUNS.values()]
    handles.append(Line2D([], [], color=INK, lw=1.2, label="exact solution"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3, columnspacing=1.4, fontsize=6.8)
    # per run: learned nu and error of u(x,T), in the empty upper right corner of (a)
    short = {"qpu": "QPU", "fake": "FakeAachen", "ideal": "ideal circuit", "classical": "classical"}
    lines = [f"{short[k]}: " + rf"$\hat\nu$ = {data[k]['run']['nu']:.5f} ({100 * abs(data[k]['run']['nu'] - NU) / NU:.1f}%), "
             + rf"$\epsilon_{{L_2}}$ = {data[k]['l2']:.4f}" for k in RUNS]
    ax_u.legend(handles=[Line2D([], [], color=RUNS[k][3], ls=RUNS[k][4], marker=RUNS[k][5], ms=3, lw=1, mec=SURFACE)
                         for k in RUNS], labels=lines, loc="upper center", fontsize=5.6, handlelength=1.8, frameon=False,
                borderaxespad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"inverse_qpu_T{T:g}_seed{SEED}.{ext}"), dpi=200)
    print(f"-> {OUT}/inverse_qpu_T{T:g}_seed{SEED}.pdf / .png")
    for k in RUNS:
        r = data[k]["run"]
        print(f"  {RUNS[k][6]:32s} nu = {r['nu']:.5f} ({100 * abs(r['nu'] - NU) / NU:5.2f}%)  eps_L2 = {data[k]['l2']:.4f}")


if __name__ == "__main__":
    main()
