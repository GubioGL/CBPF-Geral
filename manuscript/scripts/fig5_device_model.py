"""Fig. 5 -- how well the calibrated trigonometric surrogate describes the device.

  (a) residual <Z>_measured - surrogate of the calibration fit (1024 random angle sets x 10
      outputs), ibm_berlin vs FakeBerlin, against the residual expected from 1024-shot
      noise alone (Gaussian mixture with variance (1 - y^2)/S);
  (b) contrast: coefficient of the ideal term prod_{j in S_i} cos(phi_j) per output qubit
      (1 on a noiseless device);
  (c) misfit of every chain to the chip-average surrogate (chain screening; dropped above
      3 x the median);
  (d) RMS of the QPU correction Delta = <Z>_QPU - surrogate at the stencil centres, per training
      epoch (surrogate in force at that epoch; refits every 10 epochs), and at validation
      (each point averaged over the 8 chains).
"""
import numpy as np

from paper_common import (C_FAKE, C_QPU, CACHE_DIR, COL2, INK2, MUTED, SURFACE, fake_file, load_json,
                          load_npz, panel_label, real_file, save, set_style)

import matplotlib.pyplot as plt

import run_quantum_hardware as rqh  # noqa: E402
from Inverser_quantum_hardware_pinn import (  # noqa: E402
    causal_cones, cnot_gate_list, cone_features, fit_surrogate,
)
from main.functions import repeated_cnot_network_matrix  # noqa: E402

N, SHOTS = 10, 1024
DEVICES = (("real", "ibm_berlin", C_QPU, real_file), ("fake", "FakeBerlin", C_FAKE, fake_file))


def calibration_fit(file):
    cal = load_npz(file("calibration"))
    gates = cnot_gate_list(N, rqh.FIXED_LAYER, rqh.FIXED_LAYER_REPEATS)
    M = repeated_cnot_network_matrix(N, rqh.FIXED_LAYER, rqh.FIXED_LAYER_REPEATS)
    cones = causal_cones(N, gates)
    coefs, rms, contrast = fit_surrogate(cal["angles"], cal["evs"], cones, M)
    pred = np.stack([cone_features(cal["angles"], cone) @ c for cone, c in zip(cones, coefs)], axis=1)
    return cal, pred, np.array(contrast)


def shot_noise_pdf(grid, y):
    sig = np.sqrt(np.clip(1.0 - y.ravel() ** 2, 1e-4, None) / SHOTS)
    pdf = np.zeros_like(grid)
    for chunk in np.array_split(sig, 20):
        pdf += np.exp(-0.5 * (grid[:, None] / chunk[None]) ** 2).__truediv__(np.sqrt(2 * np.pi) * chunk[None]).sum(1)
    return pdf / sig.size


def main():
    set_style()
    lc = load_npz(CACHE_DIR / "learning_curves.npz")
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.95), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1, 1.35]))
    ax_a, ax_b, ax_c, ax_d = axes
    grid = np.linspace(-0.2, 0.2, 401)
    bins = np.linspace(-0.2, 0.2, 61)
    floor = {}
    for kind, name, color, file in DEVICES:
        cal, pred, contrast = calibration_fit(file)
        res = (cal["evs"] - pred).ravel()
        ax_a.hist(res, bins=bins, density=True, histtype="step", color=color, lw=1.1,
                  label=f"{name} (RMS {np.sqrt(np.mean(res ** 2)):.3f})")
        if kind == "real":
            ax_a.plot(grid, shot_noise_pdf(grid, pred), color=INK2, lw=0.8, ls=(0, (3, 1.5)),
                      label=f"shot noise only (RMS {np.sqrt(np.mean((1 - pred ** 2) / SHOTS)):.3f})")
        ax_b.plot(np.arange(1, N + 1), contrast, marker="o", ms=3.4, color=color, lw=0.9, mec=SURFACE, mew=0.5,
                  label=name)
        run = load_json(file("run", "json"))
        misfit = np.array(run["chain_misfit"])
        ax_c.plot(np.arange(1, len(misfit) + 1), misfit, marker="o", ms=3.4, color=color, lw=0.9, mec=SURFACE,
                  mew=0.5, label=name)
        ax_c.axhline(3 * np.median(misfit), color=color, lw=0.6, ls=(0, (1, 1.5)))
        floor[kind] = np.sqrt(np.mean((1 - pred ** 2) / SHOTS))
        ep = np.arange(1, len(lc[f"{kind}_rms_qpu_vs_poly"]) + 1)
        ax_d.plot(ep, lc[f"{kind}_rms_qpu_vs_poly"], color=color, lw=1.0, label=f"{name}, training centres")
    # (a)
    ax_a.set_yscale("log")
    ax_a.set_ylim(5e-2, 400)
    ax_a.set_xlim(-0.2, 0.2)
    ax_a.set_xlabel(r"$\langle Z\rangle_{\rm meas}-\hat f(\boldsymbol{\theta})$ (calibration)")
    ax_a.set_ylabel("density")
    ax_a.legend(loc="upper left", fontsize=4.9, handlelength=1.2, borderaxespad=0.1)
    panel_label(ax_a, "a", x=-0.3)
    # (b)
    ax_b.axhline(1.0, color=MUTED, lw=0.6)
    ax_b.set_xticks(range(1, N + 1, 3))
    ax_b.set_xlabel(r"output qubit $q$")
    ax_b.set_ylabel(r"contrast $c^{(q)}_{\rm ideal}$")
    ax_b.set_ylim(0.84, 1.01)
    ax_b.legend(loc="lower left", fontsize=5.4)
    panel_label(ax_b, "b", x=-0.33)
    # (c)
    ax_c.axhline(floor["real"], color=MUTED, lw=0.6)
    ax_c.text(8.4, floor["real"] * 0.93, "shot noise", fontsize=5.2, color=INK2, ha="right", va="top")
    ax_c.set_xticks(range(1, 9))
    ax_c.set_xlabel("chain")
    ax_c.set_ylabel("calibration misfit (RMS)")
    ax_c.set_ylim(0.02, 0.11)
    ax_c.text(0.03, 0.97, "dotted: 3 x median (drop)", transform=ax_c.transAxes, fontsize=5.2, color=INK2, va="top")
    panel_label(ax_c, "c", x=-0.33)
    # (d)
    for e in range(10, 111, 10):
        ax_d.axvline(e, color="#efeeea", lw=0.6, zorder=0)
    ax_d.axhline(1 / 32, color=MUTED, lw=0.6)
    ax_d.text(118, 1 / 32 * 1.04, r"$1/\sqrt{S}$", fontsize=5.4, color=INK2, va="bottom", ha="right")
    for kind, name, color, file in DEVICES:
        run = load_json(file("run", "json"))
        v = run["rms_q_qpu_vs_surrogate"] if kind == "real" else None
        if v is not None:
            ax_d.plot([116], [v], marker="D", ms=3.6, color=color, mec=SURFACE, mew=0.5, ls="none",
                      label=f"{name}, validation (8-chain mean)")
    ax_d.axhline(1 / 32 / np.sqrt(8), color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax_d.set_xlim(0, 120)
    ax_d.set_ylim(0, 0.115)
    ax_d.set_xlabel("epoch")
    ax_d.set_ylabel(r"RMS of $\Delta=\langle Z\rangle_{\rm QPU}-\hat f$")
    ax_d.legend(loc="upper left", fontsize=5.2, borderaxespad=0.1)
    panel_label(ax_d, "d", x=-0.25)
    save(fig, "fig5_device_model")


if __name__ == "__main__":
    main()
