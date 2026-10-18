"""Fig. 3 -- training the HQPINN with the ibm_berlin QPU in the loop (10 qubits x 8 chains).

  (a) ibm_berlin (120-qubit square lattice): the 8 disjoint 10-qubit chains run in parallel;
  (b) u(x, T = 0.1): FDM reference, HQPINN trained on ibm_berlin and evaluated on it,
      HQPINN trained on FakeBerlin and evaluated on ibm_berlin, classical PINN trained with the
      same protocol (n = 10, seed 0), and the initial network;
  (c) pointwise error u - u_FDM of the hardware evaluations; band: spread (min-max) of the
      network output over the 8 chains (each validation point was measured once on every chain);
  (d) MSE of u(x, T) vs epoch. Hardware/emulator curves are the bit-exact offline replay of the
      recorded runs (replay_training.py), evaluated with the calibrated surrogate in force at that
      epoch; stars: final network evaluated on the device (validation job). Classical: median and
      interquartile range over widths 1-10 x seeds 0-4; HQPINN ideal circuit: median of seeds 0-4
      (simulate_same_budget.py).

Needs manuscript/cache/learning_curves.npz and same_budget_sims.npz (run make_all.py).
"""
import numpy as np
import torch as tc
from matplotlib.lines import Line2D

from paper_common import (C_CLASSICAL, C_FAKE, C_QPU, CACHE_DIR, COL2, CVQ_DIR, GRID, INK, INK2, MUTED,
                          SURFACE, TRIVIAL_MSE, fake_file, load_json, load_npz, panel_label, real_file,
                          save, set_style)

import matplotlib.pyplot as plt

import run_quantum_hardware as rqh  # noqa: E402
from Inverser_quantum_hardware_pinn import point_grid  # noqa: E402

ROWS, COLS = 12, 10          # ibm_berlin: qubit q sits at row q // 10, column q % 10
C_QPU_TINT = "#f3a582"       # lighter step of the orange series hue (alternate chains in the map)


def chip_map(ax):
    run = load_json(real_file("run", "json"))
    snap = load_json(real_file("backend", "json"))
    pos = lambda q: (q % COLS, -(q // COLS))
    for q in range(ROWS * COLS):           # couplers of the full lattice (= FakeBerlin coupling map)
        r, c = divmod(q, COLS)
        if c + 1 < COLS:
            ax.plot(*zip(pos(q), pos(q + 1)), color=GRID, lw=0.8, zorder=1)
        if r + 1 < ROWS:
            ax.plot(*zip(pos(q), pos(q + COLS)), color=GRID, lw=0.8, zorder=1)
    used = {q for chain in run["chains"] for q in chain}
    idle = np.array([pos(q) for q in range(ROWS * COLS) if q not in used])
    ax.scatter(idle[:, 0], idle[:, 1], s=6, color="#d4d3cf", zorder=3, lw=0)
    # neighbouring chains alternate between the series hue and its tint, and carry their index
    for k, chain in enumerate(run["chains"]):
        xy = np.array([pos(q) for q in chain])
        color = C_QPU if k % 2 == 0 else C_QPU_TINT
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.6, zorder=2, solid_capstyle="round")
        ax.scatter(xy[:, 0], xy[:, 1], s=9, color=color, zorder=4, edgecolor=SURFACE, lw=0.5)
        mid = xy[len(xy) // 2 - 1:len(xy) // 2 + 1].mean(0)
        ax.text(*mid, str(k + 1), fontsize=4.8, color=INK, ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.12", fc=SURFACE, ec="none"))
    q = snap["qubits"]
    med = lambda key, scale=1.0: np.median([v[key] for v in q.values() if v[key] is not None]) * scale
    cz = np.median(list(snap["couplers"].values()))
    ax.text(0.5, -0.03, f"8 chains x 10 qubits (80/120)\n"
            rf"median $T_1$ {med('t1', 1e6):.0f} $\mu$s, $T_2$ {med('t2', 1e6):.0f} $\mu$s" "\n"
            f"readout {100 * med('readout_error'):.1f}%, CZ {1e3 * cz:.1f}" + r"$\times10^{-3}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=INK2, linespacing=1.3)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, COLS - 0.4)
    ax.set_ylim(-ROWS + 0.4, 0.6)
    ax.axis("off")
    ax.set_title(r"ibm_berlin", fontsize=7, color=INK, pad=2)


def per_chain_u(kind):
    """u(x, T) of the trained network from each chain's measurement of every validation point."""
    if kind == "real":
        val, model_file = load_npz(real_file("validation")), real_file("model", "pt")
        q_key = "q_qpu"
    else:
        val, model_file = load_npz(fake_file("prediction_real")), fake_file("model", "pt")
        q_key = "q_real"
    saved = tc.load(model_file, weights_only=False)
    model = rqh.build_model(10, 0.1)
    model.load_state_dict(saved["state_dict"], strict=False)     # quantum.* buffers are not needed
    x = val["x"]
    P, R = len(x), int(val["replicas"])
    B = R                                                          # one replica on every chain
    grid = point_grid(P, B, R, shift=0)
    point = grid[grid >= 0]
    xt = tc.tensor(np.hstack([x, np.full_like(x, 0.1)]), dtype=tc.get_default_dtype())
    u_chain = np.zeros((R, P))
    filled = np.zeros(P, int)
    with tc.no_grad():
        for p, q in zip(point, val["slot_evs"]):
            u_chain[filled[p], p] = model.from_features(xt[p:p + 1], q[None]).item()
            filled[p] += 1
        u_mean = model.from_features(xt, val[q_key]).numpy().ravel()
    return x.ravel(), u_mean, u_chain


def main():
    set_style()
    lc = load_npz(CACHE_DIR / "learning_curves.npz")
    sims = load_npz(CACHE_DIR / "same_budget_sims.npz")
    real_val = load_npz(real_file("validation"))
    fake_pred = load_npz(fake_file("prediction_real"))
    cls = load_npz(CVQ_DIR / "validation_10_NEURO11010_tanh_0_E110.npz")
    x = real_val["x"].ravel()
    u_ref = real_val["u_reference"].ravel()

    fig = plt.figure(figsize=(COL2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.62, 1, 1], height_ratios=[1, 0.95])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :])

    chip_map(ax_a)
    panel_label(ax_a, "a", x=-0.05, y=1.0)

    # (b) profiles
    sub = slice(0, None, 5)
    ax_b.plot(x, real_val["u_initial_surrogate"].ravel(), color=MUTED, lw=0.8, ls=(0, (1, 1.2)), label="initial network")
    ax_b.plot(x, cls["u"].ravel(), color=C_CLASSICAL, lw=1.1, label="classical PINN, same protocol")
    ax_b.plot(x, u_ref, color=INK, lw=1.2, label="FDM reference", zorder=3)
    ax_b.plot(x[sub], fake_pred["u_real"].ravel()[sub], ls="none", marker="^", ms=3.4, color=C_FAKE,
              mec=SURFACE, mew=0.4, label="HQPINN: FakeBerlin-trained, on ibm_berlin", zorder=4)
    ax_b.plot(x[2::5], real_val["u_qpu"].ravel()[2::5], ls="none", marker="o", ms=3.2, color=C_QPU,
              mec=SURFACE, mew=0.4, label="HQPINN: ibm_berlin-trained, on ibm_berlin", zorder=5)
    ax_b.set_xlabel(r"$x$")
    ax_b.set_ylabel(r"$u(x,T=0.1)$")
    ax_b.set_xlim(0, 1)
    panel_label(ax_b, "b", x=-0.2)

    # (c) pointwise error with the chain-to-chain spread
    for kind, color, label in (("fake", C_FAKE, "FakeBerlin-trained"), ("real", C_QPU, "ibm_berlin-trained")):
        xx, u_mean, u_chain = per_chain_u(kind)
        ax_c.fill_between(xx, u_chain.min(0) - u_ref, u_chain.max(0) - u_ref, color=color, alpha=0.15, lw=0)
        ax_c.plot(xx, u_mean - u_ref, color=color, lw=1.1)
    ax_c.plot(x, real_val["u_ideal"].ravel() - u_ref, color=C_QPU, lw=0.8, ls=(0, (3, 1.5)),
              label="ibm_berlin-trained, ideal circuit")
    ax_c.axhline(0, color=INK2, lw=0.5)
    ax_c.set_xlabel(r"$x$")
    ax_c.set_ylabel(r"$u-u_{\rm FDM}$")
    ax_c.set_xlim(0, 1)
    hb, lb = ax_b.get_legend_handles_labels()
    hc, lc_ = ax_c.get_legend_handles_labels()
    handles = dict(zip(lb + lc_, hb + hc))
    order = ["FDM reference", "initial network", "classical PINN, same protocol",
             "HQPINN: ibm_berlin-trained, on ibm_berlin", "HQPINN: FakeBerlin-trained, on ibm_berlin",
             "ibm_berlin-trained, ideal circuit"]
    names = {"ibm_berlin-trained, ideal circuit": "HQPINN: ibm_berlin-trained, ideal circuit"}
    fig.legend([handles[k] for k in order], [names.get(k, k) for k in order], loc="outside upper right",
               ncol=3, fontsize=5.8, columnspacing=1.4, handlelength=1.8)
    panel_label(ax_c, "c", x=-0.22)

    # (d) learning curves
    ep = lc["epochs"]
    c = lc["classical_mse"].reshape(-1, len(ep))
    q25, q50, q75 = np.quantile(c, [0.25, 0.5, 0.75], axis=0)
    ax_d.fill_between(ep, q25, q75, color=C_CLASSICAL, alpha=0.10, lw=0)
    ax_d.plot(ep, q50, color=C_CLASSICAL, lw=1.1, label="classical PINN, same protocol (median, IQR; 50 runs)")
    ax_d.plot(ep, np.median(sims["ideal_mse"], axis=0), color=C_QPU, lw=0.9, ls=(0, (3, 1.5)),
              label="HQPINN, ideal circuit (median of 5 seeds)")
    ax_d.plot(ep, lc["fake_mse_surrogate"], color=C_FAKE, lw=1.1, label="HQPINN on FakeBerlin (seed 0)")
    ax_d.plot(ep, lc["real_mse_surrogate"], color=C_QPU, lw=1.3, label="HQPINN on ibm_berlin (seed 0)")
    real_run, fake_run = load_json(real_file("run", "json")), load_json(fake_file("run", "json"))
    ax_d.plot([110], [real_run["mse"]], marker="*", ms=8, color=C_QPU, mec=SURFACE, mew=0.6, ls="none", zorder=6,
              label="final network evaluated on the device")
    ax_d.plot([110], [fake_run["fake"]["mse"]], marker="*", ms=8, color=C_FAKE, mec=SURFACE, mew=0.6, ls="none", zorder=6)
    ax_d.axhline(TRIVIAL_MSE[0.1], color=MUTED, lw=0.7)
    ax_d.text(110.8, TRIVIAL_MSE[0.1], "no evolution\n" r"($u=\sin 2\pi x$)", fontsize=5.5, color=INK2,
              va="center")
    for e in range(10, 111, 10):
        ax_d.plot([e, e], [0.94, 1.0], transform=ax_d.get_xaxis_transform(), color=MUTED, lw=0.6)
    ax_d.text(110.8, 0.97, "surrogate refits", transform=ax_d.get_xaxis_transform(), fontsize=5.5,
              color=INK2, va="center", ha="left")
    ax_d.set_yscale("log")
    ax_d.set_xlim(0, 112)
    ax_d.set_ylim(8e-5, 3)
    ax_d.set_xlabel("epoch (one QPU job per epoch)")
    ax_d.set_ylabel(r"MSE of $u(x,T)$")
    ax_d.legend(loc="lower left", ncol=2, fontsize=5.6, columnspacing=1.2, borderaxespad=0.3)
    panel_label(ax_d, "d", x=-0.065, y=1.0)
    save(fig, "fig3_hardware")


if __name__ == "__main__":
    main()
