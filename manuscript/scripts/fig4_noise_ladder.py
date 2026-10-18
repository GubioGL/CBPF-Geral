"""Fig. 4 -- noise ladder at the hardware budget (110 epochs, 10 collocation points per epoch).

Relative L2 error of u(x, T = 0.1) against the FDM, from the noiseless model to the real QPU,
all trained with the protocol of the ibm_berlin run:

  SIMULATION  classical PINN (widths 1-10 x seeds 0-4), and for n = 10 with the best learning-rate
              schedule of the sweep in controls_hardware_budget.py, with a tanh and with a sine middle
              block (seeds 0-4), HQPINN ideal circuit and ideal + 1024-shot
              noise (seeds 0-4, simulate_same_budget.py), HQPINN on FakeBerlin (seed 0 = this run,
              on the same physical chains as the real run; seeds 1-4 = the earlier FakeBerlin batch
              on chains chosen from the FakeBerlin snapshot, from the reference_fake_run of the real log);
  REAL QPU    FakeBerlin-trained network evaluated on ibm_berlin, and the network trained on ibm_berlin.

Filled markers: network evaluated on the device / model it was trained with. Open markers: the
same weights evaluated with the ideal circuit. Ticks: medians.
"""
import numpy as np

from paper_common import (C_CLASSICAL, C_FAKE, C_QPU, CACHE_DIR, COL1, CVQ_SUMMARY, INK, INK2, MUTED,
                          SURFACE, fake_file, load_json, load_npz, panel_label, real_file, save, set_style)

import matplotlib.pyplot as plt


def main():
    set_style()
    real = load_json(real_file("run", "json"))
    fake = load_json(fake_file("run", "json"))
    sims = load_npz(CACHE_DIR / "same_budget_sims.npz")
    cvq = load_json(CVQ_SUMMARY)
    u_ref = load_npz(real_file("validation"))["u_reference"].ravel()
    to_rel = lambda m: np.sqrt(np.asarray(m) / np.mean(u_ref ** 2))      # MSE -> relative L2

    classical = [r["relative_l2"] for row in cvq["classical"] for r in row["per_seed"].values()]
    batch = real["reference_fake_run"]["mse_all_seeds"]
    fake_batch = to_rel([batch[s] for s in sorted(batch) if s != "0"])
    # classical controls, n = 10, best schedule of the tanh LR sweep (controls_hardware_budget.py)
    ctrl = load_npz(CACHE_DIR / "controls_hardware_budget.npz")
    k = list(ctrl["widths"]).index(ctrl["sine_width"])
    rel = ctrl["tanh_rel_l2"][:, :, k]
    i, j = np.unravel_index(np.argmin(np.median(rel, axis=-1)), rel.shape[:2])

    rows = [   # (label, colour, trained-model values, ideal-circuit values, group)
        ("initial network", MUTED, [real["initial"]["surrogate"]["relative_l2"]], [], None),
        ("classical PINN,\nHQPINN learning rate", C_CLASSICAL, classical, [], "sim"),
        ("classical PINN, $n=10$,\ntuned learning rate", C_CLASSICAL, rel[i, j], [], "sim"),
        ("sine middle block, $n=10$,\ntuned learning rate", C_CLASSICAL, ctrl["sine_rel_l2"][i, j], [], "sim"),
        ("HQPINN, ideal circuit", C_QPU, sims["ideal_rel_l2"], [], "sim"),
        ("HQPINN, ideal + shot noise", C_QPU, sims["shots_rel_l2"], [], "sim"),
        ("HQPINN on FakeBerlin", C_FAKE, [fake["fake"]["relative_l2"]], [fake["ideal_circuit"]["relative_l2"]], "sim"),
        ("FakeBerlin-trained,\nevaluated on ibm_berlin", C_FAKE, [fake["real_prediction"]["relative_l2"]],
         [fake["ideal_circuit"]["relative_l2"]], "qpu"),
        ("trained and evaluated\non ibm_berlin", C_QPU, [real["relative_l2"]], [real["ideal_circuit"]["relative_l2"]], "qpu"),
    ]
    fig, ax = plt.subplots(figsize=(COL1, 0.39 * len(rows)), constrained_layout=True)
    rng = np.random.default_rng(1)
    ys = np.arange(len(rows))[::-1].astype(float)
    for y, (label, color, vals, ideal, group) in zip(ys, rows):
        vals = 100 * np.asarray(vals, float)
        if len(vals) > 1:
            jitter = rng.uniform(-0.16, 0.16, len(vals))
            many = len(vals) > 10
            ax.scatter(vals, y + jitter, s=9 if many else 12, color=color, alpha=0.5 if many else 0.6, lw=0, zorder=3)
            ax.plot([np.median(vals)] * 2, [y - 0.3, y + 0.3], color=INK, lw=1.0, zorder=4)
            if not many:      # seeds 0-4: seed 0 is the seed of the hardware runs
                ax.scatter(vals[:1], [y + jitter[0]], s=30, color=color, edgecolor=SURFACE, lw=0.8, zorder=5)
        else:
            star = group == "qpu"
            ax.scatter(vals, [y], s=80 if star else 30, color=color, edgecolor=SURFACE, lw=0.8, zorder=5,
                       marker="*" if star else "o")
        for v in ideal:
            ax.scatter([100 * v], [y - 0.2], s=20, facecolor=SURFACE, edgecolor=color, lw=0.9, zorder=4)
        if label == "HQPINN on FakeBerlin":
            ax.scatter(100 * fake_batch, y + rng.uniform(-0.16, 0.16, len(fake_batch)), s=12, color=color,
                       alpha=0.6, lw=0, zorder=3)
    sim = ys[[r[4] == "sim" for r in rows]]
    qpu = ys[[r[4] == "qpu" for r in rows]]
    ax.axhline(sim.min() - 0.5, color=MUTED, lw=0.6)
    ax.axhline(sim.max() + 0.5, color=MUTED, lw=0.6)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xscale("log")
    ax.set_xlim(2.5, 60)
    ax.set_xticks([3, 5, 10, 20, 50], ["3", "5", "10", "20", "50"])
    ax.set_xlabel(r"relative $L_2$ error of $u(x,T)$ (%)")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for y, text in ((np.mean(sim), "simulation"), (np.mean(qpu), "real QPU")):
        ax.text(1.02, y, text, transform=ax.get_yaxis_transform(), rotation=270, va="center", ha="left",
                fontsize=6.5, color=INK2)
    save(fig, "fig4_noise_ladder")


if __name__ == "__main__":
    main()
