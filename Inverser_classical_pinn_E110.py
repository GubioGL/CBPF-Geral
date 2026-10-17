"""Inverse classical PINN for viscous Burgers trained in EPOCHS = 110 epochs: learns nu jointly
with u(x,t). Classical counterpart of Inverser_quantum_hardware_pinn_E110.py: same problem, same
data, same training (CONFIG "fixed_varpro"); the quantum layer is replaced by a trainable
classical bottleneck of k neurons (k = 3, 7, 10, like the qubits).

Problem: u_t + u u_x = nu u_xx on [0, 1) periodic, t in [0, TFIN], nu = NU_TRUE, with
u(x,0) = sin(2 pi x) imposed exactly by the network. Model (ClassicalPINN of
Inverser_classical_pinn_paralelo.py, the same as Inverser_classical_pinn.py):
pre-net (3 -> 10 -> k) -> net (k -> 10 -> k) -> post-net (k -> 10 -> 1), tanh, plus the learnable nu.

Every epoch, N_COLLOCATION fresh random points (x_c, t_c) are used for the PDE residual
(finite differences on the 5-point stencil) AND as data (the exact Cole-Hopf solution there).
Training: AdamW for the network, loss = ramp * MAE(residual) + W_D * MAE(data); "varpro" for nu
alone, in log space: log nu relaxed towards the |B|-weighted median of A/B (residual = A - nu*B).

At the end, if the quantum runs exist (OUT_DIR/quantico_<backend>/summary.json), compares the
best run of each model: learned nu vs epoch and u(x, TFIN) (OUT_DIR/comparacao_classico_quantico_E110.png).

Usage (conda env `qiskit` or any env with torch + scipy):
    python Inverser_classical_pinn_E110.py               # 3/7/10 neurons x seeds 0/1/2 + comparison
    python Inverser_classical_pinn_E110.py --comparar    # only the comparison
"""
import argparse
import glob
import json
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as tc

from Inverser_classical_pinn_paralelo import (
    HT, HX, L, build_model, exact_solution, plot_loss, plot_parameter_history, plot_solution, predict,
    relative_l2_error, residual_terms, set_seed, stencil_inputs,
)

# ---- problem --------------------------------------------------------------------
NU_TRUE = 0.1
TFIN = 0.1
N_COLLOCATION = 10        # random points per epoch: residual AND data
EPOCHS = 110
NEURONS = (3, 7, 10)      # width k of the trainable bottleneck
SEEDS = (0, 1, 2)
VAL_POINTS = 100          # u(x, TFIN) evaluated on the same x points as the quantum validation

# ---- training: "fixed_varpro", same scheme as Inverser_quantum_hardware_pinn_E110.py, values
# searched for this classical model with the same protocol (data/Inverso_110/README.md) ----------
CONFIG = dict(
    lr_net=2.53e-2, lr_net_min=3.2e-4, betas_net=(0.73, 0.99),   # AdamW, cosine decay
    weight_decay=9.3e-4, grad_clip=10.0,
    r_ramp=10,                # epochs over which the residual weight ramps up from 0
    w_d=37.8,                 # data weight (the residual weight is 1)
    lr_nu=0.50, lr_nu_min=0.163,   # varpro relaxation factor beta, cosine decay
)

OUT_DIR = os.path.join("data", "Inverso_110")
QUANTUM_DIR = os.path.join(OUT_DIR, "quantico_fake_berlin")


# ----------------------------------------------------------------------
# Training: AdamW for the network, varpro for nu
# ----------------------------------------------------------------------
def cosine(lr0, lr_min, progress):
    """Cosine decay lr0 -> lr_min, progress in [0, 1]."""
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress))


def exact_at(x, t):
    """Exact solution u(x, t) (Cole-Hopf, nu = NU_TRUE): the data, shaped like x."""
    return tc.tensor(exact_solution(x.numpy(), t.numpy(), nu=NU_TRUE, L=L), dtype=x.dtype)


def loss_terms(model, xt, n_f, u_data):
    """Residual split r = A - nu*B (A, B vectors) and the MAE of the residual and of the data
    misfit at the stencil centres (the collocation points themselves)."""
    u = model(xt)
    a, b = residual_terms(u, n_f, HX, HT)
    a, b = a.view(-1), b.view(-1)
    loss_r = (a - model.learnable_parameter * b).abs().mean()
    loss_d = (u[:n_f] - u_data).view(-1).abs().mean()
    return a, b, loss_r, loss_d


def nu_minimizer(a, b):
    """nu minimizing mean|A - nu*B| with the network fixed: the |B|-weighted median of A/B."""
    a, b = a.detach().double(), b.detach().double()
    ratio, weight = a / b, b.abs()
    order = tc.argsort(ratio)
    cum = tc.cumsum(weight[order], 0)
    return float(ratio[order][tc.searchsorted(cum, 0.5 * cum[-1])])


def train(model, seed, cfg=CONFIG, Tfin=TFIN, n_col=N_COLLOCATION, epochs=EPOCHS):
    """Trains `model` for `epochs` epochs, one batch of n_col fresh random points per epoch.
    Returns the history of the loss terms and of nu."""
    nu = model.learnable_parameter
    net_params = [p for k, p in model.named_parameters() if k != "learnable_parameter"]
    opt = tc.optim.AdamW(net_params, lr=cfg["lr_net"], betas=cfg["betas_net"], weight_decay=cfg["weight_decay"])
    log_nu = tc.log(nu.detach().clone())
    generator = tc.Generator().manual_seed(seed)
    hist = {k: [] for k in ("loss_r", "loss_d", "nu")}

    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        for g in opt.param_groups:
            g["lr"] = cosine(cfg["lr_net"], cfg["lr_net_min"], progress)
        beta = cosine(cfg["lr_nu"], cfg["lr_nu_min"], progress)

        x_f = tc.rand((n_col, 1), generator=generator) * L
        t_f = HT + tc.rand((n_col, 1), generator=generator) * (Tfin - 2.0 * HT)
        xt = stencil_inputs(x_f, t_f, L, HX, HT)   # 5 blocks of n_col rows, the centres first
        a, b, loss_r, loss_d = loss_terms(model, xt, n_col, exact_at(x_f, t_f))

        ramp = min(1.0, (epoch + 1) / cfg["r_ramp"])
        opt.zero_grad()
        (ramp * loss_r + cfg["w_d"] * loss_d).backward(inputs=net_params)
        tc.nn.utils.clip_grad_norm_(net_params, cfg["grad_clip"])
        opt.step()

        nu_star = min(max(nu_minimizer(a, b), 1.0e-4), 10.0)
        log_nu += beta * (math.log(nu_star) - log_nu)
        with tc.no_grad():
            nu.copy_(tc.exp(log_nu))

        hist["loss_r"].append(float(loss_r.detach()))
        hist["loss_d"].append(float(loss_d.detach()))
        hist["nu"].append(float(nu.detach()))
    return hist


# ----------------------------------------------------------------------
# Runs, outputs
# ----------------------------------------------------------------------
def run_one(neurons, seed, out_dir):
    """Same initialization as the quantum script: set_seed, nu_init = np.random.random(), network."""
    tc.set_num_threads(1)
    t0 = time.time()
    set_seed(seed)
    nu_init = float(np.random.random())
    model = build_model(neurons, TFIN, nu_init)
    hist = train(model, seed)

    x_val = np.linspace(0.0, L, VAL_POINTS, endpoint=False).reshape(-1, 1)
    u_pred, u_ref = predict(model, x_val, TFIN), exact_solution(x_val, TFIN, nu=NU_TRUE, L=L)
    nu_learned = float(model.learnable_parameter.detach())
    out = dict(neurons=neurons, seed=seed, nu_init=nu_init, nu=nu_learned,
               nu_rel_err=abs(nu_learned - NU_TRUE) / NU_TRUE, rel_l2_T=relative_l2_error(u_pred, u_ref),
               time_s=time.time() - t0)

    tag = f"{N_COLLOCATION}_NE{neurons}_{seed}_E{EPOCHS}"
    tc.save(dict(state_dict=model.state_dict(), cfg=CONFIG, result=out), os.path.join(out_dir, f"model_{tag}.pt"))
    with open(os.path.join(out_dir, f"run_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(out_dir, f"losshist_{tag}.npz"), **{k: np.array(v) for k, v in hist.items()})
    np.savez(os.path.join(out_dir, f"validation_{tag}.npz"), x=x_val, u_pred=u_pred, u_exact=u_ref)
    plot_loss({"residual": hist["loss_r"], "data": hist["loss_d"]},
              title=f"Classical PINN (inverse, {EPOCHS} epochs) -- {neurons} neurons, seed {seed}",
              save_path=os.path.join(out_dir, f"loss_{tag}.png"))
    plot_parameter_history(hist["nu"], NU_TRUE,
                           title=f"Learned nu = {nu_learned:.5f} ({100 * out['nu_rel_err']:.2f}%) -- "
                                 f"{neurons} neurons, seed {seed}",
                           save_path=os.path.join(out_dir, f"nu_history_{tag}.png"))
    plot_solution(x_val, {"Classical PINN": u_pred}, ref=u_ref,
                  title=f"T={TFIN}, {neurons} neurons, seed {seed}: rel. L2 = {out['rel_l2_T']:.2e}",
                  save_path=os.path.join(out_dir, f"solution_{tag}.png"))
    return out


def plot_nu_summary(out_dir, sizes, size_label, prefix):
    """nu vs epoch of the runs saved in `out_dir`: one panel per size (qubits or neurons),
    one line per seed (losshist_<N_COLLOCATION>_<prefix><size>_<seed>_E<EPOCHS>.npz)."""
    colors = ("#2a78d6", "#eb6834", "#1baf7a")   # categorical slots 1-3, fixed order
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.6), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for color, s in zip(colors, SEEDS):
            nu = np.load(os.path.join(out_dir, f"losshist_{N_COLLOCATION}_{prefix}{size}_{s}_E{EPOCHS}.npz"))["nu"]
            ax.plot(np.arange(1, len(nu) + 1), nu, color=color, linewidth=2.0, label=f"seed {s}: nu = {nu[-1]:.4f}")
        ax.axhline(NU_TRUE, color="#6b6a64", linestyle="--", linewidth=1.2, label=f"target {NU_TRUE:g}")
        ax.set_yscale("log")
        ax.set_title(f"{size} {size_label}", fontsize=12)
        ax.set_xlabel("Epochs")
        ax.grid(True, linestyle="--", alpha=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel("learned nu")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"nu_summary_{N_COLLOCATION}_E{EPOCHS}.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Comparison: best classical run vs best quantum run
# ----------------------------------------------------------------------
def best_run(folder, size_key, prefix):
    """Run of `folder` with the smallest nu error, with its nu history and u(x, TFIN)."""
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(folder, f"run_*_E{EPOCHS}.json"))]
    best = min(runs, key=lambda r: r["nu_rel_err"])
    tag = f"{N_COLLOCATION}_{prefix}{best[size_key]}_{best['seed']}_E{EPOCHS}"
    val = np.load(os.path.join(folder, f"validation_{tag}.npz"))
    u = val["u_qpu"] if "u_qpu" in val.files else val["u_pred"]
    return best, np.load(os.path.join(folder, f"losshist_{tag}.npz"))["nu"], val["x"], u, val["u_exact"]


def plot_comparison():
    classical, nu_c, x, u_c, u_ref = best_run(os.path.join(OUT_DIR, "classico"), "neurons", "NE")
    quantum, nu_q, x_q, u_q, _ = best_run(QUANTUM_DIR, "n_qubits", "NQ")
    l2_c, l2_q = relative_l2_error(u_c, u_ref), relative_l2_error(u_q, u_ref)
    label_c = (f"classical ({classical['neurons']} neurons, seed {classical['seed']}): "
               f"nu = {classical['nu']:.5f} ({100 * classical['nu_rel_err']:.2f}%)")
    label_q = (f"quantum, fake QPU ({quantum['n_qubits']} qubits, seed {quantum['seed']}): "
               f"nu = {quantum['nu']:.5f} ({100 * quantum['nu_rel_err']:.2f}%)")
    blue, orange, ink = "#2a78d6", "#eb6834", "#6b6a64"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    epochs = np.arange(1, EPOCHS + 1)
    ax1.plot(epochs, nu_c, color=blue, linewidth=2.0, label=label_c)
    ax1.plot(epochs, nu_q, color=orange, linewidth=2.0, label=label_q)
    ax1.axhline(NU_TRUE, color=ink, linestyle="--", linewidth=1.2, label=f"target nu = {NU_TRUE:g}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("learned nu")
    ax1.set_title("Learned viscosity")
    ax1.legend(fontsize=8, frameon=False, loc="lower right")

    ax2.plot(x.ravel(), u_ref.ravel(), color=ink, linewidth=3.0, label="exact (Cole-Hopf)")
    ax2.plot(x.ravel(), u_c.ravel(), color=blue, linewidth=2.0, linestyle="--", label=f"classical: rel. L2 = {l2_c:.2e}")
    ax2.plot(x_q.ravel(), u_q.ravel(), color=orange, linewidth=2.0, linestyle="--",
             label=f"quantum on the fake QPU: rel. L2 = {l2_q:.2e}")
    ax2.set_xlabel("x")
    ax2.set_ylabel(f"u(x, T={TFIN})")
    ax2.set_title(f"Solution at the final time T = {TFIN}")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", alpha=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"comparacao_classico_quantico_E{EPOCHS}.png"), dpi=150)
    plt.close(fig)
    print(f"best classical: {label_c}, rel. L2 of u(x,T) {l2_c:.2e}")
    print(f"best quantum:   {label_q}, rel. L2 of u(x,T) {l2_q:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--comparar", action="store_true", help="só a comparação clássico x quântico")
    args = parser.parse_args()

    if not args.comparar:
        out_dir = os.path.join(OUT_DIR, "classico")
        os.makedirs(out_dir, exist_ok=True)
        runs = [run_one(k, s, out_dir) for k in NEURONS for s in SEEDS]
        err = np.array([r["nu_rel_err"] for r in runs])
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(dict(config=CONFIG, nu_true=NU_TRUE, Tfin=TFIN, n_collocation=N_COLLOCATION, epochs=EPOCHS,
                           data="exact solution at the collocation points", nu_rel_err_median=float(np.median(err)),
                           nu_rel_err_mean=float(err.mean()), nu_rel_err_max=float(err.max()), runs=runs), f, indent=1)
        plot_nu_summary(out_dir, NEURONS, "neurons", "NE")
        print(f"classical: NU={NU_TRUE}, T={TFIN}, N_col={N_COLLOCATION}, {EPOCHS} epochs -- learned nu (relative error)")
        for k in NEURONS:
            print(f"  {k:2d} neurons " + "  ".join(f"seed {r['seed']}: {r['nu']:.4f} ({100 * r['nu_rel_err']:.1f}%)"
                                                for r in runs if r["neurons"] == k))
    if os.path.exists(os.path.join(QUANTUM_DIR, "summary.json")):
        plot_comparison()
