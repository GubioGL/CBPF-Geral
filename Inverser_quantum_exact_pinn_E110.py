"""Inverse hybrid PINN (classical -> exact CNOT-cosine quantum layer -> classical) for viscous
Burgers, trained in only EPOCHS = 110 epochs: learns nu jointly with u(x,t).

The model, the exact Cole-Hopf data at t = TFIN and the finite-difference residual come from
Inverser_quantum_exact_pinn_paralelo.py; only the training changes (the why is in
data/Inverso/quantum_inverse_exact_E110/README.md):
  * two optimizers: AdamW for the network and, for nu alone (in log space, nu = exp(s)),
      - "varpro": log nu is relaxed towards the nu that minimizes the batch residual for the
        current network (the |B|-weighted median of A/B, residual = A - nu*B), or
      - "adam": Adam on log nu, fed with the gradient of the residual loss alone;
  * loss = ramp * w_r * MAE(residual) + w_d * MAE(data): the residual weight ramps up from 0
    over the first R_RAMP epochs, and the data weight is either fixed (w_d ~ 15-20) or set
    every epoch by gradient-norm balancing, w_k = (|g_r| + |g_d|) / |g_k| with g_k the
    gradient of loss k w.r.t. the network weights (Wang, Teng & Perdikaris, SIAM J. Sci.
    Comput. 2021);
  * one batch of N_COLLOCATION fresh random points per epoch (one forward pass), final state kept.

Usage (conda env `qiskit`):
    python Inverser_quantum_exact_pinn_E110.py                          # the 3 configurations
    python Inverser_quantum_exact_pinn_E110.py --config gradnorm_varpro
Outputs: OUT_DIR/T_<TFIN>/<config>/ (checkpoints, histories, plots, nu_summary, summary.json).
"""
import argparse
import json
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")  # headless backend -- safe across worker processes
import matplotlib.pyplot as plt
import numpy as np
import torch as tc

from Inverser_quantum_exact_pinn_paralelo import (
    DATA_GRID, HT, HX, L, N_DATA, build_model, exact_solution, plot_loss, plot_parameter_history,
    plot_solution, predict, relative_l2_error, residual_terms, set_seed, stencil_inputs,
)

# ---- problem ------------------------------------------------------------------
NU_TRUE = 0.1
TFIN = 0.1
N_COLLOCATION = 10        # random collocation points, redrawn every epoch
EPOCHS = 110
QUBITS = (3, 7, 10)
SEEDS = (0, 1, 2)

OUT_DIR = os.path.join("data", "Inverso", "quantum_inverse_exact_E110")

# ---- the 3 best configurations (data/Inverso/quantum_inverse_exact_E110/README_melhores_configuracoes.md)
# nu error over 3 qubit counts x 20 validation seeds: median ~1% for all three (tied).
CONFIGS = {
    "gradnorm_varpro": dict(
        lr_net=2.55e-2, lr_net_min=6.9e-4, schedule_net="cosine", betas_net=(0.87, 0.999),
        weight_decay=2.6e-5, grad_clip=10.0, r_ramp=10,
        weighting="gradnorm",
        nu_opt="varpro", lr_nu=0.46, lr_nu_min=0.099, schedule_nu="exp",
    ),
    "fixed_varpro": dict(
        lr_net=1.85e-2, lr_net_min=1.8e-4, schedule_net="cosine", betas_net=(0.75, 0.99),
        weight_decay=1.9e-4, grad_clip=0.5, r_ramp=20,
        weighting="fixed", w_d=15.4,
        nu_opt="varpro", lr_nu=0.166, lr_nu_min=0.013, schedule_nu="cosine",
    ),
    "fixed_adam": dict(
        lr_net=2.17e-2, lr_net_min=4.7e-3, schedule_net="exp", betas_net=(0.78, 0.99),
        weight_decay=1.0e-4, grad_clip=10.0, r_ramp=40,
        weighting="fixed", w_d=20.8,
        nu_opt="adam", lr_nu=0.213, lr_nu_min=9.9e-3, schedule_nu="cosine", betas_nu=(0.86, 0.999),
    ),
}


# ----------------------------------------------------------------------
# Problem: exact data at t = TFIN
# ----------------------------------------------------------------------
def problem_setup(Tfin=TFIN, nu=NU_TRUE):
    """data_points (x, t, u) at t = Tfin: the exact solution on the same N_DATA positions
    of a DATA_GRID uniform grid as Inverser_quantum_exact_pinn_paralelo.py."""
    x_grid = np.linspace(0.0, L, DATA_GRID, endpoint=False)
    x_np = x_grid[np.linspace(0, DATA_GRID - 1, N_DATA).astype(int)]
    to_tensor = lambda a: tc.tensor(np.asarray(a).reshape(-1, 1), dtype=tc.get_default_dtype())
    x_data = to_tensor(x_np)
    return x_data, tc.full_like(x_data, float(Tfin)), to_tensor(exact_solution(x_np, Tfin, nu=nu))


def lr_at(lr0, lr_min, kind, progress):
    """LR at `progress` in [0, 1] of the training: exponential or cosine decay lr0 -> lr_min."""
    if kind == "exp":
        return lr0 * (lr_min / lr0) ** progress
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------
# Training: two optimizers + loss balancing
# ----------------------------------------------------------------------
def loss_terms(model, xt, n_f, u_data):
    """Residual split r = A - nu*B (A, B vectors) and the MAE of the residual and of the data."""
    u = model(xt)
    a, b = residual_terms(u, n_f, HX, HT)
    a, b = a.view(-1), b.view(-1)
    loss_r = (a - model.learnable_parameter * b).abs().mean()
    loss_d = (u[5 * n_f:] - u_data).view(-1).abs().mean()
    return a, b, loss_r, loss_d


def nu_minimizer(a, b):
    """nu minimizing mean|A - nu*B| with the network fixed: the |B|-weighted median of A/B."""
    a, b = a.detach().double(), b.detach().double()
    ratio, weight = a / b, b.abs()
    order = tc.argsort(ratio)
    cum = tc.cumsum(weight[order], 0)
    return float(ratio[order][tc.searchsorted(cum, 0.5 * cum[-1])])


def gradnorm_weights(loss_r, loss_d, net_params):
    """w_k = (|g_r| + |g_d|) / |g_k|, g_k = gradient of loss k w.r.t. the network weights."""
    norm = lambda loss: float(tc.sqrt(sum(
        (g ** 2).sum() for g in tc.autograd.grad(loss, net_params, retain_graph=True))))
    s_r, s_d = norm(loss_r), norm(loss_d)
    return (s_r + s_d) / max(s_r, 1e-12), (s_r + s_d) / max(s_d, 1e-12)


def train(model, cfg, seed, data_points, Tfin=TFIN, n_col=N_COLLOCATION, epochs=EPOCHS, measure=None):
    """Trains `model` (QuantumExactInversePINN) for `epochs` epochs with configuration `cfg`
    (one of CONFIGS). Returns the history of the loss terms, weights and nu.

    measure: optional, for the QPU in the loop (Inverser_quantum_hardware_pinn_E110.py):
    measure(epoch, xt) gets the epoch's points (stencil + data) before the forward pass and
    returns the offset added to the output of `model.quantum` during that epoch."""
    x_data, t_data, u_data = data_points
    xt_data = tc.cat([x_data, t_data], dim=1)
    nu = model.learnable_parameter
    net_params = [p for k, p in model.named_parameters() if k != "learnable_parameter"]

    opt_net = tc.optim.AdamW(net_params, lr=cfg["lr_net"], betas=cfg["betas_net"],
                             weight_decay=cfg["weight_decay"])
    log_nu = tc.log(nu.detach().clone()).requires_grad_(True)
    if cfg["nu_opt"] == "adam":
        opt_nu = tc.optim.Adam([log_nu], lr=cfg["lr_nu"], betas=cfg["betas_nu"])

    generator = tc.Generator().manual_seed(seed)   # first draw = sample_collocation_points(seed=seed)
    w_r, w_d = 1.0, cfg.get("w_d", 1.0)
    hist = {k: [] for k in ("loss_r", "loss_d", "w_r", "w_d", "nu")}

    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        for g in opt_net.param_groups:
            g["lr"] = lr_at(cfg["lr_net"], cfg["lr_net_min"], cfg["schedule_net"], progress)
        lr_nu = lr_at(cfg["lr_nu"], cfg["lr_nu_min"], cfg["schedule_nu"], progress)

        x_f = tc.rand((n_col, 1), generator=generator) * L
        t_f = HT + tc.rand((n_col, 1), generator=generator) * (Tfin - 2.0 * HT)
        xt = tc.cat([stencil_inputs(x_f, t_f, L, HX, HT), xt_data], dim=0)
        if measure is not None:
            model.quantum.offset = measure(epoch, xt)
        a, b, loss_r, loss_d = loss_terms(model, xt, n_col, u_data)
        if cfg["weighting"] == "gradnorm":
            w_r, w_d = gradnorm_weights(loss_r, loss_d, net_params)
        ramp = min(1.0, (epoch + 1) / cfg["r_ramp"])

        if cfg["nu_opt"] == "adam":
            # nu sees the unweighted residual loss: the weights only shape the network step
            nu_grad = tc.autograd.grad(loss_r, nu, retain_graph=True)[0]
        opt_net.zero_grad()
        (ramp * w_r * loss_r + w_d * loss_d).backward(inputs=net_params)
        tc.nn.utils.clip_grad_norm_(net_params, cfg["grad_clip"])
        opt_net.step()
        if measure is not None:
            model.quantum.offset = None

        if cfg["nu_opt"] == "adam":
            opt_nu.zero_grad()
            opt_nu.param_groups[0]["lr"] = lr_nu
            log_nu.grad = (nu_grad * nu.detach()).view_as(log_nu)   # d L_r / d log(nu)
            opt_nu.step()
        else:  # varpro: lr_nu is the relaxation factor towards the batch minimizer
            nu_star = min(max(nu_minimizer(a, b), 1.0e-4), 10.0)
            with tc.no_grad():
                log_nu.add_(lr_nu * (math.log(nu_star) - log_nu))
        with tc.no_grad():
            nu.copy_(tc.exp(log_nu))

        hist["loss_r"].append(float(loss_r.detach()))
        hist["loss_d"].append(float(loss_d.detach()))
        hist["w_r"].append(w_r)
        hist["w_d"].append(w_d)
        hist["nu"].append(float(nu.detach()))
    return hist


# ----------------------------------------------------------------------
# Evaluation, outputs
# ----------------------------------------------------------------------
def evaluate(model, Tfin=TFIN, nu=NU_TRUE):
    """Relative error of nu and relative L2 error of u at t = Tfin and on the whole (x, t) grid."""
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_pred = predict(model, x_plot, Tfin)
    u_ref = exact_solution(x_plot, Tfin, nu=nu)
    xg, tg = np.meshgrid(np.linspace(0.0, L, 100, endpoint=False), np.linspace(0.0, Tfin, 21))
    xt = tc.tensor(np.stack([xg.ravel(), tg.ravel()], 1), dtype=tc.get_default_dtype())
    with tc.no_grad():
        u_xt = model(xt).numpy().ravel()
    nu_learned = float(model.learnable_parameter.detach())
    return dict(nu=nu_learned, nu_rel_err=abs(nu_learned - nu) / nu,
                rel_l2_T=relative_l2_error(u_pred, u_ref),
                rel_l2_xt=relative_l2_error(u_xt, exact_solution(xg.ravel(), tg.ravel(), nu=nu)),
                x_plot=x_plot, u_pred=u_pred, u_ref=u_ref)


def run_one(config_name, n_qubits, seed):
    """One run, saved in OUT_DIR/T_<TFIN>/<config_name>/. Same initialization as
    Inverser_quantum_exact_pinn_paralelo.run_one (set_seed, nu_init = np.random.random(),
    then the network)."""
    tc.set_num_threads(1)
    cfg = CONFIGS[config_name]
    set_seed(seed)
    nu_init = float(np.random.random())
    model = build_model(n_qubits, TFIN, nu_init)
    hist = train(model, cfg, seed, problem_setup())
    out = dict(evaluate(model), config=config_name, n_qubits=n_qubits, seed=seed, nu_init=nu_init)

    out_dir = os.path.join(OUT_DIR, f"T_{TFIN}", config_name)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{N_COLLOCATION}_NQ{n_qubits}_{seed}_E{EPOCHS}"
    model.save(os.path.join(out_dir, f"model_{tag}.pt"), training_params=dict(
        L=L, TFIN=TFIN, NU=NU_TRUE, HX=HX, HT=HT, N_COLLOCATION=N_COLLOCATION, N_DATA=N_DATA,
        SEED=seed, EPOCHS=EPOCHS, DATA_SOURCE="exact_cole_hopf", CONFIG=config_name, CFG=cfg,
        NU_INIT=nu_init, NU_LEARNED=out["nu"], NU_REL_ERR=out["nu_rel_err"],
        REL_L2_U_T=out["rel_l2_T"], REL_L2_U_XT=out["rel_l2_xt"]))
    np.savez(os.path.join(out_dir, f"losshist_{tag}.npz"), **{k: np.array(v) for k, v in hist.items()})
    plot_loss({"residual": hist["loss_r"], "data": hist["loss_d"]},
              title=f"Quantum Exact PINN (inverse, {EPOCHS} epochs) -- NQ={n_qubits}, seed {seed}",
              save_path=os.path.join(out_dir, f"loss_{tag}.png"))
    plot_parameter_history(hist["nu"], NU_TRUE,
                           title=f"Learned nu = {out['nu']:.5f} ({100 * out['nu_rel_err']:.2f}%) -- "
                                 f"NQ={n_qubits}, seed {seed}",
                           save_path=os.path.join(out_dir, f"nu_history_{tag}.png"))
    plot_solution(out.pop("x_plot"), {"Quantum Exact PINN": out.pop("u_pred")}, ref=out.pop("u_ref"),
                  title=f"T={TFIN}, NQ={n_qubits}, seed {seed} (rel. L2 = {out['rel_l2_T']:.2e})",
                  save_path=os.path.join(out_dir, f"solution_{tag}.png"))
    return out


def plot_nu_summary(out_dir):
    """nu vs epoch of the runs saved in `out_dir` (one configuration): one panel per qubit
    count, one line per seed."""
    colors = ("#2a78d6", "#eb6834", "#1baf7a")   # categorical slots 1-3, fixed order
    fig, axes = plt.subplots(1, len(QUBITS), figsize=(4.2 * len(QUBITS), 3.6), sharey=True)
    for ax, nq in zip(axes, QUBITS):
        for color, s in zip(colors, SEEDS):
            nu = np.load(os.path.join(out_dir, f"losshist_{N_COLLOCATION}_NQ{nq}_{s}_E{EPOCHS}.npz"))["nu"]
            ax.plot(np.arange(1, len(nu) + 1), nu, color=color, linewidth=2.0,
                    label=f"seed {s}: nu = {nu[-1]:.4f}")
        ax.axhline(NU_TRUE, color="#6b6a64", linestyle="--", linewidth=1.2, label=f"target {NU_TRUE:g}")
        ax.set_yscale("log")
        ax.set_title(f"{nq} qubits", fontsize=12)
        ax.set_xlabel("Epochs")
        ax.grid(True, linestyle="--", alpha=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("learned nu")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"nu_summary_{N_COLLOCATION}_E{EPOCHS}.png"), dpi=150)
    plt.close(fig)


def _run_job(job):
    return run_one(*job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", choices=tuple(CONFIGS) + ("all",), default="all")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    names = tuple(CONFIGS) if args.config == "all" else (args.config,)

    jobs = [(name, nq, s) for name in names for nq in QUBITS for s in SEEDS]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_run_job, jobs))

    print(f"NU={NU_TRUE}, T={TFIN}, N_col={N_COLLOCATION}, {EPOCHS} epochs -- learned nu (relative error)")
    for name in names:
        runs = [r for r in results if r["config"] == name]
        plot_nu_summary(os.path.join(OUT_DIR, f"T_{TFIN}", name))
        err = np.array([r["nu_rel_err"] for r in runs])
        print(f"{name}: nu error median {100 * np.median(err):.2f}%, mean {100 * err.mean():.2f}%, "
              f"max {100 * err.max():.2f}%")
        for nq in QUBITS:
            print(f"  NQ{nq:<2d} " + "  ".join(f"seed {r['seed']}: {r['nu']:.4f} ({100 * r['nu_rel_err']:.1f}%)"
                                            for r in runs if r["n_qubits"] == nq))
        with open(os.path.join(OUT_DIR, f"T_{TFIN}", name, "summary.json"), "w") as f:
            json.dump(dict(config=CONFIGS[name], nu_true=NU_TRUE, Tfin=TFIN, n_collocation=N_COLLOCATION,
                           epochs=EPOCHS, nu_rel_err_median=float(np.median(err)),
                           nu_rel_err_mean=float(err.mean()), nu_rel_err_max=float(err.max()), runs=runs),
                      f, indent=1)
