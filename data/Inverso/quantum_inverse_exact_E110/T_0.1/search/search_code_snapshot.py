# Frozen copy of Inverser_quantum_exact_pinn_E110.py as it ran the search (--search, --validate)
# that produced search_trials.json and validation.json. Not maintained; run from the repo root:
#   python data/Inverso/quantum_inverse_exact_E110/T_0.1/search/search_code_snapshot.py --search
# (from the repo root, so that Inverser_quantum_exact_pinn_paralelo is importable: set PYTHONPATH=.)
"""Inverse hybrid PINN (classical -> exact CNOT-cosine quantum layer -> classical)
for viscous Burgers, trained in only EPOCHS = 110 epochs: learns nu jointly with u(x,t).

Same model, exact Cole-Hopf data and FD residual as Inverser_quantum_exact_pinn_paralelo.py
(imported from there); what changes is the training, built for a 110-epoch budget (one
forward pass of the network per epoch, so it stays affordable on a QPU):
  * two optimizers: AdamW for the network (pre + post nets) and Adam for nu alone, each with
    its own learning rate and decay. nu is optimized in log space (nu = exp(s)), so it moves
    by relative steps and stays positive -- from nu_init ~ U(0, 1) to 0.1 in a few tens of
    epochs. The nu optimizer sees the gradient of the residual loss alone (nu does not
    appear in the data loss), not scaled by the residual weight;
  * loss = w_r * L_residual + w_d * L_data, with the weights fixed or adaptive:
      - "gradnorm": w_k = (|g_r| + |g_d|) / |g_k|, g_k = gradient of L_k w.r.t. the network
        weights (learning-rate annealing, Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021);
      - "ntk": w_k = (tr K_r + tr K_d) / tr K_k, tr K_k = mean over the points of
        |d f_i / d theta|^2 (f = residual or data misfit; Wang, Yu & Perdikaris, J. Comput.
        Phys. 2022);
    both refreshed every W_EVERY epochs and smoothed with an EMA;
  * the final state is returned (no best snapshot, no L-BFGS: 110 epochs are 110 steps).

Usage (conda env `qiskit`):
    python Inverser_quantum_exact_pinn_E110.py --search        # 1. configuration search (TUNING_SEEDS)
    python Inverser_quantum_exact_pinn_E110.py --validate      # 2. its winners on VALIDATION_SEEDS
    python Inverser_quantum_exact_pinn_E110.py                 # 3. test: 3, 7, 10 qubits x seeds 0, 1, 2
"""
import argparse
import json
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch as tc
from torch.func import functional_call, jacrev

from Inverser_quantum_exact_pinn_paralelo import LR as LR_ORIGINAL, train as train_original
from Inverser_quantum_exact_pinn_paralelo import (
    DATA_GRID, HT, HX, L, N_DATA, build_model, exact_solution, initial_condition, plot_loss,
    plot_parameter_history, plot_solution, predict, relative_l2_error, residual_terms,
    sample_collocation_points, set_seed, stencil_inputs,
)

# ---- problem -----------------------------------------------------------------
NU_TRUE = 0.1
TFIN = 0.1
N_COLLOCATION = 10        # random collocation points, redrawn every epoch
EPOCHS = 110
QUBITS = (3, 7, 10)
SEEDS = (0, 1, 2)
TUNING_SEEDS = tuple(range(100, 106))       # the search never sees SEEDS...
VALIDATION_SEEDS = tuple(range(200, 220))   # ...and picks its winners on these

OUT_DIR = os.path.join("data", "Inverso", "quantum_inverse_exact_E110")

# ---- training: the best configuration of --search, chosen by --validate (gradnorm + varpro),
# rounded; nu error over 3 qubit counts x 20 validation seeds: median 0.98%, mean 1.2%, max 5.4%
# (original protocol at 110 epochs: median 39%)
DEFAULT_CFG = dict(
    loss="mae",              # 'mae' or 'mse', for both terms
    lr_net=2.55e-2, lr_net_min=6.9e-4, schedule_net="cosine", betas_net=(0.87, 0.999),
    weight_decay=2.6e-5, grad_clip=10.0,
    nu_opt="varpro",         # 'adam' (on log nu) or 'varpro' (log nu relaxed towards the batch minimizer)
    lr_nu=0.46, lr_nu_min=0.099, schedule_nu="exp", betas_nu=(0.9, 0.999),
    nu_warmup=0,             # epochs with nu frozen while the network fits the data first
    r_ramp=10,               # epochs over which the residual weight ramps up from 0 (0: off)
    weighting="gradnorm",    # 'fixed', 'gradnorm' or 'ntk'
    w_r=1.0, w_d=1.0,        # fixed weights (initial values for the adaptive ones)
    w_every=1, w_ema=0.0,    # adaptive weights: refresh period (epochs) and EMA factor
)
# the best with Adam on log(nu) and fixed weights, rounded (validation: median 1.04%, mean 1.4%,
# max 6.2% -- statistically tied with DEFAULT_CFG)
ADAM_NU_CFG = dict(
    DEFAULT_CFG, lr_net=2.17e-2, lr_net_min=4.7e-3, schedule_net="exp", betas_net=(0.78, 0.99),
    weight_decay=1.0e-4, nu_opt="adam", lr_nu=0.213, lr_nu_min=9.9e-3, schedule_nu="cosine",
    betas_nu=(0.86, 0.999), r_ramp=40, weighting="fixed", w_d=20.8,
)


# ----------------------------------------------------------------------
# Problem: exact data at t = TFIN and the evaluation grid
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
    """LR at `progress` in [0, 1] of the training: constant, exponential or cosine decay."""
    if kind == "const":
        return lr0
    if kind == "exp":
        return lr0 * (lr_min / lr0) ** progress
    if kind == "cosine":
        return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule '{kind}'")


# ----------------------------------------------------------------------
# Training: two optimizers + adaptive loss weights
# ----------------------------------------------------------------------
def loss_terms(model, xt, n_f, u_data, loss):
    """Residual split r = A - nu*B (A, B vectors), data misfit d, and the reduced residual and
    data losses (MAE or MSE)."""
    u = model(xt)
    a, b = residual_terms(u, n_f, HX, HT)
    a, b = a.view(-1), b.view(-1)
    d = (u[5 * n_f:] - u_data).view(-1)
    reduce = (lambda v: v.abs().mean()) if loss == "mae" else (lambda v: v.pow(2).mean())
    return a, b, reduce(a - model.learnable_parameter * b), reduce(d)


def nu_minimizer(a, b, loss):
    """nu minimizing the residual loss of the batch with the network fixed: least squares
    sum(A B)/sum(B^2) for MSE, the |B|-weighted median of A/B for MAE."""
    a, b = a.detach().double(), b.detach().double()
    if loss == "mse":
        return float((a * b).sum() / (b * b).sum())
    ratio, weight = a / b, b.abs()
    order = tc.argsort(ratio)
    cum = tc.cumsum(weight[order], 0)
    return float(ratio[order][tc.searchsorted(cum, 0.5 * cum[-1])])


def ntk_traces(model, net_names, xt, n_f, u_data):
    """Mean NTK diagonal of the residual and data blocks: mean_i |d f_i / d theta|^2."""
    params = {k: v.detach() for k, v in model.named_parameters() if k in net_names}

    def outputs(p):
        u = functional_call(model, p, (xt,))
        a, b = residual_terms(u, n_f, HX, HT)
        return (a - model.learnable_parameter.detach() * b).view(-1), (u[5 * n_f:] - u_data).view(-1)

    jac_r, jac_d = jacrev(outputs)(params)
    trace = lambda jac: sum(j.flatten(1).pow(2).sum(1) for j in jac.values()).mean()
    return float(trace(jac_r)), float(trace(jac_d))


def train_e110(model, cfg, seed, data_points, Tfin=TFIN, n_col=N_COLLOCATION, epochs=EPOCHS):
    """Trains `model` (QuantumExactInversePINN) for `epochs` epochs, one batch of n_col fresh
    random collocation points per epoch. Returns the history (loss terms, weights, nu, LRs)."""
    cfg = dict(DEFAULT_CFG, **cfg)
    x_data, t_data, u_data = data_points
    xt_data = tc.cat([x_data, t_data], dim=1)
    nu = model.learnable_parameter
    net_names = [k for k, _ in model.named_parameters() if k != "learnable_parameter"]
    net_params = [p for k, p in model.named_parameters() if k != "learnable_parameter"]

    opt_net = tc.optim.AdamW(net_params, lr=cfg["lr_net"], betas=tuple(cfg["betas_net"]),
                             weight_decay=cfg["weight_decay"])
    log_nu = tc.log(nu.detach().clone()).requires_grad_(True)
    opt_nu = tc.optim.Adam([log_nu], lr=cfg["lr_nu"], betas=tuple(cfg["betas_nu"]))

    generator = tc.Generator().manual_seed(seed)   # first draw = sample_collocation_points(seed=seed)
    w_r, w_d = float(cfg["w_r"]), float(cfg["w_d"])
    hist = {k: [] for k in ("loss_r", "loss_d", "w_r", "w_d", "nu", "lr_net", "lr_nu")}

    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        lr_net = lr_at(cfg["lr_net"], cfg["lr_net_min"], cfg["schedule_net"], progress)
        nu_progress = max(epoch - cfg["nu_warmup"], 0) / max(epochs - 1 - cfg["nu_warmup"], 1)
        lr_nu = lr_at(cfg["lr_nu"], cfg["lr_nu_min"], cfg["schedule_nu"], nu_progress)
        for g in opt_net.param_groups:
            g["lr"] = lr_net
        for g in opt_nu.param_groups:
            g["lr"] = lr_nu

        x_f = tc.rand((n_col, 1), generator=generator) * L
        t_f = HT + tc.rand((n_col, 1), generator=generator) * (Tfin - 2.0 * HT)
        xt = tc.cat([stencil_inputs(x_f, t_f, L, HX, HT), xt_data], dim=0)
        a, b, loss_r, loss_d = loss_terms(model, xt, n_col, u_data, cfg["loss"])

        if cfg["weighting"] != "fixed" and epoch % cfg["w_every"] == 0:
            if cfg["weighting"] == "gradnorm":
                g_r = tc.autograd.grad(loss_r, net_params, retain_graph=True, allow_unused=True)
                g_d = tc.autograd.grad(loss_d, net_params, retain_graph=True, allow_unused=True)
                norm = lambda gs: float(tc.sqrt(sum((g ** 2).sum() for g in gs if g is not None)))
                s_r, s_d = norm(g_r), norm(g_d)
            else:  # ntk
                s_r, s_d = ntk_traces(model, net_names, xt, n_col, u_data)
            new_r, new_d = (s_r + s_d) / max(s_r, 1e-12), (s_r + s_d) / max(s_d, 1e-12)
            if epoch == 0:
                w_r, w_d = new_r, new_d
            else:
                ema = cfg["w_ema"]
                w_r, w_d = ema * w_r + (1 - ema) * new_r, ema * w_d + (1 - ema) * new_d
        ramp = min(1.0, (epoch + 1) / cfg["r_ramp"]) if cfg["r_ramp"] else 1.0

        # nu sees the unweighted residual loss: the weights only shape the network step
        nu_grad = tc.autograd.grad(loss_r, nu, retain_graph=True)[0]
        opt_net.zero_grad()
        (ramp * w_r * loss_r + w_d * loss_d).backward(inputs=net_params)
        tc.nn.utils.clip_grad_norm_(net_params, cfg["grad_clip"])
        opt_net.step()
        if epoch >= cfg["nu_warmup"]:
            if cfg["nu_opt"] == "adam":
                opt_nu.zero_grad()
                log_nu.grad = (nu_grad * nu.detach()).view_as(log_nu)  # d L_r / d log(nu)
                opt_nu.step()
            else:  # varpro: relax log(nu) towards the batch minimizer, lr_nu = relaxation factor
                nu_star = min(max(nu_minimizer(a, b, cfg["loss"]), 1.0e-4), 10.0)
                with tc.no_grad():
                    log_nu.add_(lr_nu * (math.log(nu_star) - log_nu))
            with tc.no_grad():
                nu.copy_(tc.exp(log_nu))

        hist["loss_r"].append(float(loss_r.detach()))
        hist["loss_d"].append(float(loss_d.detach()))
        hist["w_r"].append(w_r)
        hist["w_d"].append(w_d)
        hist["nu"].append(float(nu.detach()))
        hist["lr_net"].append(lr_net)
        hist["lr_nu"].append(lr_nu)
    return hist


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


def run_one(n_qubits, seed, cfg, save=False, label=""):
    """One run: same initialization as Inverser_quantum_exact_pinn_paralelo.run_one (set_seed,
    nu_init = np.random.random(), then the network)."""
    tc.set_num_threads(1)
    set_seed(seed)
    nu_init = float(np.random.random())
    model = build_model(n_qubits, TFIN, nu_init)
    data_points = problem_setup()
    t0 = time.time()
    hist = train_e110(model, cfg, seed, data_points)
    out = evaluate(model)
    out.update(n_qubits=n_qubits, seed=seed, nu_init=nu_init, time_s=time.time() - t0,
               loss_r=hist["loss_r"][-1], loss_d=hist["loss_d"][-1], w_r=hist["w_r"][-1], w_d=hist["w_d"][-1])
    if save:
        save_run(model, hist, out, dict(DEFAULT_CFG, **cfg), label)
    for k in ("x_plot", "u_pred", "u_ref"):
        out.pop(k)
    return out


def save_run(model, hist, out, cfg, label):
    out_dir = os.path.join(OUT_DIR, f"T_{TFIN}", label) if label else os.path.join(OUT_DIR, f"T_{TFIN}")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{N_COLLOCATION}_NQ{out['n_qubits']}_{out['seed']}_E{EPOCHS}"
    model.save(os.path.join(out_dir, f"model_{tag}.pt"), training_params=dict(
        L=L, TFIN=TFIN, NU=NU_TRUE, HX=HX, HT=HT, N_COLLOCATION=N_COLLOCATION, N_DATA=N_DATA,
        SEED=out["seed"], EPOCHS=EPOCHS, DATA_SOURCE="exact_cole_hopf", CFG=cfg,
        NU_INIT=out["nu_init"], NU_LEARNED=out["nu"], NU_REL_ERR=out["nu_rel_err"],
        REL_L2_U_T=out["rel_l2_T"], REL_L2_U_XT=out["rel_l2_xt"]))
    np.savez(os.path.join(out_dir, f"losshist_{tag}.npz"), **{k: np.array(v) for k, v in hist.items()})
    plot_loss({"residual": hist["loss_r"], "data": hist["loss_d"]},
              title=f"Quantum Exact PINN (inverse, {EPOCHS} epochs) -- NQ={out['n_qubits']}, seed {out['seed']}",
              save_path=os.path.join(out_dir, f"loss_{tag}.png"))
    plot_parameter_history(hist["nu"], NU_TRUE,
                           title=f"Learned nu = {out['nu']:.5f} ({100 * out['nu_rel_err']:.2f}%) -- "
                                 f"NQ={out['n_qubits']}, seed {out['seed']}",
                           save_path=os.path.join(out_dir, f"nu_history_{tag}.png"))
    plot_solution(out["x_plot"], {"Quantum Exact PINN": out["u_pred"]}, ref=out["u_ref"],
                  title=f"T={TFIN}, NQ={out['n_qubits']}, seed {out['seed']} (rel. L2 = {out['rel_l2_T']:.2e})",
                  save_path=os.path.join(out_dir, f"solution_{tag}.png"))


def run_baseline(n_qubits, seed):
    """The protocol of Inverser_quantum_exact_pinn_paralelo.py cut at EPOCHS epochs: one AdamW
    (lr 1e-2) for the network and nu together, MAE, fixed weights 1:1, best monitored snapshot,
    no L-BFGS. The reference the new training is compared with."""
    tc.set_num_threads(1)
    set_seed(seed)
    nu_init = float(np.random.random())
    model = build_model(n_qubits, TFIN, nu_init)
    x_f, t_f = sample_collocation_points(N_COLLOCATION, L, TFIN, HT, seed=seed)
    monitor = sample_collocation_points(N_COLLOCATION, L, TFIN, HT, distribution="uniform_exact")
    model, _ = train_original(
        model, x_f, t_f, epochs=EPOCHS, lr=LR_ORIGINAL, lbfgs_epochs=0, monitor_points=monitor,
        resample_fn=lambda: sample_collocation_points(N_COLLOCATION, L, TFIN, HT),
        data_points=problem_setup(), L=L)
    out = evaluate(model)
    for k in ("x_plot", "u_pred", "u_ref"):
        out.pop(k)
    return dict(out, n_qubits=n_qubits, seed=seed, nu_init=nu_init)


# ----------------------------------------------------------------------
# Search of the training configuration (tuning seeds only)
# ----------------------------------------------------------------------
WEIGHTINGS = ("fixed", "gradnorm", "ntk")


def sample_config(rng, weighting=None):
    nu_opt = str(rng.choice(["adam", "varpro"]))
    lr_net = float(10 ** rng.uniform(-2.5, -1.0))
    cfg = dict(
        loss=str(rng.choice(["mae", "mae", "mse"])),
        lr_net=lr_net, lr_net_min=lr_net * float(10 ** rng.uniform(-2.0, 0.0)),
        schedule_net=str(rng.choice(["exp", "cosine", "const"])),
        betas_net=(float(rng.uniform(0.7, 0.95)), float(rng.choice([0.9, 0.99, 0.999]))),
        weight_decay=float(10 ** rng.uniform(-6.0, -2.0)),
        grad_clip=float(rng.choice([0.5, 1.0, 2.0, 10.0])),
        nu_opt=nu_opt,
        schedule_nu=str(rng.choice(["exp", "cosine", "const"])),
        betas_nu=(float(rng.uniform(0.5, 0.95)), float(rng.choice([0.9, 0.99, 0.999]))),
        nu_warmup=int(rng.choice([0, 0, 5, 10, 20])),
        r_ramp=int(rng.choice([0, 0, 10, 20, 40])),
        weighting=weighting or str(rng.choice(WEIGHTINGS)),
        w_d=float(10 ** rng.uniform(-1.0, 2.0)),
        w_every=int(rng.choice([1, 5, 10, 20])),
        w_ema=float(rng.choice([0.0, 0.5, 0.9])),
    )
    lr_nu = float(10 ** rng.uniform(-2.0, -0.5)) if nu_opt == "adam" else float(rng.uniform(0.05, 1.0))
    lo = -2.0 if nu_opt == "adam" else -1.5
    cfg.update(lr_nu=lr_nu, lr_nu_min=lr_nu * float(10 ** rng.uniform(lo, 0.0)))
    return cfg


def mutate(cfg, rng):
    """Neighbour of `cfg` (same weighting and nu optimizer): some keys resampled, the
    continuous ones perturbed."""
    child, fresh = dict(cfg), sample_config(rng, cfg["weighting"])
    for key in child:
        if key not in ("weighting", "nu_opt", "lr_nu", "lr_nu_min") and rng.random() < 0.2:
            child[key] = fresh[key]
    jitter = lambda v, s: float(v * math.exp(rng.normal(0.0, s)))
    child["lr_net"] = float(np.clip(jitter(child["lr_net"], 0.3), 1e-3, 0.3))
    child["lr_net_min"] = float(min(jitter(child["lr_net_min"], 0.3), child["lr_net"]))
    hi = 0.3 if child["nu_opt"] == "adam" else 1.0
    child["lr_nu"] = float(np.clip(jitter(child["lr_nu"], 0.3), 1e-3, hi))
    child["lr_nu_min"] = float(min(jitter(child["lr_nu_min"], 0.3), child["lr_nu"]))
    child["w_d"] = jitter(child["w_d"], 0.5)
    return child


def score(outs):
    """Mean log10 of the relative error of nu over the runs (floored at 0.1%, so a run that
    happens to cross 0.1 at the last epoch does not dominate)."""
    err = np.nan_to_num(np.array([o["nu_rel_err"] for o in outs]), nan=10.0)
    return float(np.mean(np.log10(np.maximum(err, 1.0e-3))))


def _run_job(job):
    key, cfg, n_qubits, seed = job
    try:
        out = run_baseline(n_qubits, seed) if cfg is None else run_one(n_qubits, seed, cfg)
    except Exception as exc:   # a diverged configuration: worst score, the search goes on
        out = dict(n_qubits=n_qubits, seed=seed, nu=float("nan"), nu_rel_err=float("nan"),
                   rel_l2_T=float("nan"), rel_l2_xt=float("nan"), error=repr(exc))
    return key, out


def evaluate_configs(pool, configs, seeds):
    """{key: cfg (None = baseline)} -> {key: [run outputs over QUBITS x seeds]}."""
    jobs = [(key, cfg, nq, s) for key, cfg in configs.items() for nq in QUBITS for s in seeds]
    results = {key: [] for key in configs}
    for key, out in pool.map(_run_job, jobs, chunksize=4):
        results[key].append(out)
    return results


def summary_row(outs):
    err = np.array([o["nu_rel_err"] for o in outs])
    return dict(score=score(outs), nu_err_median=float(np.nanmedian(err)), nu_err_max=float(np.nanmax(err)),
                nu_err_mean=float(np.nanmean(err)),
                rel_l2_xt_median=float(np.nanmedian([o["rel_l2_xt"] for o in outs])))


def search(args):
    out_dir = os.path.join(OUT_DIR, f"T_{TFIN}", "search")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.search_seed)
    trials = []
    t0 = time.time()

    def run_batch(pool, cfgs, label):
        results = evaluate_configs(pool, dict(enumerate(cfgs)), TUNING_SEEDS)
        for i, cfg in enumerate(cfgs):
            trials.append(dict(cfg=cfg, **summary_row(results[i])))
        trials.sort(key=lambda t: t["score"])
        best = trials[0]
        print(f"  {label}: {len(cfgs)} configs ({time.time() - t0:.0f} s) | best: nu err median "
              f"{100 * best['nu_err_median']:.2f}%, max {100 * best['nu_err_max']:.2f}% "
              f"[{best['cfg']['weighting']}, {best['cfg']['nu_opt']}]", flush=True)

    print(f"Search: {args.trials} random configs + {args.rounds} refinement rounds, "
          f"{len(QUBITS)} qubit counts x tuning seeds {TUNING_SEEDS}, {args.workers} processes", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        base = evaluate_configs(pool, {"baseline": None}, TUNING_SEEDS)["baseline"]
        print(f"  baseline (original protocol, {EPOCHS} epochs): nu err median "
              f"{100 * summary_row(base)['nu_err_median']:.2f}%", flush=True)
        run_batch(pool, [sample_config(rng) for _ in range(args.trials)], "random search")
        for r in range(args.rounds):
            # refine the best of every (weighting, nu optimizer) pair, so that each
            # loss-balancing scheme gets the same tuning effort
            parents = []
            for w in WEIGHTINGS:
                for nu_opt in ("adam", "varpro"):
                    parents += [t["cfg"] for t in trials
                                if t["cfg"]["weighting"] == w and t["cfg"]["nu_opt"] == nu_opt][:args.refine]
            run_batch(pool, [mutate(p, rng) for p in parents for _ in range(args.children)],
                      f"refinement {r + 1}")

    best_by = {}
    for t in trials:
        best_by.setdefault(f"{t['cfg']['weighting']}+{t['cfg']['nu_opt']}", t)
    with open(os.path.join(out_dir, "search_trials.json"), "w") as f:
        json.dump(dict(baseline=summary_row(base), best=trials[0], best_by_scheme=best_by, trials=trials),
                  f, indent=1)
    print("Best per (weighting + nu optimizer), tuning seeds:")
    for key, t in sorted(best_by.items(), key=lambda kv: kv[1]["score"]):
        print(f"  {key:16s} nu err median {100 * t['nu_err_median']:6.2f}%  mean {100 * t['nu_err_mean']:6.2f}%  "
              f"max {100 * t['nu_err_max']:6.2f}% | rel L2(x,t) {t['rel_l2_xt_median']:.2e}")
    print(f"saved {os.path.join(out_dir, 'search_trials.json')} ({time.time() - t0:.0f} s)")


def validate(args):
    """The search picks the best of hundreds of configurations on only len(QUBITS) x
    len(TUNING_SEEDS) runs, so its winners are optimistic: the top `args.top` of every
    (weighting, nu optimizer) pair are re-run on VALIDATION_SEEDS and chosen there."""
    out_dir = os.path.join(OUT_DIR, f"T_{TFIN}", "search")
    with open(os.path.join(out_dir, "search_trials.json")) as f:
        trials = json.load(f)["trials"]
    candidates = {}
    for w in WEIGHTINGS:
        for nu_opt in ("adam", "varpro"):
            top = [t["cfg"] for t in trials if t["cfg"]["weighting"] == w and t["cfg"]["nu_opt"] == nu_opt]
            candidates.update({f"{w}+{nu_opt}#{i + 1}": cfg for i, cfg in enumerate(top[:args.top])})
    t0 = time.time()
    print(f"Validation: {len(candidates)} configs x {len(QUBITS)} qubit counts x {len(VALIDATION_SEEDS)} seeds",
          flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        results = evaluate_configs(pool, dict(candidates, baseline=None), VALIDATION_SEEDS)
    rows = {key: dict(cfg=candidates.get(key), **summary_row(outs)) for key, outs in results.items()}
    best_by = {}
    for key, row in sorted(rows.items(), key=lambda kv: kv[1]["score"]):
        if key != "baseline":
            best_by.setdefault(key.split("#")[0], dict(row, rank_in_search=int(key.split("#")[1])))
    print("Best per (weighting + nu optimizer), validation seeds:")
    print(f"  {'baseline':16s} nu err median {100 * rows['baseline']['nu_err_median']:6.2f}%  "
          f"mean {100 * rows['baseline']['nu_err_mean']:6.2f}%  max {100 * rows['baseline']['nu_err_max']:6.2f}%")
    for key, row in sorted(best_by.items(), key=lambda kv: kv[1]["score"]):
        print(f"  {key:16s} nu err median {100 * row['nu_err_median']:6.2f}%  mean {100 * row['nu_err_mean']:6.2f}%  "
              f"max {100 * row['nu_err_max']:6.2f}% | rel L2(x,t) {row['rel_l2_xt_median']:.2e} "
              f"(#{row['rank_in_search']} in the search)")
    with open(os.path.join(out_dir, "validation.json"), "w") as f:
        json.dump(dict(seeds=VALIDATION_SEEDS, best_by_scheme=best_by, all=rows), f, indent=1)
    print(f"saved {os.path.join(out_dir, 'validation.json')} ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------
# Test: QUBITS x SEEDS (never used by the search)
# ----------------------------------------------------------------------
def plot_nu_summary(save_path):
    """nu vs epoch of the saved test runs: one panel per qubit count, one line per seed."""
    import matplotlib.pyplot as plt
    colors = ("#2a78d6", "#eb6834", "#1baf7a")   # categorical slots 1-3, fixed order
    fig, axes = plt.subplots(1, len(QUBITS), figsize=(4.2 * len(QUBITS), 3.6), sharey=True)
    for ax, nq in zip(axes, QUBITS):
        for color, s in zip(colors, SEEDS):
            tag = f"{N_COLLOCATION}_NQ{nq}_{s}_E{EPOCHS}"
            nu = np.load(os.path.join(OUT_DIR, f"T_{TFIN}", f"losshist_{tag}.npz"))["nu"]
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
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def test(args):
    """DEFAULT_CFG, the validated best of every (weighting, nu optimizer) pair and the
    baseline on QUBITS x SEEDS; checkpoints and plots of DEFAULT_CFG."""
    validation_path = os.path.join(OUT_DIR, f"T_{TFIN}", "search", "validation.json")
    configs = {"DEFAULT_CFG": DEFAULT_CFG, "ADAM_NU_CFG": ADAM_NU_CFG}
    if os.path.exists(validation_path):
        with open(validation_path) as f:
            configs.update({key: row["cfg"] for key, row in json.load(f)["best_by_scheme"].items()})
    configs["baseline"] = None

    with ProcessPoolExecutor(args.workers) as pool:
        results = evaluate_configs(pool, configs, SEEDS)
    for nq in QUBITS:   # checkpoints + plots of the default (best) configuration
        for s in SEEDS:
            run_one(nq, s, DEFAULT_CFG, save=True)

    plot_nu_summary(os.path.join(OUT_DIR, f"T_{TFIN}", f"nu_summary_{N_COLLOCATION}_E{EPOCHS}.png"))

    print(f"Test: NU={NU_TRUE}, T={TFIN}, N_col={N_COLLOCATION}, {EPOCHS} epochs, seeds {SEEDS}")
    print("learned nu (relative error %) per qubit count, seeds in order")
    rows = sorted(((summary_row(outs), key, outs) for key, outs in results.items()), key=lambda r: r[0]["score"])
    for row, key, outs in rows:
        cells = []
        for nq in QUBITS:
            runs = sorted((o for o in outs if o["n_qubits"] == nq), key=lambda o: o["seed"])
            cells.append(f"NQ{nq}: " + " ".join(f"{o['nu']:.4f}({100 * o['nu_rel_err']:.1f})" for o in runs))
        print(f"{key:20s} median {100 * row['nu_err_median']:6.2f}% mean {100 * row['nu_err_mean']:6.2f}% | "
              + " | ".join(cells))
    summary = {key: dict(cfg=configs[key], runs=outs, **row) for row, key, outs in rows}
    with open(os.path.join(OUT_DIR, f"T_{TFIN}", "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--search", action="store_true", help="search the training configuration")
    parser.add_argument("--validate", action="store_true", help="re-run the search winners on VALIDATION_SEEDS")
    parser.add_argument("--top", type=int, default=5, help="validated configs per (weighting, nu optimizer)")
    parser.add_argument("--trials", type=int, default=480, help="random configurations")
    parser.add_argument("--rounds", type=int, default=2, help="refinement rounds")
    parser.add_argument("--refine", type=int, default=4, help="parents per (weighting, nu optimizer) pair")
    parser.add_argument("--children", type=int, default=6, help="neighbours per parent")
    parser.add_argument("--search-seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    if args.search:
        search(args)
    elif args.validate:
        validate(args)
    else:
        test(args)
