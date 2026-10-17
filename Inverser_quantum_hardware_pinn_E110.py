"""Inverse hybrid PINN for viscous Burgers with the quantum layer on an IBM QPU, trained in
EPOCHS = 110 epochs with the QPU in the loop: learns nu jointly with u(x,t).

Problem: u_t + u u_x = nu u_xx on [0, 1) periodic, t in [0, TFIN], nu = NU_TRUE, with
u(x,0) = sin(2 pi x) imposed exactly by the network. Model (Inverser_quantum_exact_pinn_paralelo.py):
pre-net -> RY angles -> fixed brickwork CNOT layer -> <Z_i> -> post-net, plus the learnable nu.

Every epoch, N_COLLOCATION fresh random points (x_c, t_c) are used twice:
  * PDE residual: r = A - nu*B (A = u_t + u u_x, B = u_xx), finite differences on the 5-point
    stencil of each point;
  * data: the exact (Cole-Hopf) solution at the same points, u(x_c, t_c).
So the QPU measures only the stencil centres: the same <Z> feeds the residual and the data.

Training (CONFIG, "fixed_varpro"), two optimizers:
  * AdamW for the network, loss = ramp * MAE(residual) + W_D * MAE(data), the residual weight
    ramping up from 0 over the first epochs;
  * "varpro" for nu alone, in log space: with the network fixed, the nu minimizing the batch
    residual is the |B|-weighted median of A/B, and log nu is relaxed towards it by a factor
    beta that decays over the training.

QPU in the loop (machinery of Inverser_quantum_hardware_pinn.py):
  1. calibration (1 job per qubit count, cached): random angles on every n-qubit chain of the
     chip -> least squares for the measured response, a degree-1 trigonometric polynomial in
     each angle of the causal cone (HardwareSurrogateLayer);
  2. training (1 job per epoch): <Z> measured at the current angles of the stencil centres;
     the forward pass uses the QPU values and the gradient goes through the polynomial
     (straight-through), q(phi) = poly(phi) + stopgrad(q_QPU(phi_c) - poly(phi_c)), with the
     same correction on the 5 stencil points of centre c. Every RECAL_EVERY epochs the
     polynomial is refitted with the calibration + all measurements;
  3. validation (1 job): u(x, TFIN) of the trained network running on the QPU vs the exact solution.

Fake QPU (default): the noise model of BACKEND simulated locally with Aer (FakeChainCircuit),
no IBM job. Real QPU (--real): PackedCircuit in a Session -- not run in the verification.

Usage (conda env `qiskit`):
    python Inverser_quantum_hardware_pinn_E110.py                        # fake: 3/7/10 qubits x seeds 0/1/2
    python Inverser_quantum_hardware_pinn_E110.py --qubits 7 10
    python Inverser_quantum_hardware_pinn_E110.py --real --qubits 3 --seeds 0
Outputs (runs, calibrations, summary.json, nu_summary_10_E110.png): OUT_DIR/quantico_<backend>/.
"""
import argparse
import glob
import json
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")  # headless backend -- safe across worker processes
import matplotlib.pyplot as plt
import numpy as np
import torch as tc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantum_hardware"))
from qiskit_ibm_runtime import QiskitRuntimeService, Session  # noqa: E402

from Inverser_quantum_exact_pinn_paralelo import (  # noqa: E402  -- the model and the FD residual
    FIXED_LAYER, FIXED_LAYER_REPEATS, HT, HX, INPUT_ROTATION_AXIS, L, ExactQuantumLayer, build_model,
    exact_solution, plot_loss, plot_parameter_history, plot_solution, relative_l2_error, residual_terms,
    set_seed, stencil_inputs,
)
from Inverser_quantum_hardware_pinn import (  # noqa: E402  -- the hardware machinery
    FAKE_BACKENDS, MITIGATION, MITIGATIONS, HardwareSurrogateLayer, PackedCircuit, causal_cones,
    cnot_gate_list, find_chains, fit_surrogate, network_angles, network_output, screen_chains,
)
from main.functions import repeated_cnot_network_matrix  # noqa: E402
from regression_hardware_nqubits import INSTANCE  # noqa: E402
from run_quantum_hardware import FakeChainCircuit  # noqa: E402

# ---- problem --------------------------------------------------------------------
NU_TRUE = 0.1
TFIN = 0.1
N_COLLOCATION = 10        # random points per epoch: residual AND data
EPOCHS = 110
QUBITS = (3, 7, 10)
SEEDS = (0, 1, 2)

# ---- training: "fixed_varpro" (data README, data/Inverso_110/README.md) -----------
CONFIG = dict(
    lr_net=3.07e-2, lr_net_min=1.2e-3, betas_net=(0.77, 0.99),   # AdamW, cosine decay
    weight_decay=6.4e-6, grad_clip=10.0,
    r_ramp=20,                # epochs over which the residual weight ramps up from 0
    w_d=85.8,                 # data weight (the residual weight is 1)
    lr_nu=0.97, lr_nu_min=0.128,   # varpro relaxation factor beta, cosine decay
)

# ---- QPU ------------------------------------------------------------------------
BACKEND = "ibm_berlin"            # the fake is FAKE_BACKENDS[BACKEND]
PRECISION = 1 / 32                # per expectation value -> 1024 shots per circuit
CAL_SAMPLES = 1024                # calibration samples per output (~ chains x parameter sets)
MAX_CHAIN_SCORE_PER_QUBIT = 0.05  # chains with a worse error score (find_chains) are dropped
CHAIN_OUTLIER = 3.0               # ... and so are those with calibration misfit > 3 x the median
RECAL_EVERY = 10                  # refit the polynomial with the training measurements
VAL_POINTS = 100                  # validation: x points at t = TFIN, one replica per chain
SESSION_MAX_TIME = "2h"           # --real: safety limit of the Session

OUT_DIR = os.path.join("data", "Inverso_110")


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


def train(model, seed, measure=None, cfg=CONFIG, Tfin=TFIN, n_col=N_COLLOCATION, epochs=EPOCHS):
    """Trains `model` for `epochs` epochs, one batch of n_col fresh random points per epoch.
    measure(epoch, xt), optional (QPU in the loop): gets the epoch's stencil points before the
    forward pass and returns the offset added to the output of `model.quantum`.
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
        if measure is not None:
            model.quantum.offset = measure(epoch, xt)
        a, b, loss_r, loss_d = loss_terms(model, xt, n_col, exact_at(x_f, t_f))

        ramp = min(1.0, (epoch + 1) / cfg["r_ramp"])
        opt.zero_grad()
        (ramp * loss_r + cfg["w_d"] * loss_d).backward(inputs=net_params)
        tc.nn.utils.clip_grad_norm_(net_params, cfg["grad_clip"])
        opt.step()
        if measure is not None:
            model.quantum.offset = None

        nu_star = min(max(nu_minimizer(a, b), 1.0e-4), 10.0)
        log_nu += beta * (math.log(nu_star) - log_nu)
        with tc.no_grad():
            nu.copy_(tc.exp(log_nu))

        hist["loss_r"].append(float(loss_r.detach()))
        hist["loss_d"].append(float(loss_d.detach()))
        hist["nu"].append(float(nu.detach()))
    return hist


# ----------------------------------------------------------------------
# QPU: chains, calibration
# ----------------------------------------------------------------------
def get_backend(real):
    if real:
        return QiskitRuntimeService(instance=INSTANCE).backend(BACKEND)
    return FAKE_BACKENDS[BACKEND]()


def make_qpu(backend, chains, n_qubits, real, threads=0):
    gates = cnot_gate_list(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS)
    if real:
        return PackedCircuit(backend, chains, n_qubits, gates, PRECISION, False, MITIGATIONS[MITIGATION])
    qpu = FakeChainCircuit(backend, chains, n_qubits, gates, PRECISION, None)
    qpu.estimator.backend.set_options(max_parallel_threads=threads)
    return qpu


def circuit_structure(n_qubits):
    gates = cnot_gate_list(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS)
    return repeated_cnot_network_matrix(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS), causal_cones(n_qubits, gates)


def calibrate(backend, n_qubits, real, out_dir):
    """Chains of n qubits + calibration measurements (1 job), cached in out_dir. Chains with a
    bad error score or that do not follow the chip-average polynomial are dropped."""
    path = os.path.join(out_dir, f"calibration_NQ{n_qubits}_{backend.name}.npz")
    if os.path.exists(path):
        cal = np.load(path)
        return cal["chains"].tolist(), cal["angles"], cal["evs"]
    M, cones = circuit_structure(n_qubits)
    chains, scores, _ = find_chains(backend, n_qubits)
    chains = [c for c, s in zip(chains, scores) if s <= MAX_CHAIN_SCORE_PER_QUBIT * n_qubits]
    qpu = make_qpu(backend, chains, n_qubits, real)
    rng = np.random.default_rng(0)
    angle_sets = rng.uniform(-np.pi, np.pi, size=(math.ceil(CAL_SAMPLES / qpu.B), qpu.B, n_qubits))
    angles, evs = angle_sets.reshape(-1, n_qubits), qpu.run(angle_sets, [], verbose=False).reshape(-1, n_qubits)
    keep, _ = screen_chains(angles, evs, qpu.B, cones, M, CHAIN_OUTLIER)
    rows = keep[np.arange(len(angles)) % qpu.B]
    chains = [c for c, k in zip(chains, keep) if k]
    os.makedirs(out_dir, exist_ok=True)
    np.savez(path, chains=np.array(chains), angles=angles[rows], evs=evs[rows], backend=backend.name)
    return chains, angles[rows], evs[rows]


# ----------------------------------------------------------------------
# One run: training with the QPU in the loop + validation on the QPU
# ----------------------------------------------------------------------
def run_one(n_qubits, seed, real=False, threads=0, epochs=EPOCHS):
    """Initialization as Inverser_quantum_exact_pinn_paralelo.run_one (set_seed,
    nu_init = np.random.random(), then the network)."""
    tc.set_num_threads(1)
    t0 = time.time()
    backend = get_backend(real)
    out_dir = os.path.join(OUT_DIR, f"quantico_{backend.name}")
    chains, cal_angles, cal_evs = calibrate(backend, n_qubits, real, out_dir)
    M, cones = circuit_structure(n_qubits)
    qpu = make_qpu(backend, chains, n_qubits, real, threads)
    coefs, fit_rms, _ = fit_surrogate(cal_angles, cal_evs, cones, M)

    set_seed(seed)
    nu_init = float(np.random.random())
    model = build_model(n_qubits, TFIN, nu_init)
    model.quantum = HardwareSurrogateLayer(cones, coefs)
    job_ids, meas_angles, meas_evs = [], [cal_angles], [cal_evs]

    def measure(epoch, xt):
        """<Z> on the QPU at the stencil centres -> offset of their 5 stencil points."""
        if epoch and epoch % RECAL_EVERY == 0:
            model.quantum.set_coefs(fit_surrogate(np.concatenate(meas_angles), np.concatenate(meas_evs), cones, M)[0])
        centres = xt[:N_COLLOCATION]
        with tc.no_grad():
            angles = network_angles(model, centres[:, :1], centres[:, 1:])
            q_poly = model.quantum(angles)
        q_qpu, slot_angles, slot_evs = qpu.evaluate(angles.numpy(), job_ids, shift=epoch, verbose=False)
        meas_angles.append(slot_angles)
        meas_evs.append(slot_evs)
        return (tc.as_tensor(q_qpu, dtype=q_poly.dtype) - q_poly).repeat(5, 1)

    with (Session(backend=backend, max_time=SESSION_MAX_TIME) if real else nullcontext()) as session:
        if real:
            qpu.set_mode(session)
        hist = train(model, seed, measure=measure, epochs=epochs)

        # ---------- validation: the trained network running on the QPU ----------
        x_val = np.linspace(0.0, L, VAL_POINTS, endpoint=False).reshape(-1, 1)
        x_t = tc.tensor(x_val, dtype=tc.get_default_dtype())
        t_t = tc.full_like(x_t, TFIN)
        with tc.no_grad():
            angles = network_angles(model, x_t, t_t)
            u_sur = model(tc.cat([x_t, t_t], 1)).numpy()
            q_exact = ExactQuantumLayer(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS, INPUT_ROTATION_AXIS)(angles)
            u_ideal = network_output(model, x_t, t_t, q_exact).numpy()
        q_qpu, _, _ = qpu.evaluate(angles.numpy(), job_ids, replicas=qpu.B, verbose=False)
    with tc.no_grad():
        u_qpu = network_output(model, x_t, t_t, q_qpu).numpy()
    u_ref = exact_solution(x_val, TFIN, nu=NU_TRUE, L=L)

    nu_learned = float(model.learnable_parameter.detach())
    out = dict(n_qubits=n_qubits, seed=seed, backend=backend.name, nu_init=nu_init,
               nu=nu_learned, nu_rel_err=abs(nu_learned - NU_TRUE) / NU_TRUE,
               rel_l2_T_qpu=relative_l2_error(u_qpu, u_ref), rel_l2_T_measured_model=relative_l2_error(u_sur, u_ref),
               rel_l2_T_ideal_circuit=relative_l2_error(u_ideal, u_ref), chains=len(chains),
               fit_rms_calibration=float(np.mean(fit_rms)), n_jobs=len(job_ids), time_s=time.time() - t0)

    # ---------- outputs ----------
    tag = f"{N_COLLOCATION}_NQ{n_qubits}_{seed}_E{epochs}"
    tc.save(dict(state_dict=model.state_dict(), config=model.config, cfg=CONFIG, cones=cones,
                 coefs=[c.tolist() for c in model.quantum.coef.numpy()], chains=chains, result=out),
            os.path.join(out_dir, f"model_{tag}.pt"))
    with open(os.path.join(out_dir, f"run_{tag}.json"), "w") as f:
        json.dump(dict(out, cfg=CONFIG, job_ids=job_ids, chain_list=chains), f, indent=1)
    np.savez(os.path.join(out_dir, f"losshist_{tag}.npz"), **{k: np.array(v) for k, v in hist.items()})
    np.savez(os.path.join(out_dir, f"validation_{tag}.npz"), x=x_val, angles=angles.numpy(), q_qpu=q_qpu,
             u_qpu=u_qpu, u_measured_model=u_sur, u_ideal_circuit=u_ideal, u_exact=u_ref)
    plot_loss({"residual (QPU values)": hist["loss_r"], "data (QPU values)": hist["loss_d"]},
              title=f"Hardware PINN (inverse, {epochs} epochs, {backend.name}) -- NQ={n_qubits}, seed {seed}",
              save_path=os.path.join(out_dir, f"loss_{tag}.png"))
    plot_parameter_history(hist["nu"], NU_TRUE,
                           title=f"Learned nu = {nu_learned:.5f} ({100 * out['nu_rel_err']:.2f}%) -- "
                                 f"{backend.name}, NQ={n_qubits}, seed {seed}",
                           save_path=os.path.join(out_dir, f"nu_history_{tag}.png"))
    plot_solution(x_val, {f"QPU ({backend.name})": u_qpu, "measured model": u_sur,
                          "same weights, ideal circuit": u_ideal}, ref=u_ref,
                  title=f"T={TFIN}, NQ={n_qubits}, seed {seed}: rel. L2 QPU = {out['rel_l2_T_qpu']:.2e}",
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


def summarize(out_dir):
    """summary.json and nu_summary of every run saved in out_dir; returns the runs."""
    runs = sorted((json.load(open(p)) for p in glob.glob(os.path.join(out_dir, f"run_*_E{EPOCHS}.json"))),
                  key=lambda r: (r["n_qubits"], r["seed"]))
    for r in runs:
        for k in ("cfg", "job_ids", "chain_list"):
            r.pop(k, None)
    err = np.array([r["nu_rel_err"] for r in runs])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(config=CONFIG, nu_true=NU_TRUE, Tfin=TFIN, n_collocation=N_COLLOCATION, epochs=EPOCHS,
                       data="exact solution at the collocation points", nu_rel_err_median=float(np.median(err)),
                       nu_rel_err_mean=float(err.mean()), nu_rel_err_max=float(err.max()), runs=runs), f, indent=1)
    done = sorted({r["n_qubits"] for r in runs if sum(q["n_qubits"] == r["n_qubits"] for q in runs) == len(SEEDS)})
    if done:
        plot_nu_summary(out_dir, done, "qubits", "NQ")
    return runs


def _run_job(job):
    return run_one(*job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=list(QUBITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--real", action="store_true", help=f"QPU real {BACKEND} (padrão: o fake, local)")
    parser.add_argument("--workers", type=int, default=6, help="runs em paralelo no fake")
    args = parser.parse_args()

    backend = get_backend(args.real)
    out_dir = os.path.join(OUT_DIR, f"quantico_{backend.name}")
    for n in args.qubits:   # once per qubit count, before the runs that share it
        calibrate(backend, n, args.real, out_dir)
    jobs = [(n, s) for n in args.qubits for s in args.seeds]
    if args.real:           # one Session per run, one after the other
        for n, s in jobs:
            run_one(n, s, real=True)
    else:
        threads = max(1, (os.cpu_count() or 2) // args.workers)
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(_run_job, [(n, s, False, threads) for n, s in jobs]))

    runs = summarize(out_dir)
    print(f"{backend.name}: NU={NU_TRUE}, T={TFIN}, N_col={N_COLLOCATION}, {EPOCHS} epochs -- learned nu (relative error)")
    for n in sorted({r["n_qubits"] for r in runs}):
        print(f"  NQ{n:<2d} " + "  ".join(f"seed {r['seed']}: {r['nu']:.4f} ({100 * r['nu_rel_err']:.1f}%)"
                                       for r in runs if r["n_qubits"] == n))
