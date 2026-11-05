"""Automatic search of the optimizer settings of the forward hybrid PINN trained on the FAKE
IBM QPU (run_quantum_ibm_faker.py): seed 0, T = TFIN, exactly EPOCHS = 110 epochs, the
learning rate always starts at LR0 = 0.1 and then decays.

Searched (the "optimizer"): shape of the LR decay (linear / cosine / exponential, smooth or
in stairs, with an optional plateau at 0.1 first), final LR, AdamW betas, weight decay,
amsgrad, gradient clipping and an EMA (Polyak) average of the weights over the last epochs,
which averages out the shot noise the QPU puts on every step.

Two stages, so the search does not cost hundreds of 7-minute Aer runs:
  0. calibration on the fake (1 job, the same as run_quantum_ibm_faker.py; cached);
  1. PROXY: the calibrated polynomial plays the chip and every <Z> gets the binomial noise
     of 1024 shots (fit RMS 0.029 ~ shot noise 0.031: the polynomial already reproduces the
     fake at the shot-noise level). Every configuration is trained for the same 110 epochs
     on the same collocation stream and evaluated on R noise realizations (common random
     numbers: the same R for every configuration). Random search + refinement around the
     best ones, in parallel on the CPU cores. Score: mean log10(MSE of u(x,Tfin) vs FDM)
     + LOSS_WEIGHT * mean log10(final residual loss on the monitor grid);
  2. FAKE: the TOP best configurations + the optimizer of run_quantum_ibm_faker.py (ADAM_BEST,
     the winner of the first search) + the old 0.1 -> 0.01 stairs (baseline) are trained again
     with the FakeBerlin (Aer) in the loop, REPEATS times each (the Aer noise changes every
     run: one run alone is not enough to rank), validated on the fake (100 points x one
     replica per chain) and the best by median fake MSE is saved.

Adam + L-BFGS (50 + 60 epochs, Adam from lr 0.5) was tried and dropped: lr 0.5 diverges and
L-BFGS lowers the residual loss but raises the error of u(x,Tfin) under the shot noise.

--qubits n < 10: the 10-qubit chains of the real run are cut into disjoint contiguous
pieces of n qubits (4 qubits -> 16 chains); outputs in optimizer_search/NQ<n>/.

Everything goes to data/quantum_ibm_faker/T_<Tfin>/optimizer_search/[NQ<n>/][<tag>/].

Uso (ambiente conda `qiskit`):
    python search_optimizer_ibm_faker.py                       # busca completa, 10 qubits
    python search_optimizer_ibm_faker.py --qubits 4 --seed 42  # 4 qubits, outra seed
    python search_optimizer_ibm_faker.py --proxy-only          # só o estágio 1
    python search_optimizer_ibm_faker.py --fake-only --top 5 --repeats 3   # só o estágio 2 (reusa o proxy)
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

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")  # no window: figures are only saved
import matplotlib.pyplot as plt

import numpy as np
import torch as tc

from Inverser_quantum_hardware_pinn import (
    FAKE_BACKENDS, HardwareSurrogateLayer, causal_cones, cnot_gate_list, fit_surrogate, screen_chains,
)
from main.functions import (
    plot_loss, plot_solution, reference_at_time, relative_l2_error, repeated_cnot_network_matrix,
    sample_collocation_points, set_seed, solve_burgers_fd,
)
from run_quantum_hardware import FakeChainCircuit, build_model, residual_loss, stencil_inputs
from run_quantum_ibm_faker import (
    BACKEND, BETAS, CAL_SAMPLES, CHAIN_OUTLIER, EMA_DECAY, EMA_START, EPOCHS, FIXED_LAYER, FIXED_LAYER_REPEATS,
    GRAD_CLIP, HT, L, LR_HOLD, LR_MIN, N_COLLOCATION, N_EVALUATION_POINTS, N_QUBITS, OUT_DIR, PRECISION, REAL_RUN,
    RECAL_EVERY, SCHEDULING, TFIN, WEIGHT_DECAY,
)

SEED = 0
LR0 = 0.1                         # fixed: every schedule starts here and decays
SHOTS = int(np.ceil(PRECISION ** -2))
LOSS_WEIGHT = 0.2                 # weight of log10(residual loss) in the score (MSE dominates)
# the old training of run_quantum_ibm_faker.py: 0.1 -> 0.01 in stairs of 10 epochs
BASELINE = dict(schedule="legacy", lr_min=0.01, hold=0, every=10, beta1=0.9, beta2=0.999,
                weight_decay=1.0e-6, grad_clip=1.0, amsgrad=False, ema=0.0, ema_start=0.5)
# the optimizer of run_quantum_ibm_faker.py now: the best of the first search (seed 0, 10 qubits,
# fake median MSE 2.3e-3 over 3 runs)
ADAM_BEST = dict(schedule="exp", lr_min=LR_MIN, hold=LR_HOLD, every=1, beta1=BETAS[0], beta2=BETAS[1],
                 weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, amsgrad=False, ema=EMA_DECAY, ema_start=EMA_START)
REFERENCES = [BASELINE, ADAM_BEST]


# ----------------------------------------------------------------------
# Search space
# ----------------------------------------------------------------------
def sample_config(rng):
    return dict(
        schedule=str(rng.choice(["linear", "cosine", "exp"])),
        lr_min=float(10 ** rng.uniform(-4.0, -1.7)),
        hold=int(rng.choice([0, 5, 10, 20, 30])),
        every=int(rng.choice([1, 5, 10])),
        beta1=float(rng.uniform(0.6, 0.95)),
        beta2=float(rng.choice([0.9, 0.95, 0.99, 0.999])),
        weight_decay=float(10 ** rng.uniform(-7.0, -3.0)),
        grad_clip=float(rng.choice([0.25, 0.5, 1.0, 2.0, 5.0])),
        amsgrad=bool(rng.random() < 0.3),
        ema=float(rng.choice([0.0, 0.0, 0.8, 0.9, 0.95, 0.98])),
        ema_start=float(rng.choice([0.5, 0.7, 0.85])),
    )


def mutate(cfg, rng):
    """Neighbour of `cfg`: some keys resampled, the continuous ones perturbed."""
    child, fresh = dict(cfg), sample_config(rng)
    for key in child:
        if rng.random() < 0.25:
            child[key] = fresh[key]
    child["lr_min"] = float(np.clip(child["lr_min"] * math.exp(rng.normal(0, 0.4)), 1e-4, 0.05))
    child["beta1"] = float(np.clip(child["beta1"] + rng.normal(0, 0.04), 0.5, 0.97))
    child["weight_decay"] = float(np.clip(child["weight_decay"] * math.exp(rng.normal(0, 0.7)), 1e-8, 1e-2))
    return child


def lr_schedule(cfg, epochs):
    """epoch (1-based) -> LR: LR0 for `hold` epochs, then down to lr_min at the last stair."""
    if cfg["schedule"] == "legacy":
        return lambda epoch: max(LR0 - 0.01 * ((epoch - 1) // 10), 0.01)
    kind, lr_min, hold, every = cfg["schedule"], cfg["lr_min"], cfg["hold"], cfg["every"]
    span = max(((epochs - hold - 1) // every) * every, 1)

    def lr(epoch):
        if epoch <= hold:
            return LR0
        p = min(((epoch - hold - 1) // every) * every / span, 1.0)
        if kind == "exp":
            return LR0 * (lr_min / LR0) ** p
        f = 1.0 - p if kind == "linear" else 0.5 * (1.0 + math.cos(math.pi * p))
        return lr_min + (LR0 - lr_min) * f
    return lr


def describe(cfg):
    if cfg["schedule"] == "legacy":
        sched = "degraus 0.1→0.01 (antigo)"
    else:
        sched = (f"{cfg['schedule']} {LR0}→{cfg['lr_min']:.1e}"
                 + (f", platô {cfg['hold']}" if cfg["hold"] else "")
                 + (f", degraus de {cfg['every']}" if cfg["every"] > 1 else ""))
    text = (f"{sched} | β=({cfg['beta1']:.2f},{cfg['beta2']}) | wd {cfg['weight_decay']:.0e} | clip "
            f"{cfg['grad_clip']}" + (" | amsgrad" if cfg["amsgrad"] else "")
            + (f" | EMA {cfg['ema']} desde {cfg['ema_start']:.0%}" if cfg["ema"] else ""))
    return text


# ----------------------------------------------------------------------
# QPUs: the proxy (polynomial + shot noise) and the fake (Aer), same interface
# ----------------------------------------------------------------------
class ProxyQPU:
    """The calibrated polynomial as the device; every <Z> is the mean of SHOTS +-1 samples."""

    def __init__(self, cones, coefs, B, rng):
        self.truth, self.B, self.rng = HardwareSurrogateLayer(cones, coefs), B, rng

    def evaluate(self, angles, replicas=1, shift=0):
        a = np.repeat(angles, replicas, axis=0)
        with tc.no_grad():
            z = self.truth(tc.as_tensor(a, dtype=tc.get_default_dtype())).double().numpy()
        evs = 2.0 * self.rng.binomial(SHOTS, np.clip((1.0 + z) / 2.0, 0.0, 1.0)) / SHOTS - 1.0
        return evs.reshape(len(angles), replicas, -1).mean(1), a, evs


class FakeQPU:
    def __init__(self, hw):
        self.hw, self.B, self.job_ids = hw, hw.B, []

    def evaluate(self, angles, replicas=1, shift=0):
        return self.hw.evaluate(angles, self.job_ids, replicas=replicas, shift=shift, verbose=False)


# ----------------------------------------------------------------------
# One training: 110 epochs of QPU-in-the-loop AdamW (as run_quantum_ibm_faker.py)
# ----------------------------------------------------------------------
def problem(Tfin):
    """FDM reference on the evaluation grid and the fixed monitor grid."""
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=0.01)
    x_plot = np.linspace(0.0, L, N_EVALUATION_POINTS).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    x_mon, t_mon = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, distribution="uniform_exact")
    return dict(x_plot=x_plot, u_ref=u_ref, xt_mon=stencil_inputs(x_mon, t_mon))


def train(cfg, qpu, ctx, epochs, keep_model=False):
    n, n_col, Tfin = ctx["n"], N_COLLOCATION, TFIN
    x_plot, u_ref, xt_mon = ctx["x_plot"], ctx["u_ref"], ctx["xt_mon"]
    xt_plot = tc.tensor(np.hstack([x_plot, np.full_like(x_plot, Tfin)]), dtype=tc.get_default_dtype())
    set_seed(ctx["seed"])                                # same initialization as the faker run
    model = build_model(n, Tfin)
    generator = tc.Generator().manual_seed(ctx["seed"])  # same collocation stream
    model.quantum = HardwareSurrogateLayer(ctx["cones"], ctx["coefs"])
    opt = tc.optim.AdamW(model.parameters(), lr=LR0, betas=(cfg["beta1"], cfg["beta2"]),
                         weight_decay=cfg["weight_decay"], amsgrad=cfg["amsgrad"])
    lr_of = lr_schedule(cfg, epochs)
    ema, ema_from = None, int(round(cfg["ema_start"] * epochs)) + 1
    meas_a, meas_e = [ctx["cal_angles"]], [ctx["cal_evs"]]
    hist = dict(lr=[], loss_qpu=[], loss_monitor=[])
    coefs = ctx["coefs"]
    diverged = None
    for epoch in range(1, epochs + 1):
        lr = lr_of(epoch)
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        backup = [p.detach().clone() for p in model.parameters()]
        x_f = tc.rand((n_col, 1), generator=generator) * L
        t_f = HT + tc.rand((n_col, 1), generator=generator) * (Tfin - 2.0 * HT)
        xt = stencil_inputs(x_f, t_f)
        with tc.no_grad():
            angles = model.angles(xt[:n_col])
            q_poly = model.quantum(angles)
        q_qpu, a, e = qpu.evaluate(angles.numpy(), shift=epoch)
        # QPU value in the forward pass, gradient through the polynomial (qpu_step)
        model.quantum.offset = (tc.as_tensor(q_qpu, dtype=q_poly.dtype) - q_poly).repeat(5, 1)
        try:
            loss = residual_loss(model, xt)
        finally:
            model.quantum.offset = None
        opt.zero_grad()
        loss.backward()
        tc.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        if not all(tc.isfinite(p).all() for p in model.parameters()) or not tc.isfinite(loss):
            with tc.no_grad():                        # diverged: keep the last finite weights and stop
                for p, b in zip(model.parameters(), backup):
                    p.copy_(b)
            diverged = epoch
            break
        meas_a.append(a)
        meas_e.append(e)
        if RECAL_EVERY and epoch % RECAL_EVERY == 0:
            coefs, _, _ = fit_surrogate(np.concatenate(meas_a), np.concatenate(meas_e), ctx["cones"], ctx["M"])
            model.quantum.set_coefs(coefs)
        if cfg["ema"] and epoch >= ema_from:
            with tc.no_grad():
                if ema is None:
                    ema = {k: p.detach().clone() for k, p in model.named_parameters()}
                else:
                    for k, p in model.named_parameters():
                        ema[k].mul_(cfg["ema"]).add_(p.detach(), alpha=1.0 - cfg["ema"])
        with tc.no_grad():
            hist["loss_monitor"].append(float(residual_loss(model, xt_mon)))
        hist["lr"].append(lr)
        hist["loss_qpu"].append(float(loss.detach()))
    if ema is not None:
        with tc.no_grad():
            for k, p in model.named_parameters():
                p.copy_(ema[k])

    model.eval()
    with tc.no_grad():
        angles = model.angles(xt_plot)
        q_val, _, _ = qpu.evaluate(angles.numpy(), replicas=qpu.B)
        u_val = model.from_features(xt_plot, q_val).numpy()
        u_sur = model(xt_plot).numpy()
        loss_final = float(residual_loss(model, xt_mon))
    out = dict(mse=float(np.mean((u_val - u_ref) ** 2)), mae=float(np.mean(np.abs(u_val - u_ref))),
               relative_l2=relative_l2_error(u_val, u_ref), mse_surrogate=float(np.mean((u_sur - u_ref) ** 2)),
               loss_monitor=loss_final, loss_qpu_last10=float(np.mean(hist["loss_qpu"][-10:])), diverged=diverged)
    if keep_model:
        out.update(hist=hist, u_val=u_val, u_sur=u_sur, state_dict=model.state_dict(),
                   coefs=[np.asarray(c).tolist() for c in coefs])
    return out


# ----------------------------------------------------------------------
# Workers (spawned processes: the context goes in through the initializer)
# ----------------------------------------------------------------------
_CTX = None


def _init_worker(ctx, threads):
    global _CTX
    tc.set_num_threads(threads)
    _CTX = dict(ctx, **problem(TFIN))


def _proxy_trial(job):
    trial_id, cfg = job
    runs = []
    for r in range(_CTX["realizations"]):
        qpu = ProxyQPU(_CTX["cones"], _CTX["coefs"], _CTX["B"], np.random.default_rng(1000 + r))
        runs.append(train(cfg, qpu, _CTX, _CTX["epochs"]))
    mse = np.nan_to_num(np.array([run["mse"] for run in runs]), nan=np.inf)
    loss = np.nan_to_num(np.array([run["loss_monitor"] for run in runs]), nan=np.inf)
    score = float(np.mean(np.log10(mse)) + _CTX["loss_weight"] * np.mean(np.log10(loss)))
    return dict(id=trial_id, cfg=cfg, score=score, mse=mse.tolist(), mse_mean=float(mse.mean()),
                loss_monitor=loss.tolist(), loss_mean=float(loss.mean()),
                mse_surrogate=float(np.mean([run["mse_surrogate"] for run in runs])))


def _fake_trial(job):
    name, cfg = job
    fake = FAKE_BACKENDS[_CTX["backend"]]()
    n = _CTX["n"]
    gates = cnot_gate_list(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    qpu = FakeQPU(FakeChainCircuit(fake, _CTX["chains"], n, gates, PRECISION, SCHEDULING))
    t0 = time.time()
    out = train(cfg, qpu, _CTX, _CTX["epochs"], keep_model=True)
    return dict(name=name, cfg=cfg, time_s=time.time() - t0, n_jobs=len(qpu.job_ids), **out)


# ----------------------------------------------------------------------
def calibrate(fake, chains, n, gates, cones, M, seed, cache):
    """Phase 1 of run_quantum_ibm_faker.py (same angles, same chain screening); cached."""
    if cache.exists():
        d = np.load(cache)
        print(f"0/2 calibração: {cache.name} (cache)")
        return d["chains"].tolist(), d["angles"], d["evs"]
    hw = FakeChainCircuit(fake, chains, n, gates, PRECISION, SCHEDULING)
    T_cal = math.ceil(CAL_SAMPLES / hw.B)
    print(f"0/2 calibração no fake: 1 job × {T_cal} conjuntos ({hw.B} cadeias)...", flush=True)
    angle_sets = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=(T_cal, hw.B, n))
    evs = hw.run(angle_sets, [], verbose=False)
    angles, evs = angle_sets.reshape(-1, n), evs.reshape(-1, n)
    keep, misfit = screen_chains(angles, evs, hw.B, cones, M, CHAIN_OUTLIER)
    if not keep.all():
        print(f"  cadeias descartadas: {[c for c, k in zip(chains, keep) if not k]}")
        rows = keep[np.arange(len(angles)) % hw.B]
        angles, evs = angles[rows], evs[rows]
        chains = [c for c, k in zip(chains, keep) if k]
    np.savez(cache, chains=np.array(chains), angles=angles, evs=evs, backend=fake.name)
    return chains, angles, evs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", default=BACKEND, choices=tuple(FAKE_BACKENDS))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--qubits", type=int, default=N_QUBITS,
                   help="qubits do circuito (< 10: pedaços das cadeias de 10 qubits do run real)")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="épocas de cada treino (fixo em 110)")
    p.add_argument("--trials", type=int, default=160, help="configurações aleatórias no proxy")
    p.add_argument("--refine", type=int, default=8, help="melhores do proxy que geram vizinhos")
    p.add_argument("--children", type=int, default=6, help="vizinhos por configuração refinada")
    p.add_argument("--rounds", type=int, default=2, help="rodadas de refinamento")
    p.add_argument("--realizations", type=int, default=4, help="realizações do ruído de shot por configuração")
    p.add_argument("--loss-weight", type=float, default=LOSS_WEIGHT)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--top", type=int, default=3, help="melhores do proxy retreinadas no fake (+ o baseline)")
    p.add_argument("--repeats", type=int, default=2,
                   help="treinos no fake por configuração (o ruído do Aer muda a cada um; escolhe pela mediana)")
    p.add_argument("--fake-workers", type=int, default=4, help="treinos no fake em paralelo")
    p.add_argument("--proxy-only", action="store_true", help="não roda o estágio 2 (fake)")
    p.add_argument("--fake-only", action="store_true",
                   help="pula o proxy: usa o proxy_trials.json de uma busca anterior e roda só o estágio 2")
    p.add_argument("--recalibrate", action="store_true", help="ignora a calibração em cache")
    p.add_argument("--tag", default="", help="subpasta de optimizer_search/ para esta busca")
    return p.parse_args()


def main():
    args = parse_args()
    n, Tfin = args.qubits, TFIN
    search_dir = OUT_DIR / f"T_{Tfin}" / "optimizer_search"
    if n != N_QUBITS:
        search_dir = search_dir / f"NQ{n}"
    out_dir = search_dir / args.tag if args.tag else search_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = cnot_gate_list(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    M = repeated_cnot_network_matrix(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    cones = causal_cones(n, gates)
    fake = FAKE_BACKENDS[args.backend]()
    real_run = json.loads(REAL_RUN.read_text()) if REAL_RUN.exists() else None
    if real_run is None:
        raise SystemExit(f"{REAL_RUN} não existe: preciso das cadeias do run real.")
    t_start = time.time()
    print(f"Busca do otimizador | {fake.name} | {n} qubits | seed {args.seed} | T={Tfin} | {N_COLLOCATION} pontos/época | "
          f"{args.epochs} épocas | AdamW, lr inicial {LR0}\nSaída em {out_dir}", flush=True)

    cache = search_dir / f"calibration_seed{args.seed}_{fake.name}.npz"
    if args.recalibrate and cache.exists():
        cache.unlink()
    # the physical chains of the real run (10 qubits); with fewer qubits, each one is cut into
    # disjoint contiguous pieces of n qubits (still coupled in line, same physical qubits)
    chains = [c[k * n:(k + 1) * n] for c in real_run["chains"] for k in range(len(c) // n)]
    chains, cal_angles, cal_evs = calibrate(fake, chains, n, gates, cones, M, args.seed, cache)
    coefs, fit_rms, _ = fit_surrogate(cal_angles, cal_evs, cones, M)
    print(f"  {len(chains)} cadeias de {n} qubits | RMS do ajuste {np.mean(fit_rms):.4f} (ruído de shot ≈ "
          f"{PRECISION:.4f})")
    ctx = dict(n=n, seed=args.seed, epochs=args.epochs, cones=cones, M=M, coefs=coefs, cal_angles=cal_angles,
               cal_evs=cal_evs, B=len(chains), chains=chains, backend=args.backend,
               realizations=args.realizations, loss_weight=args.loss_weight)

    # ---------- 1. proxy search ----------
    rng = np.random.default_rng(12345)
    trials, seen = [], set()

    def run_batch(pool, cfgs, label):
        jobs = []
        for cfg in cfgs:
            key = json.dumps(cfg, sort_keys=True)
            if key not in seen:
                seen.add(key)
                jobs.append((len(trials) + len(jobs), cfg))
        t0 = time.time()
        results = list(pool.map(_proxy_trial, jobs))
        trials.extend(results)
        trials.sort(key=lambda t: t["score"])
        best = trials[0]
        print(f"  {label}: {len(jobs)} configurações em {time.time() - t0:.0f} s | melhor até agora: MSE "
              f"{best['mse_mean']:.2e}, loss {best['loss_mean']:.2e} -- {describe(best['cfg'])}", flush=True)

    if args.fake_only:
        trials = json.loads((out_dir / "proxy_trials.json").read_text())
        print(f"1/2 proxy: {len(trials)} configurações de proxy_trials.json (--fake-only)")
    else:
        n_total = len(REFERENCES) + args.trials + args.rounds * args.refine * args.children
        print(f"1/2 proxy (polinômio calibrado + ruído de {SHOTS} shots): ~{n_total} configurações × "
              f"{args.realizations} realizações, {args.workers} processos...", flush=True)
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(ctx, 1)) as pool:
            run_batch(pool, REFERENCES + [sample_config(rng) for _ in range(args.trials)], "busca aleatória")
            for r in range(args.rounds):
                parents = [t["cfg"] for t in trials[:args.refine]]
                run_batch(pool, [mutate(p, rng) for p in parents for _ in range(args.children)],
                          f"refinamento {r + 1}")
        (out_dir / "proxy_trials.json").write_text(json.dumps(trials, indent=1))
    base = next(t for t in trials if t["cfg"] == BASELINE)
    print(f"  baseline (degraus antigos) no proxy: MSE {base['mse_mean']:.2e}, loss {base['loss_mean']:.2e} "
          f"(posição {trials.index(base) + 1}/{len(trials)})")
    adam_ref = next((t for t in trials if t["cfg"] == ADAM_BEST), None)
    if adam_ref:
        print(f"  otimizador atual de run_quantum_ibm_faker.py no proxy: MSE {adam_ref['mse_mean']:.2e}, loss "
              f"{adam_ref['loss_mean']:.2e} (posição {trials.index(adam_ref) + 1}/{len(trials)})")
    print("  top 5 do proxy:")
    for t in trials[:5]:
        print(f"    MSE {t['mse_mean']:.2e} | loss {t['loss_mean']:.2e} | {describe(t['cfg'])}")
    summary = dict(started=datetime.fromtimestamp(t_start).isoformat(), backend=fake.name, seed=args.seed,
                   epochs=args.epochs, lr0=LR0, Tfin=Tfin, n_collocation=N_COLLOCATION, chains=chains,
                   args=vars(args), proxy_baseline=base, proxy_top=trials[:10])
    if args.proxy_only:
        (out_dir / "best_optimizer.json").write_text(json.dumps(dict(summary, best=trials[0]), indent=2))
        print(f"--proxy-only: melhor configuração em {out_dir / 'best_optimizer.json'}")
        return

    # ---------- 2. the best configurations with the fake QPU in the loop ----------
    candidates = ([("baseline", BASELINE), ("adam_best", ADAM_BEST)]
                  + [(f"proxy{i + 1}", t["cfg"]) for i, t in
                     enumerate([t for t in trials if t["cfg"] not in REFERENCES][:args.top])])
    jobs = [(f"{name}_r{r + 1}" if args.repeats > 1 else name, cfg) for name, cfg in candidates
            for r in range(args.repeats)]
    threads = max(1, (os.cpu_count() or 2) // args.fake_workers)
    print(f"2/2 fake ({fake.name}, Aer) no loop: {len(candidates)} configurações × {args.repeats} treinos de "
          f"{args.epochs} épocas, {args.fake_workers} em paralelo (~{3.5 * args.epochs / 60:.0f} min cada)...",
          flush=True)
    for name, cfg in candidates:
        print(f"  {name:9s} {describe(cfg)}")
    t0 = time.time()
    with ProcessPoolExecutor(args.fake_workers, initializer=_init_worker, initargs=(ctx, threads)) as pool:
        fakes = list(pool.map(_fake_trial, jobs))
    for f in fakes:
        print(f"  {f['name']:12s} MSE fake {f['mse']:.3e} | modelo medido {f['mse_surrogate']:.3e} | loss "
              f"monitor {f['loss_monitor']:.3e} | {f['time_s'] / 60:.1f} min"
              + (f" | divergiu na época {f['diverged']}" if f["diverged"] else ""))
    print(f"  {time.time() - t0:.0f} s")
    run_score = lambda f: (f["mse"] * (f["loss_monitor"] ** args.loss_weight)
                           if np.isfinite(f["mse"]) and np.isfinite(f["loss_monitor"]) else np.inf)
    per_config = []
    for name, cfg in candidates:
        runs = [f for f in fakes if f["cfg"] == cfg]
        mse = np.array([f["mse"] for f in runs])
        per_config.append(dict(name=name, cfg=cfg, mse=mse.tolist(), mse_median=float(np.median(mse)),
                               mse_mean=float(mse.mean()), loss_monitor=[f["loss_monitor"] for f in runs],
                               score_median=float(np.median([run_score(f) for f in runs])),
                               best_run=min(runs, key=run_score)))
    per_config.sort(key=lambda c: c["score_median"])
    if args.repeats > 1:
        print("  por configuração (mediana sobre as repetições):")
        for c in per_config:
            print(f"    {c['name']:9s} MSE mediana {c['mse_median']:.3e} | runs " +
                  ", ".join(f"{m:.2e}" for m in c["mse"]))
    best = per_config[0]["best_run"]
    baseline = next(c for c in per_config if c["name"] == "baseline")["best_run"]

    # ---------- outputs ----------
    tag = f"{N_COLLOCATION}_NQ{n}_{args.seed}_{fake.name}_HW{args.epochs}"
    prob = problem(Tfin)
    for f in fakes:
        np.savez(out_dir / f"history_{f['name']}_{tag}.npz", lr=f["hist"]["lr"], loss_qpu=f["hist"]["loss_qpu"],
                 loss_monitor=f["hist"]["loss_monitor"], x=prob["x_plot"], u_fake=f["u_val"],
                 u_surrogate=f["u_sur"], u_reference=prob["u_ref"])
    tc.save(dict(state_dict=best["state_dict"], cfg=best["cfg"], seed=args.seed, cones=cones, coefs=best["coefs"],
                 chains=chains, backend=fake.name), out_dir / f"model_best_{tag}.pt")
    plot_loss({f"best ({best['name']}), fake batch": best["hist"]["loss_qpu"],
               "best, monitor grid": best["hist"]["loss_monitor"],
               "baseline, fake batch": baseline["hist"]["loss_qpu"],
               "baseline, monitor grid": baseline["hist"]["loss_monitor"]},
              title=f"Loss -- seed {args.seed}, {fake.name}, T={Tfin}", save_path=out_dir / f"loss_{tag}.png")
    plot_solution(prob["x_plot"], {f"best optimizer ({best['name']}), fake QPU": best["u_val"],
                                   "baseline optimizer, fake QPU": baseline["u_val"]},
                  ref=prob["u_ref"], save_path=out_dir / f"validation_{tag}.png")
    plt.figure(figsize=(7, 3.5))
    for f in fakes:
        plt.plot(range(1, args.epochs + 1), f["hist"]["lr"], label=f["name"], linewidth=2.0)
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("learning rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"lr_{tag}.png", dpi=150)
    plt.close()

    strip = lambda f: {k: v for k, v in f.items() if k not in ("hist", "u_val", "u_sur", "state_dict", "coefs")}
    summary.update(fake=[strip(f) for f in fakes], best=strip(best), finished=datetime.now().isoformat(),
                   per_config=[dict(c, best_run=c["best_run"]["name"]) for c in per_config],
                   total_time_s=time.time() - t_start)
    (out_dir / "best_optimizer.json").write_text(json.dumps(summary, indent=2))
    print(f"Melhor no fake: {per_config[0]['name']} (MSE mediana {per_config[0]['mse_median']:.3e}), melhor run "
          f"{best['name']} -- {describe(best['cfg'])}\n  u(x,{Tfin}) vs FDM (MSE): fake "
          f"{best['mse']:.3e} (baseline {baseline['mse']:.3e}) | loss monitor {best['loss_monitor']:.3e} "
          f"(baseline {baseline['loss_monitor']:.3e})\nsaída em {out_dir} ({(time.time() - t_start) / 60:.0f} min)")


if __name__ == "__main__":
    main()
