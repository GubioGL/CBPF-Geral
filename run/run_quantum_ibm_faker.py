"""Forward hybrid PINN for viscous Burgers trained on the FAKE IBM QPU (FakeBerlin, Aer) with
EXACTLY the configuration of the real-QPU run of run_quantum_hardware_real.py, then the
trained network is evaluated (prediction only, 1 job) on the REAL QPU.

Same configuration as the real run (data/quantum_forward_hardware_real/, REAL_RUN):
  * seed 0 (initialization + stream of collocation points), 10 qubits, 10 points per epoch,
    brickwork CNOT layer, 1024 shots, CAL_SAMPLES = 1024, mitigation 'none';
  * the SAME physical chains of the chip the real run used (read from its log), so the fake
    simulates the noise model of the very qubits the real run measured;
  * 110 epochs of QPU-in-the-loop AdamW, 1 job per epoch, with the optimizer settings found by
    search_optimizer_ibm_faker.py: lr 0.1 for 5 epochs then exponential down to 1.4e-3,
    betas (0.76, 0.99), gradient clip 2, and the EMA of the weights over the second half of
    the training as the trained network; polynomial refitted every 10 epochs;

Phases:
  1. calibration on the fake (1 job, 128 parameter sets);
  2. training on the fake (110 jobs);
  3. validation on the fake (1 job, 200 points x 8 replicas);
  4. prediction on the REAL QPU (1 job in job mode, 200 points x 8 replicas, raw circuits):
     u(x,Tfin) of the fake-trained network with the <Z> measured on the real chip,
     compared with FDM, with the fake validation and with the real-trained run.

Everything goes to data/quantum_ibm_faker/T_<Tfin>/ (tag 10_NQ10_<seed>_fake_berlin_HW110).

Uso (ambiente conda `qiskit`; conta IBM salva em ~/.qiskit):
    python run_quantum_ibm_faker.py                    # treina no fake e prediz no QPU real
    python run_quantum_ibm_faker.py --no-real          # só o fake (nenhum job na IBM)
    python run_quantum_ibm_faker.py --predict-only     # só a fase 4, com o modelo já treinado
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
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")  # no window: figures are only saved

import numpy as np
import qiskit
import qiskit_aer
import qiskit_ibm_runtime
import torch as tc
from qiskit import qpy

from Inverser_quantum_hardware_pinn import (
    FAKE_BACKENDS, MITIGATIONS, HardwareSurrogateLayer, PackedCircuit,
    causal_cones, cnot_gate_list, find_chains, fit_surrogate, get_backend, screen_chains,
)
from main.functions import (
    cnot_cosine_expectations, plot_loss, plot_solution, reference_at_time,
    relative_l2_error, repeated_cnot_network_matrix, sample_collocation_points, set_seed,
    solve_burgers_fd,
)
from regression_hardware_nqubits import INSTANCE
from run_quantum_hardware import (FakeChainCircuit, build_model,
    residual_loss, stencil_inputs,
)
# ---- fake QPU -------------------------------------------------------------------
BACKEND = "ibm_berlin"            # simulated with FAKE_BACKENDS[BACKEND]
MAX_CHAIN_SCORE_PER_QUBIT = 0.05
CHAIN_OUTLIER = 3.0               # 
SCHEDULING = None
PRECISION = 1 / 32                # per expectation value -> 1024 shots per circuit
CAL_SAMPLES = 1024                # calibration samples per output (~ chains x parameter sets)

# ---- physical parameters (same as run_quantumExact.ipynb) ---------------------
L, NU = 1.0, 0.01
HX, HT = L / 512, 1.0e-3
# ---- architecture ---------------------------------------------------------------
N_QUBITS = 10
PRE_NEURONS, PRE_DEPTH = 10, 1
POST_NEURONS, POST_DEPTH = 10, 1
# the hardware encodes with RY; for <Z> it is the same feature map as the notebook's 'rx'
INPUT_ROTATION_AXIS = "ry"
ANGLE_SCALE = None
# chain networks only (no SWAPs); brickwork has the smallest causal cone (<= 4 qubits)
FIXED_LAYER = "brickwork_cnot"
FIXED_LAYER_REPEATS = 1
MAX_CONE = 6                      # 3^6 = 729 coefficients per output
# ---- training (QPU in the loop) ---------------------------------------------------
N_COLLOCATION = 20
TFIN = 1.0
SEEDS = (0, 1, 2, 3, 4)           # trained together: their points share each QPU job
EPOCHS = 110
# AdamW with the best settings found by search_optimizer_ibm_faker.py (seed 0, 10 qubits, T=1:
# median MSE of u(x,1) on the fake 2.3e-3 over 3 runs, against 2.1e-2 of the old 0.1 -> 0.01 stairs)
LR0, LR_MIN, LR_HOLD = 0.1, 0.001375536953729357, 5   # 0.1 for 5 epochs, then exponential down to LR_MIN
BETAS = (0.7581225476161662, 0.99)
WEIGHT_DECAY = 2.1065109541876345e-06
GRAD_CLIP = 2.0
# EMA (Polyak) average of the weights from epoch EMA_START * epochs on; the averaged weights are
# the trained network (they average out the shot noise the QPU puts on every step)
EMA_DECAY, EMA_START = 0.8, 0.5
RECAL_EVERY = 10                  # refit the polynomial with the training measurements (0 = never)
COLLOCATION_DISTRIBUTION = "random"
N_EVALUATION_POINTS = 100


def learning_rate(epoch, epochs=EPOCHS):
    """LR of `epoch` (1-based): LR0 for LR_HOLD epochs, then exponential down to LR_MIN at the last one."""
    if epoch <= LR_HOLD:
        return LR0
    p = min((epoch - LR_HOLD - 1) / max(epochs - LR_HOLD - 1, 1), 1.0)
    return LR0 * (LR_MIN / LR0) ** p


def training_config(epochs):
    return dict(epochs=epochs, optimizer="AdamW", lr0=LR0, lr_min=LR_MIN, lr_hold=LR_HOLD, lr_decay="exponential",
                betas=BETAS, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, ema_decay=EMA_DECAY, ema_start=EMA_START,
                recal_every=RECAL_EVERY, collocation_distribution=COLLOCATION_DISTRIBUTION,
                n_evaluation_points=N_EVALUATION_POINTS, L=L, nu=NU, hx=HX, ht=HT)


def qpu_step(model, opt, xt, angles, q_qpu):
    """run_quantum_hardware.qpu_step with this file's GRAD_CLIP."""
    with tc.no_grad():
        q_poly = model.quantum(angles)
    model.quantum.offset = (tc.as_tensor(q_qpu, dtype=q_poly.dtype) - q_poly).repeat(5, 1)
    try:
        loss = residual_loss(model, xt)
    finally:
        model.quantum.offset = None
    opt.zero_grad()
    loss.backward()
    tc.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    return float(loss.detach())


from run_quantum_hardware_real import backend_snapshot

ROOT = _ROOT
BACKEND = "ibm_berlin"            # real QPU of the prediction; FakeBerlin for the training
MITIGATION = "none"               # as in the real run (and the fake runs the raw circuits)
OUT_DIR = ROOT / "data" / "quantum_ibm_faker"
# real-QPU run whose configuration (seed, chains) is reproduced and whose result is compared
REAL_RUN = ROOT / "data" / "quantum_forward_hardware_real" / "T_0.1" / "run_10_NQ10_0_ibm_berlin_none_HW110.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", default=BACKEND, choices=tuple(FAKE_BACKENDS), help="QPU real (e o seu fake)")
    p.add_argument("--real-run", default=str(REAL_RUN), help="run no QPU real a reproduzir (seed, cadeias)")
    p.add_argument("--seed", type=int, default=None, help="padrão: a seed do --real-run")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="épocas com o fake no loop")
    p.add_argument("--find-chains", action="store_true",
                   help="escolhe as cadeias pela calibração do fake em vez de usar as do --real-run")
    p.add_argument("--val-replicas", type=int, default=None,
                   help="réplicas por ponto na validação/predição (padrão: uma em cada cadeia)")
    p.add_argument("--no-real", action="store_true", help="não envia o job de predição ao QPU real")
    p.add_argument("--predict-only", action="store_true", help="só a predição no QPU real (modelo já treinado)")
    return p.parse_args()


def main():
    args = parse_args()
    n, n_col, Tfin = N_QUBITS, N_COLLOCATION, TFIN
    real_run = json.loads(Path(args.real_run).read_text()) if Path(args.real_run).exists() else None
    if real_run is None:
        print(f"AVISO: run real {args.real_run} não existe -- seed 0, cadeias escolhidas no fake, sem comparação.")
    seed = args.seed if args.seed is not None else (real_run["seed"] if real_run else 0)
    out_dir = OUT_DIR / f"T_{Tfin}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gates = cnot_gate_list(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    M = repeated_cnot_network_matrix(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    cones = causal_cones(n, gates)
    assert max(map(len, cones)) <= MAX_CONE

    fake = FAKE_BACKENDS[args.backend]()
    tag = f"{n_col}_NQ{n}_{seed}_{fake.name}_HW{args.epochs}"
    path = lambda kind, ext: out_dir / f"{kind}_{tag}.{ext}"
    log_path = path("run", "json")

    def save_log():
        log_path.write_text(json.dumps(log, indent=2, default=str))

    # ---------- FDM reference and evaluation grid ----------
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, N_EVALUATION_POINTS).reshape(-1, 1)
    u_ref_eval, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    xt_plot = tc.tensor(np.hstack([x_plot, np.full_like(x_plot, Tfin)]), dtype=tc.get_default_dtype())
    x_mon, t_mon = sample_collocation_points(n_col, L, Tfin, HT, distribution="uniform_exact")
    xt_mon = stencil_inputs(x_mon, t_mon)

    def errors(u):
        return dict(mse=float(np.mean((u - u_ref_eval) ** 2)), mae=float(np.mean(np.abs(u - u_ref_eval))),
                    relative_l2=relative_l2_error(u, u_ref_eval))

    if args.predict_only:
        if not path("model", "pt").exists():
            raise SystemExit(f"--predict-only: {path('model', 'pt')} não existe (rode o treino antes).")
        log = json.loads(log_path.read_text())
        saved = tc.load(path("model", "pt"), weights_only=False)
        model = build_model(n, Tfin)
        model.quantum = HardwareSurrogateLayer(cones, [np.asarray(c) for c in saved["coefs"]])
        model.load_state_dict(saved["state_dict"])
        chains = saved["chains"]
        fake_val = np.load(path("validation_fake", "npz"))
        u_fake = fake_val["u_qpu"]
    else:
        # ---------- chains: the ones of the real run (same physical qubits) ----------
        if real_run and not args.find_chains:
            chains = real_run["chains"]
            print(f"Cadeias do run real ({len(chains)}): mesmos qubits físicos no {fake.name}")
        else:
            chains, scores, _ = find_chains(fake, n)
            max_score = MAX_CHAIN_SCORE_PER_QUBIT * n
            chains = [c for c, s in zip(chains, scores) if s <= max_score]
            print(f"{len(chains)} cadeias de {n} qubits com nota ≤ {max_score:.2f} no {fake.name}")
        make_circuit = lambda chains: FakeChainCircuit(fake, chains, n, gates, PRECISION, SCHEDULING)
        hw = make_circuit(chains)
        shots = int(np.ceil(PRECISION ** -2))
        T_cal, T_epoch = math.ceil(CAL_SAMPLES / hw.B), math.ceil(n_col / hw.B)
        if hw.n_2q != hw.n_2q_expected:
            print(f"AVISO: esperado {hw.n_2q_expected} portas {hw.g2}: o transpilador inseriu SWAPs.")

        log = dict(
            started=datetime.now().isoformat(), fake_backend=fake.name, real_backend=args.backend, seed=seed,
            n_qubits=n, n_collocation=n_col, Tfin=Tfin, nu=NU, network=FIXED_LAYER,
            n_cnot_layers=FIXED_LAYER_REPEATS, cones=cones, training_config=training_config(args.epochs),
            mitigation=MITIGATION, scheduling=SCHEDULING, chains=chains, chains_from_real_run=bool(real_run) and not args.find_chains,
            blocks=hw.B, shots=shots, precision=PRECISION, cal_samples=CAL_SAMPLES, chain_outlier=CHAIN_OUTLIER,
            isa_fake=dict(depth=hw.depth, two_qubit_gate=hw.g2, n_2q=hw.n_2q, n_2q_expected=hw.n_2q_expected,
                          qubits_used=hw.n_qubits_used),
            real_run=dict(file=args.real_run, mse=real_run.get("mse"), surrogate=real_run.get("surrogate"),
                          ideal_circuit=real_run.get("ideal_circuit"), chains=real_run["chains"]) if real_run else None,
            fake_job_ids=[], recal=[],
            versions=dict(python=sys.version.split()[0], platform=platform.platform(), qiskit=qiskit.__version__,
                          qiskit_ibm_runtime=qiskit_ibm_runtime.__version__, qiskit_aer=qiskit_aer.__version__,
                          torch=tc.__version__, numpy=np.__version__),
            command=" ".join(sys.argv))
        print(f"Treino: {fake.name} (Aer density matrix) | seed {seed} | {hw.B} cadeias × {n} qubits | profundidade "
              f"ISA {hw.depth} | {hw.n_2q} portas {hw.g2} | {shots} shots | {args.epochs} épocas × {T_epoch} conjuntos")
        print(f"Saída em {out_dir}")

        set_seed(seed)                      # same initialization as the real run
        model = build_model(n, Tfin)
        generator = tc.Generator().manual_seed(seed)   # same collocation stream as the real run
        tc.save(dict(state_dict=model.state_dict(), config=model.config, seed=seed), path("params_initial", "pt"))
        with tc.no_grad():
            u_ideal0 = model(xt_plot).numpy()          # quantum = None: the exact formula
        t0 = time.time()

        # ---------- 1. calibration on the fake ----------
        print(f"1/4 calibração no fake: 1 job × {T_cal} conjuntos...", flush=True)
        rng = np.random.default_rng(seed)
        angle_sets = rng.uniform(-np.pi, np.pi, size=(T_cal, hw.B, n))
        t_job = time.time()
        evs = hw.run(angle_sets, log["fake_job_ids"], verbose=False)
        cal_angles, cal_evs = angle_sets.reshape(-1, n), evs.reshape(-1, n)
        np.savez(path("calibration", "npz"), angles=cal_angles, evs=cal_evs, chains=np.array(chains),
                 backend=fake.name)
        log.update(calibration_file=str(path("calibration", "npz")), calibration_time_s=time.time() - t_job)

        keep, misfit = screen_chains(cal_angles, cal_evs, hw.B, cones, M, CHAIN_OUTLIER)
        log["chain_misfit"] = misfit.tolist()
        if not keep.all():
            print(f"  cadeias descartadas pela calibração (desajuste > {CHAIN_OUTLIER} × mediana "
                  f"{np.median(misfit):.3f}): " + ", ".join(f"{chains[b]} ({misfit[b]:.3f})"
                                                           for b in np.flatnonzero(~keep)))
            rows = keep[np.arange(len(cal_angles)) % hw.B]
            cal_angles, cal_evs = cal_angles[rows], cal_evs[rows]
            log["dropped_chains"] = [c for c, k in zip(chains, keep) if not k]
            chains = [c for c, k in zip(chains, keep) if k]
            hw = make_circuit(chains)
            log.update(chains=chains, blocks=hw.B)

        coefs, fit_rms, contrast = fit_surrogate(cal_angles, cal_evs, cones, M)
        ideal = np.prod(np.where(M.astype(bool)[None], np.cos(cal_angles)[:, None, :], 1.0), axis=2)
        log.update(fit_rms=fit_rms, contrast=contrast,
                   rms_hw_vs_exact=float(np.sqrt(np.mean((cal_evs - ideal) ** 2))))
        print(f"  {log['calibration_time_s']:.0f} s | ajuste: RMS do resíduo {np.mean(fit_rms):.4f} (ruído de shot ≈ "
              f"{PRECISION:.4f}) | contraste {np.min(contrast):.3f}–{np.max(contrast):.3f} | RMS fake − fórmula "
              f"exata {log['rms_hw_vs_exact']:.4f}", flush=True)
        model.quantum = HardwareSurrogateLayer(cones, coefs)

        with tc.no_grad():
            u_sur0 = model(xt_plot).numpy()
        np.savez(path("initial", "npz"), x=x_plot, u_surrogate=u_sur0, u_ideal=u_ideal0, u_reference=u_ref_eval)
        plot_solution(x_plot, {"initial network, measured model": u_sur0, "initial network, ideal circuit": u_ideal0},
                      ref=u_ref_eval, save_path=path("initial", "png"))
        log["initial"] = dict(surrogate=errors(u_sur0), ideal_circuit=errors(u_ideal0),
                              monitor_loss=float(residual_loss(model, xt_mon).detach()))
        save_log()

        # ---------- 2. training with the fake QPU in the loop ----------
        hist = dict(lr=[], loss_qpu=[], loss_monitor=[log["initial"]["monitor_loss"]], x_f=[], t_f=[],
                    angles=[], evs=[], epoch_time_s=[])
        opt = tc.optim.AdamW(model.parameters(), lr=LR0, betas=BETAS, weight_decay=WEIGHT_DECAY)
        ema, ema_from = None, int(round(EMA_START * args.epochs)) + 1
        print(f"2/4 treino no fake: {args.epochs} épocas, AdamW lr {LR0} ({LR_HOLD} épocas) -> {LR_MIN:.1e} "
              f"(exponencial), β={BETAS[0]:.2f},{BETAS[1]}, clip {GRAD_CLIP}, EMA {EMA_DECAY} dos pesos desde a "
              f"época {ema_from}, polinômio reajustado a cada {RECAL_EVERY} épocas...", flush=True)
        t_hw = time.time()
        for epoch in range(1, args.epochs + 1):
            t_epoch = time.time()
            lr = learning_rate(epoch, args.epochs)
            for group in opt.param_groups:
                group["lr"] = lr
            model.train()
            x_f = tc.rand((n_col, 1), generator=generator) * L
            t_f = HT + tc.rand((n_col, 1), generator=generator) * (Tfin - 2.0 * HT)
            xt = stencil_inputs(x_f, t_f)
            with tc.no_grad():
                angles = model.angles(xt[:n_col])
            q_qpu, meas_angles, meas_evs = hw.evaluate(angles.numpy(), log["fake_job_ids"], shift=epoch, verbose=False)
            loss = qpu_step(model, opt, xt, angles, q_qpu)
            if epoch >= ema_from:
                with tc.no_grad():
                    if ema is None:
                        ema = {k: p.detach().clone() for k, p in model.named_parameters()}
                    else:
                        for k, p in model.named_parameters():
                            ema[k].mul_(EMA_DECAY).add_(p.detach(), alpha=1.0 - EMA_DECAY)
            with tc.no_grad():
                loss_mon = float(residual_loss(model, xt_mon))
            for key, value in dict(lr=lr, loss_qpu=loss, loss_monitor=loss_mon, x_f=x_f.numpy(), t_f=t_f.numpy(),
                                   angles=meas_angles, evs=meas_evs, epoch_time_s=time.time() - t_epoch).items():
                hist[key].append(value)
            if RECAL_EVERY and epoch % RECAL_EVERY == 0:
                coefs, fit_rms, _ = fit_surrogate(np.concatenate([cal_angles, *hist["angles"]]),
                                                  np.concatenate([cal_evs, *hist["evs"]]), cones, M)
                model.quantum.set_coefs(coefs)
                log["recal"].append(dict(epoch=epoch, fit_rms=fit_rms,
                                         n_samples=len(cal_angles) + sum(map(len, hist["angles"]))))
            if epoch % 10 == 0 or epoch == args.epochs:
                rate = (time.time() - t_hw) / epoch
                print(f"  época {epoch}/{args.epochs} | lr {lr:.2e} | loss fake {loss:.4e} | loss monitor "
                      f"{loss_mon:.4e} | {rate:.1f} s/época, faltam ~{rate * (args.epochs - epoch) / 60:.0f} min",
                      flush=True)
        log["training_time_s"] = time.time() - t_hw
        if ema is not None:                   # the trained network is the EMA of the weights
            with tc.no_grad():
                for k, p in model.named_parameters():
                    p.copy_(ema[k])
            with tc.no_grad():
                log["monitor_loss_ema"] = float(residual_loss(model, xt_mon))
        tc.save(dict(state_dict=model.state_dict(), config=model.config, seed=seed, cones=cones,
                     coefs=[c.tolist() for c in coefs], chains=chains, backend=fake.name), path("model", "pt"))
        np.savez(path("history", "npz"), lr=np.array(hist["lr"]), loss_qpu=np.array(hist["loss_qpu"]),
                 loss_monitor=np.array(hist["loss_monitor"]), x_f=np.array(hist["x_f"]), t_f=np.array(hist["t_f"]),
                 angles=np.array(hist["angles"]), evs=np.array(hist["evs"]),
                 epoch_time_s=np.array(hist["epoch_time_s"]), cal_angles=cal_angles, cal_evs=cal_evs)
        losses = {f"fake QPU ({fake.name}), training batch": hist["loss_qpu"],
                  "measured model, fixed grid": hist["loss_monitor"][1:]}
        real_hist = Path(args.real_run).parent / Path(args.real_run).name.replace("run_", "history_").replace(".json", ".npz")
        if real_hist.exists():
            losses[f"real QPU ({args.backend}), training batch"] = np.load(real_hist)["loss_qpu"]
        plot_loss(losses, title=f"Loss -- seed {seed}, {fake.name} vs {args.backend}, T={Tfin}",
                  save_path=path("loss", "png"))
        save_log()

        # ---------- 3. validation on the fake ----------
        R_val = args.val_replicas or hw.B
        print(f"3/4 validação no fake: {N_EVALUATION_POINTS} pontos × {R_val} réplicas...", flush=True)
        model.eval()
        with tc.no_grad():
            angles = model.angles(xt_plot)
        q_fake, _, _ = hw.evaluate(angles.numpy(), log["fake_job_ids"], replicas=R_val, verbose=False)
        with tc.no_grad():
            u_fake = model.from_features(xt_plot, q_fake).numpy()
        np.savez(path("validation_fake", "npz"), x=x_plot, angles=angles.numpy(), q_qpu=q_fake, u_qpu=u_fake,
                 u_reference=u_ref_eval, replicas=R_val)
        log["fake"] = errors(u_fake)
        log["fake_wall_time_s"] = time.time() - t0
        save_log()
        print(f"  MSE de u(x,{Tfin}) no fake {log['fake']['mse']:.3e} | antes do treino "
              f"{log['initial']['surrogate']['mse']:.3e}")

    # ---------- network without QPU: measured polynomial and ideal circuit ----------
    model.eval()
    with tc.no_grad():
        angles = model.angles(xt_plot)
        u_sur = model(xt_plot).numpy()
        q_sur = model.quantum(angles).numpy()
        q_exact = cnot_cosine_expectations(angles, model.cnot_matrix)
        u_ideal = model.from_features(xt_plot, q_exact).numpy()
        q_exact = q_exact.numpy()
    log.update(surrogate=errors(u_sur), ideal_circuit=errors(u_ideal))
    save_log()
    curves = {f"prediction on fake QPU ({fake.name})": u_fake}

    # ---------- 4. prediction on the REAL QPU ----------
    if args.no_real:
        print("--no-real: nenhum job enviado à IBM.")
    else:
        print("Predição no QPU real desativada.")
        # real = get_backend(argparse.Namespace(fake=False, backend=args.backend))
        # hw_real = PackedCircuit(real, chains, n, gates, PRECISION, False, MITIGATIONS[MITIGATION])
        # R_val = args.val_replicas or hw_real.B
        # T_val = math.ceil(N_EVALUATION_POINTS * R_val / hw_real.B)
        # print(f"4/4 predição no QPU real {real.name}: 1 job × {T_val} conjuntos ({N_EVALUATION_POINTS} pontos × "
        #       f"{R_val} réplicas) | {hw_real.B} cadeias, profundidade ISA {hw_real.depth}, {hw_real.n_2q} portas "
        #       f"{hw_real.g2} | mitigação '{MITIGATION}'", flush=True)
        # if hw_real.n_2q != hw_real.n_2q_expected:
        #     print(f"AVISO: esperado {hw_real.n_2q_expected} portas {hw_real.g2}: o transpilador inseriu SWAPs.")
        # path("backend_real", "json").write_text(json.dumps(backend_snapshot(real, chains, hw_real.g2), indent=2,
        #                                                    default=str))
        # with open(path("isa_real", "qpy"), "wb") as f:
        #     qpy.dump(hw_real.isa, f)
        # real_job_ids = []
        # t_job = time.time()
        # q_real, slot_angles, slot_evs = hw_real.evaluate(angles.numpy(), real_job_ids, replicas=R_val)
        # with tc.no_grad():
        #     u_real = model.from_features(xt_plot, q_real).numpy()
        # log["real_prediction"] = dict(
        #     backend=real.name, instance=INSTANCE, job_ids=real_job_ids, mitigation=MITIGATION,
        #     mitigation_options=MITIGATIONS[MITIGATION], chains=chains, replicas=R_val, parameter_sets=T_val,
        #     isa=dict(depth=hw_real.depth, two_qubit_gate=hw_real.g2, n_2q=hw_real.n_2q,
        #              n_2q_expected=hw_real.n_2q_expected, qubits_used=hw_real.n_qubits_used),
        #     time_s=time.time() - t_job, finished=datetime.now().isoformat(), **errors(u_real),
        #     rms_q_real_vs_fake_polynomial=float(np.sqrt(np.mean((q_real - q_sur) ** 2))),
        #     rms_q_real_vs_exact=float(np.sqrt(np.mean((q_real - q_exact) ** 2))))
        # np.savez(path("prediction_real", "npz"), x=x_plot, angles=angles.numpy(), q_real=q_real, q_surrogate=q_sur,
        #          q_exact=q_exact, u_real=u_real, u_fake=u_fake, u_surrogate=u_sur, u_ideal=u_ideal,
        #          u_reference=u_ref_eval, slot_angles=slot_angles, slot_evs=slot_evs, replicas=R_val,
        #          job_id=real_job_ids[-1])
        # curves = {f"prediction on real QPU ({real.name}), trained on {fake.name}": u_real, **curves}
        # save_log()

    plot_solution(x_plot, curves, ref=u_ref_eval, save_path=path("validation", "png"))
    log["finished"] = datetime.now().isoformat()
    save_log()

    print(f"u(x,{Tfin}) vs FDM (MSE): fake {log['fake']['mse']:.3e} | modelo medido {log['surrogate']['mse']:.3e} | "
          f"circuito ideal {log['ideal_circuit']['mse']:.3e}"
          + (f" | QPU real (treinado no fake) {log['real_prediction']['mse']:.3e}" if "real_prediction" in log else "")
          + (f" | QPU real (treinado no real) {log['real_run']['mse']:.3e}" if log.get("real_run") else ""))
    print(f"saída em {out_dir}")


if __name__ == "__main__":
    main()
