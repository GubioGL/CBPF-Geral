"""Forward hybrid PINN for viscous Burgers on a REAL IBM QPU -- the best seed of the
fake-QPU batch run of run_quantum_hardware.py, with the same parameters.

Same protocol as run_quantum_hardware.py (model, loss, calibration polynomial,
straight-through QPU-in-the-loop AdamW, learning-rate schedule, validation -- everything
is imported from there), for ONE seed:

  * seed: the one with the lowest MSE in the fake batch run (REFERENCE_RUN); its random
    initialization and its stream of random collocation points are reproduced exactly
    (set_seed(seed) and tc.Generator().manual_seed(seed), as in the batch run);
  * circuit: on the real chip all chains go in ONE packed circuit (PackedCircuit of
    Inverser_quantum_hardware_pinn.py); the chains are chosen on the CURRENT calibration
    of the device, so they are not the ones of the fake run;
  * mitigation: 'none' by default -- the raw circuits, as the fake (Aer) ran them;
  * every job (calibration, 110 epochs, validation) runs inside one Session.

Saved (data/quantum_forward_hardware_real/T_<Tfin>/, tag <n_col>_NQ<n>_<seed>_<backend>_<mitigation>_HW<epochs>):
  plan_<tag>.json            what will run: config, chains, shots, jobs, ISA circuit info,
                             reference fake run, package versions
  backend_<tag>.json         device snapshot (T1, T2, readout/sx/CZ errors of the chosen qubits
                             + backend.properties() when available)
  isa_<tag>.qpy              the transpiled ISA circuit(s) sent to the QPU
  params_initial_<tag>.pt    network before training;  model_<tag>.pt = trained network
  initial_<tag>.npz/.png     estimate of u(x,Tfin) BEFORE training (ideal circuit and measured
                             polynomial, no QPU job) -- written as soon as the calibration ends
  calibration_<tag>.npz      angles / <Z> of the calibration job
  checkpoint_<tag>.pt        everything, after every epoch (--resume continues from it)
  history_<tag>.npz          per epoch: lr, losses, collocation points, angles and <Z> measured
  validation_<tag>.npz/.png  u(x,Tfin) of the trained network on the QPU vs FDM
  run_<tag>.json             log: job ids, session, errors, timings, comparison with the fake run

Uso (ambiente conda `qiskit`; conta IBM salva em ~/.qiskit):
    python run_quantum_hardware_real.py --plan          # conecta, escolhe as cadeias, transpila: NENHUM job
    python run_quantum_hardware_real.py                 # roda no QPU real (BACKEND = ibm_berlin)
    python run_quantum_hardware_real.py --resume        # continua do checkpoint (Session expirou, job falhou...)
    python run_quantum_hardware_real.py --fake --epochs 3   # ensaio local do script inteiro com FakeBerlin
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
from contextlib import nullcontext
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
from qiskit_ibm_runtime import Session

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
from run_quantum_hardware import (
    ANGLE_SCALE, CAL_SAMPLES, CHAIN_OUTLIER, COLLOCATION_DISTRIBUTION, EPOCHS, FIXED_LAYER,
    FIXED_LAYER_REPEATS, GRAD_CLIP, HT, HX, INPUT_ROTATION_AXIS, L, LR0, LR_EVERY, LR_MIN, LR_STEP,
    MAX_CHAIN_SCORE_PER_QUBIT, MAX_CONE, N_COLLOCATION, N_EVALUATION_POINTS, N_QUBITS, NU,
    POST_DEPTH, POST_NEURONS, PRE_DEPTH, PRE_NEURONS, PRECISION, RECAL_EVERY, SCHEDULING, TFIN,
    WEIGHT_DECAY, FakeChainCircuit, build_model, learning_rate, qpu_step, residual_loss, stencil_inputs,
)

ROOT = _ROOT
BACKEND = "ibm_berlin"            # the device FakeBerlin models (the reference run)
MITIGATION = "none"               # the fake ran the raw circuits; see MITIGATIONS for the others
SESSION_MAX_TIME = "1h"           # safety limit of the Session (110 epochs ~ 10-30 min of jobs)
OUT_DIR = ROOT / "data" / "quantum_forward_hardware_real"
# fake-QPU batch run whose best seed is reproduced here
REFERENCE_RUN = ROOT / "data" / "quantum_forward_hardware" / "T_0.1" / "run_10_NQ10_seeds0-4_fake_berlin_HW110.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", default=BACKEND, help="QPU da IBM ('least_busy': o menos ocupado)")
    p.add_argument("--fake", action="store_true", help="ensaio local com o fake do --backend (nenhum job na IBM)")
    p.add_argument("--plan", action="store_true", help="só prepara e mostra o que vai rodar: nenhum job")
    p.add_argument("--resume", action="store_true", help="continua do checkpoint desta configuração")
    p.add_argument("--reference", default=str(REFERENCE_RUN), help="run do lote no fake (escolhe a melhor seed)")
    p.add_argument("--seed", type=int, default=None, help="padrão: a melhor seed de --reference")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="épocas com o QPU no loop (1 job cada)")
    p.add_argument("--mitigation", choices=tuple(MITIGATIONS), default=MITIGATION,
                   help="supressão/mitigação do Estimator (mesma em todos os jobs)")
    p.add_argument("--blocks", type=int, default=None, help="cadeias usadas (padrão: todas que cabem)")
    p.add_argument("--max-chain-score", type=float, default=MAX_CHAIN_SCORE_PER_QUBIT * N_QUBITS,
                   help="descarta cadeias com nota de erro acima disso")
    p.add_argument("--val-replicas", type=int, default=None,
                   help="réplicas por ponto na validação (padrão: uma em cada cadeia)")
    p.add_argument("--max-time", default=SESSION_MAX_TIME, help="tempo máximo da Session (trava de segurança)")
    p.add_argument("--no-session", action="store_true", help="job mode: cada job entra na fila (contas sem Session)")
    args = p.parse_args()
    if args.fake:
        args.mitigation = "none"           # Aer runs the raw circuits
    return args


def load_reference(path, seed):
    """Best (or the given) seed of the fake batch run, with its results and files."""
    path = Path(path)
    if not path.exists():
        print(f"AVISO: run de referência {path} não existe -- seed {seed or 0}, sem comparação com o fake.")
        return seed or 0, None
    ref = json.loads(path.read_text())
    runs = {r["seed"]: r for r in ref["runs"]}
    if seed is None:
        seed = min(runs, key=lambda s: runs[s]["mse"])
    if seed not in runs:
        print(f"AVISO: seed {seed} não está no run de referência -- sem comparação com o fake.")
        return seed, None
    # this script must reproduce the same training; a mismatch means a constant changed
    tc_ref, tc_now = ref["training_config"], training_config(ref["training_config"]["epochs"])
    diff = {k: (tc_ref[k], tc_now[k]) for k in tc_now if k in tc_ref and tc_ref[k] != tc_now[k]}
    if diff:
        print(f"AVISO: parâmetros diferentes do run de referência (referência, agora): {diff}")
    i = ref["seeds"].index(seed)
    stem = path.stem.replace("run_", "").replace(f"seeds{ref['seeds'][0]}-{ref['seeds'][-1]}", str(seed))
    return seed, dict(file=str(path), backend=ref["backend"], seed=seed, index=i, run=runs[seed],
                      chains=ref["chains"], training_config=tc_ref, mse_all_seeds={s: r["mse"] for s, r in runs.items()},
                      validation_file=str(path.parent / f"validation_{stem}.npz"),
                      losshist_file=str(path.parent / f"losshist_{path.stem.replace('run_', '')}.npz"))


def training_config(epochs):
    return dict(epochs=epochs, lr0=LR0, lr_step=LR_STEP, lr_every=LR_EVERY, lr_min=LR_MIN,
                weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, recal_every=RECAL_EVERY,
                collocation_distribution=COLLOCATION_DISTRIBUTION,
                n_evaluation_points=N_EVALUATION_POINTS, L=L, nu=NU, hx=HX, ht=HT)


def backend_snapshot(backend, chains, g2):
    """Calibration data of the qubits / couplers used, as the backend reports them now."""
    target = backend.target

    def err(gate, qargs):
        try:
            return target[gate][qargs].error
        except KeyError:
            return None

    qubits = {}
    for q in sorted({q for chain in chains for q in chain}):
        props = target.qubit_properties[q] if target.qubit_properties else None
        qubits[q] = dict(t1=getattr(props, "t1", None), t2=getattr(props, "t2", None),
                         frequency=getattr(props, "frequency", None),
                         readout_error=err("measure", (q,)), sx_error=err("sx", (q,)))
    couplers = {f"{a}-{b}": err(g2, (a, b)) if err(g2, (a, b)) is not None else err(g2, (b, a))
                for chain in chains for a, b in zip(chain, chain[1:])}
    snap = dict(name=backend.name, num_qubits=backend.num_qubits, dt=getattr(target, "dt", None),
                operation_names=sorted(target.operation_names), two_qubit_gate=g2,
                qubits=qubits, couplers=couplers, snapshot_time=datetime.now().isoformat())
    try:
        props = backend.properties()
        if props is not None:
            snap["last_update_date"] = str(props.last_update_date)
            snap["properties"] = props.to_dict()
    except Exception as e:          # not every backend / version exposes properties()
        snap["properties_error"] = repr(e)
    return snap


def main():
    args = parse_args()
    n, n_col, Tfin = N_QUBITS, N_COLLOCATION, TFIN
    seed, reference = load_reference(args.reference, args.seed)
    out_dir = OUT_DIR / f"T_{Tfin}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gates = cnot_gate_list(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    M = repeated_cnot_network_matrix(n, FIXED_LAYER, FIXED_LAYER_REPEATS)
    cones = causal_cones(n, gates)
    assert max(map(len, cones)) <= MAX_CONE

    backend = get_backend(args)
    options = MITIGATIONS[args.mitigation]
    tag = f"{n_col}_NQ{n}_{seed}_{backend.name}_{args.mitigation}_HW{args.epochs}"
    path = lambda kind, ext: out_dir / f"{kind}_{tag}.{ext}"
    ckpt_path = path("checkpoint", "pt")

    # ---------- chains: from the checkpoint, or the best ones on the device now ----------
    ckpt = None
    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"--resume: checkpoint {ckpt_path} não existe.")
        ckpt = tc.load(ckpt_path, weights_only=False)
        chains = ckpt["chains"]
        print(f"Retomando de {ckpt_path} (época {ckpt['epoch']}/{args.epochs})")
    else:
        chains, scores, _ = find_chains(backend, n)
        kept = [c for c, s in zip(chains, scores) if s <= args.max_chain_score]   # sorted best first
        print(f"{len(kept)} de {len(chains)} cadeias de {n} qubits com nota ≤ {args.max_chain_score:.2f} "
              f"(descartadas: {', '.join(f'{s:.3f}' for s in scores[len(kept):]) or 'nenhuma'})")
        chains = kept[:args.blocks] if args.blocks else kept
        if not chains:
            raise SystemExit(f"Nenhuma cadeia de {n} qubits conectados em {backend.name}.")

    def make_circuit(chains):
        if args.fake:
            return FakeChainCircuit(backend, chains, n, gates, PRECISION, SCHEDULING)
        return PackedCircuit(backend, chains, n, gates, PRECISION, False, options)

    hw = make_circuit(chains)
    shots = int(np.ceil(PRECISION ** -2))
    R_val = args.val_replicas or hw.B
    T_cal, T_epoch = math.ceil(CAL_SAMPLES / hw.B), math.ceil(n_col / hw.B)
    T_val = math.ceil(N_EVALUATION_POINTS * R_val / hw.B)
    n_sets = T_cal + args.epochs * T_epoch + T_val
    if hw.n_2q != hw.n_2q_expected:
        print(f"AVISO: esperado {hw.n_2q_expected} portas {hw.g2}: o transpilador inseriu SWAPs.")

    if ckpt:
        log = ckpt["log"]
    else:
        log = dict(
            started=datetime.now().isoformat(), backend=backend.name, fake=args.fake, instance=None if args.fake else INSTANCE,
            seed=seed, n_qubits=n, n_collocation=n_col, Tfin=Tfin, nu=NU,
            network=FIXED_LAYER, n_cnot_layers=FIXED_LAYER_REPEATS, cones=cones,
            architecture=dict(pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH, post_neurons=POST_NEURONS,
                              post_depth=POST_DEPTH, input_rotation_axis=INPUT_ROTATION_AXIS, angle_scale=ANGLE_SCALE),
            training_config=training_config(args.epochs),
            mitigation=args.mitigation, mitigation_options=None if args.fake else options,
            chains=chains, blocks=hw.B, shots=shots, precision=PRECISION, cal_samples=CAL_SAMPLES,
            max_chain_score=args.max_chain_score, chain_outlier=CHAIN_OUTLIER,
            isa=dict(depth=hw.depth, two_qubit_gate=hw.g2, n_2q=hw.n_2q, n_2q_expected=hw.n_2q_expected,
                     qubits_used=hw.n_qubits_used),
            jobs_planned=dict(calibration=dict(jobs=1, parameter_sets=T_cal),
                              training=dict(jobs=args.epochs, parameter_sets_per_job=T_epoch),
                              validation=dict(jobs=1, parameter_sets=T_val, replicas=R_val),
                              total_parameter_sets=n_sets, total_shots=n_sets * shots),
            session=dict(use=not args.fake and not args.no_session, max_time=args.max_time, ids=[]),
            reference_fake_run=reference, job_ids=[], recal=[], resumes=[],
            versions=dict(python=sys.version.split()[0], platform=platform.platform(), qiskit=qiskit.__version__,
                          qiskit_ibm_runtime=qiskit_ibm_runtime.__version__, qiskit_aer=qiskit_aer.__version__,
                          torch=tc.__version__, numpy=np.__version__),
            command=" ".join(sys.argv))
    log_path = path("run", "json")

    def save_log():
        log_path.write_text(json.dumps(log, indent=2, default=str))

    print(f"Backend: {backend.name}{' (fake)' if args.fake else ''} | seed {seed}"
          + (f" (melhor do fake: MSE {reference['run']['mse']:.3e})" if reference else ""))
    print(f"{hw.B} cadeias × {n} qubits = {hw.n_qubits_used} de {backend.num_qubits} qubits | profundidade ISA "
          f"{hw.depth} | {hw.n_2q} portas {hw.g2} | mitigação '{args.mitigation}' | {shots} shots")
    print(f"Jobs: calibração 1 × {T_cal} conjuntos | treino {args.epochs} × {T_epoch} conjuntos | validação 1 × "
          f"{T_val} conjuntos ({N_EVALUATION_POINTS} pontos × {R_val} réplicas) -> {args.epochs + 2} jobs, "
          f"{n_sets} conjuntos × {shots} shots")
    print(f"lr {LR0} -> {LR_MIN} (−{LR_STEP} a cada {LR_EVERY} épocas), polinômio reajustado a cada "
          f"{RECAL_EVERY} épocas | saída em {out_dir}")

    # ---------- network and reference ----------
    set_seed(seed)                      # same initialization as this seed in the fake batch run
    model = build_model(n, Tfin)
    generator = tc.Generator().manual_seed(seed)   # same collocation stream as in the batch run
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, N_EVALUATION_POINTS).reshape(-1, 1)
    u_ref_eval, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    xt_plot = tc.tensor(np.hstack([x_plot, np.full_like(x_plot, Tfin)]), dtype=tc.get_default_dtype())
    x_mon, t_mon = sample_collocation_points(n_col, L, Tfin, HT, distribution="uniform_exact")
    xt_mon = stencil_inputs(x_mon, t_mon)

    def errors(u):
        return dict(mse=float(np.mean((u - u_ref_eval) ** 2)), mae=float(np.mean(np.abs(u - u_ref_eval))),
                    relative_l2=relative_l2_error(u, u_ref_eval))

    if not ckpt:
        tc.save(dict(state_dict=model.state_dict(), config=model.config, seed=seed), path("params_initial", "pt"))
        path("plan", "json").write_text(json.dumps(log, indent=2, default=str))
        path("backend", "json").write_text(json.dumps(backend_snapshot(backend, chains, hw.g2), indent=2, default=str))
        with open(path("isa", "qpy"), "wb") as f:
            qpy.dump(hw.isa, f)
        with tc.no_grad():
            u_ideal0 = model(xt_plot).numpy()          # quantum = None: the exact formula
        print(f"Rede inicial (circuito ideal): MSE de u(x,{Tfin}) {errors(u_ideal0)['mse']:.3e}")
    if args.plan:
        plot_solution(x_plot, {"initial network, ideal circuit": u_ideal0}, ref=u_ref_eval,
                      save_path=path("initial_plan", "png"))
        print(f"--plan: nada foi enviado ao QPU. Plano em {path('plan', 'json')}")
        return

    t0 = time.time()
    use_session = log["session"]["use"]
    with (Session(backend=backend, max_time=args.max_time) if use_session else nullcontext()) as session:
        if use_session:
            hw.set_mode(session)
            log["session"]["ids"].append(session.session_id)
            print("Session ID:", session.session_id, flush=True)   # dá para acompanhar no dashboard da IBM
        save_log()

        if ckpt:
            # ---------- resume: calibration and the first epochs are in the checkpoint ----------
            model.quantum = HardwareSurrogateLayer(cones, ckpt["coefs"])
            model.load_state_dict(ckpt["model"])
            generator.set_state(ckpt["generator"])
            cal_angles, cal_evs, coefs = ckpt["cal_angles"], ckpt["cal_evs"], ckpt["coefs"]
            hist, start = ckpt["hist"], ckpt["epoch"]
            log["resumes"].append(dict(epoch=start, time=datetime.now().isoformat()))
        else:
            # ---------- 1. calibration ----------
            print(f"1/3 calibração: 1 job × {T_cal} conjuntos...", flush=True)
            rng = np.random.default_rng(seed)
            angle_sets = rng.uniform(-np.pi, np.pi, size=(T_cal, hw.B, n))
            t_job = time.time()
            evs = hw.run(angle_sets, log["job_ids"])
            cal_angles, cal_evs = angle_sets.reshape(-1, n), evs.reshape(-1, n)
            np.savez(path("calibration", "npz"), angles=cal_angles, evs=cal_evs, chains=np.array(chains),
                     backend=backend.name, mitigation=args.mitigation, job_id=log["job_ids"][-1])
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
                if use_session:
                    hw.set_mode(session)
                R_val = args.val_replicas or hw.B
                log.update(chains=chains, blocks=hw.B)

            coefs, fit_rms, contrast = fit_surrogate(cal_angles, cal_evs, cones, M)
            ideal = np.prod(np.where(M.astype(bool)[None], np.cos(cal_angles)[:, None, :], 1.0), axis=2)
            log.update(fit_rms=fit_rms, contrast=contrast,
                       rms_hw_vs_exact=float(np.sqrt(np.mean((cal_evs - ideal) ** 2))))
            print(f"  {log['calibration_time_s']:.0f} s | ajuste: RMS do resíduo {np.mean(fit_rms):.4f} (ruído de "
                  f"shot ≈ {PRECISION:.4f}) | contraste do termo ideal {np.min(contrast):.3f}–{np.max(contrast):.3f} "
                  f"(ideal = 1) | RMS hardware − fórmula exata {log['rms_hw_vs_exact']:.4f}", flush=True)
            model.quantum = HardwareSurrogateLayer(cones, coefs)

            # ---------- estimate of the solution BEFORE training (no QPU job) ----------
            with tc.no_grad():
                angles0 = model.angles(xt_plot)
                u_sur0 = model(xt_plot).numpy()
            curves0 = {"initial network, measured model": u_sur0, "initial network, ideal circuit": u_ideal0}
            np.savez(path("initial", "npz"), x=x_plot, angles=angles0.numpy(), u_surrogate=u_sur0,
                     u_ideal=u_ideal0, u_reference=u_ref_eval)
            plot_solution(x_plot, curves0, ref=u_ref_eval, save_path=path("initial", "png"))
            log["initial"] = dict(surrogate=errors(u_sur0), ideal_circuit=errors(u_ideal0),
                                  monitor_loss=float(residual_loss(model, xt_mon).detach()))
            print(f"  antes do treino: MSE de u(x,{Tfin}) modelo medido {log['initial']['surrogate']['mse']:.3e} | "
                  f"circuito ideal {log['initial']['ideal_circuit']['mse']:.3e} -> {path('initial', 'png').name}")
            hist = dict(lr=[], loss_qpu=[], loss_monitor=[log["initial"]["monitor_loss"]], x_f=[], t_f=[],
                        angles=[], evs=[], job_ids=[], epoch_time_s=[])
            start = 0

        # ---------- 2. training with the QPU in the loop ----------
        opt = tc.optim.AdamW(model.parameters(), lr=LR0, weight_decay=WEIGHT_DECAY)
        if ckpt:
            opt.load_state_dict(ckpt["opt"])

        def save_checkpoint(epoch):
            tc.save(dict(epoch=epoch, model=model.state_dict(), opt=opt.state_dict(),
                         generator=generator.get_state(), coefs=coefs, cal_angles=cal_angles,
                         cal_evs=cal_evs, hist=hist, chains=chains, log=log), ckpt_path)
            save_log()

        if not ckpt:
            save_checkpoint(0)
        if start < args.epochs:
            print(f"2/3 treino: épocas {start + 1}–{args.epochs}, 1 job por época...", flush=True)
        t_hw = time.time()
        for epoch in range(start + 1, args.epochs + 1):
            t_epoch = time.time()
            lr = learning_rate(epoch)
            for group in opt.param_groups:
                group["lr"] = lr
            model.train()
            x_f = tc.rand((n_col, 1), generator=generator) * L
            t_f = HT + tc.rand((n_col, 1), generator=generator) * (Tfin - 2.0 * HT)
            xt = stencil_inputs(x_f, t_f)
            with tc.no_grad():
                angles = model.angles(xt[:n_col])
            q_qpu, meas_angles, meas_evs = hw.evaluate(angles.numpy(), log["job_ids"], shift=epoch, verbose=False)
            loss = qpu_step(model, opt, xt, angles, q_qpu)
            with tc.no_grad():
                loss_mon = float(residual_loss(model, xt_mon))
            for key, value in dict(lr=lr, loss_qpu=loss, loss_monitor=loss_mon, x_f=x_f.numpy(), t_f=t_f.numpy(),
                                   angles=meas_angles, evs=meas_evs, job_ids=log["job_ids"][-1],
                                   epoch_time_s=time.time() - t_epoch).items():
                hist[key].append(value)

            if RECAL_EVERY and epoch % RECAL_EVERY == 0:
                coefs, fit_rms, _ = fit_surrogate(np.concatenate([cal_angles, *hist["angles"]]),
                                                  np.concatenate([cal_evs, *hist["evs"]]), cones, M)
                model.quantum.set_coefs(coefs)
                log["recal"].append(dict(epoch=epoch, fit_rms=fit_rms,
                                         n_samples=len(cal_angles) + sum(map(len, hist["angles"]))))
            save_checkpoint(epoch)
            rate = (time.time() - t_hw) / (epoch - start)
            print(f"  época {epoch}/{args.epochs} | lr {lr:.2f} | loss QPU {loss:.4e} | loss monitor {loss_mon:.4e} "
                  f"| job {log['job_ids'][-1]} | {hist['epoch_time_s'][-1]:.1f} s, faltam "
                  f"~{rate * (args.epochs - epoch) / 60:.0f} min", flush=True)
        log["training_time_s"] = log.get("training_time_s", 0.0) + time.time() - t_hw
        tc.save(dict(state_dict=model.state_dict(), config=model.config, seed=seed, cones=cones,
                     coefs=[c.tolist() for c in coefs], chains=chains, backend=backend.name), path("model", "pt"))
        save_log()

        # ---------- 3. validation: the trained network running on the QPU ----------
        print(f"3/3 validação: 1 job × {T_val} conjuntos ({N_EVALUATION_POINTS} pontos × {R_val} réplicas)...",
              flush=True)
        model.eval()
        with tc.no_grad():
            angles = model.angles(xt_plot)
            u_sur = model(xt_plot).numpy()
            q_sur = model.quantum(angles).numpy()
            q_exact = cnot_cosine_expectations(angles, model.cnot_matrix)
            u_ideal = model.from_features(xt_plot, q_exact).numpy()
            q_exact = q_exact.numpy()
        q_qpu, val_angles, val_evs = hw.evaluate(angles.numpy(), log["job_ids"], replicas=R_val)
        if use_session:
            try:
                log["session"]["details"] = session.details()
                log["session"]["usage_s"] = session.usage()
            except Exception as e:
                log["session"]["details_error"] = repr(e)
    with tc.no_grad():
        u_qpu = model.from_features(xt_plot, q_qpu).numpy()

    initial = np.load(path("initial", "npz"))
    log.update(errors(u_qpu), surrogate=errors(u_sur), ideal_circuit=errors(u_ideal),
               rms_q_qpu_vs_surrogate=float(np.sqrt(np.mean((q_qpu - q_sur) ** 2))),
               rms_q_qpu_vs_exact=float(np.sqrt(np.mean((q_qpu - q_exact) ** 2))),
               final_loss_qpu=hist["loss_qpu"][-1], final_loss_monitor=hist["loss_monitor"][-1],
               finished=datetime.now().isoformat(), wall_time_s=log.get("wall_time_s", 0.0) + time.time() - t0)
    save_log()

    # ---------- outputs ----------
    np.savez(path("validation", "npz"), x=x_plot, angles=angles.numpy(), q_qpu=q_qpu, q_surrogate=q_sur,
             q_exact=q_exact, u_qpu=u_qpu, u_surrogate=u_sur, u_ideal=u_ideal, u_reference=u_ref_eval,
             u_initial_surrogate=initial["u_surrogate"], u_initial_ideal=initial["u_ideal"],
             slot_angles=val_angles, slot_evs=val_evs, replicas=R_val)
    np.savez(path("history", "npz"), lr=np.array(hist["lr"]), loss_qpu=np.array(hist["loss_qpu"]),
             loss_monitor=np.array(hist["loss_monitor"]), x_f=np.array(hist["x_f"]), t_f=np.array(hist["t_f"]),
             angles=np.array(hist["angles"]), evs=np.array(hist["evs"]),
             job_ids=np.array(hist["job_ids"]), epoch_time_s=np.array(hist["epoch_time_s"]),
             cal_angles=cal_angles, cal_evs=cal_evs)

    curves = {f"trained, QPU ({backend.name})": u_qpu, "trained, measured model": u_sur,
              "trained, ideal circuit": u_ideal}
    losses = {f"QPU ({backend.name}), training batch": hist["loss_qpu"],
              "measured model, fixed grid": hist["loss_monitor"][1:]}
    if reference and Path(reference["validation_file"]).exists():
        curves[f"fake QPU ({reference['backend']}), same seed"] = np.load(reference["validation_file"])["u_qpu"]
        log["reference_fake_mse"] = reference["run"]["mse"]
    if reference and Path(reference["losshist_file"]).exists():
        losses[f"fake QPU ({reference['backend']}), same seed"] = np.load(reference["losshist_file"])["loss_qpu"][:, reference["index"]]
    plot_solution(x_plot, curves, ref=u_ref_eval, save_path=path("validation", "png"), collocation_x=hist["x_f"][-1])
    plot_solution(x_plot, {"before training (measured model)": initial["u_surrogate"],
                           f"after training (QPU {backend.name})": u_qpu}, ref=u_ref_eval,
                  save_path=path("before_after", "png"))
    plot_loss(losses, title=f"Loss -- seed {seed}, {backend.name}, T={Tfin}", save_path=path("loss", "png"))
    save_log()

    print(f"u(x,{Tfin}) vs FDM: MSE antes {log['initial']['surrogate']['mse']:.3e} -> depois: QPU {log['mse']:.3e} | "
          f"modelo medido {log['surrogate']['mse']:.3e} | circuito ideal {log['ideal_circuit']['mse']:.3e}"
          + (f" | fake (mesma seed) {reference['run']['mse']:.3e}" if reference else ""))
    print(f"L2 relativo QPU {log['relative_l2']:.3e} | RMS <Z>: QPU vs modelo medido "
          f"{log['rms_q_qpu_vs_surrogate']:.4f}, vs fórmula exata {log['rms_q_qpu_vs_exact']:.4f}")
    print(f"tempo total {log['wall_time_s']:.0f} s | {len(log['job_ids'])} jobs | saída em {out_dir}")


if __name__ == "__main__":
    main()
