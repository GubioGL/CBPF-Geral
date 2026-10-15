"""Inverse hybrid PINN for viscous Burgers with the quantum layer on an IBM QPU, trained in
EPOCHS = 110 epochs with the QPU in the loop: learns nu jointly with u(x,t).

The training is the one of Inverser_quantum_exact_pinn_E110.py (same CONFIGS, same `train`:
AdamW for the network, varpro or Adam for log nu, fixed or gradient-norm loss weights), and
the hardware machinery the one of Inverser_quantum_hardware_pinn.py:
  1. calibration (1 job per qubit count, cached): random angles on every n-qubit chain of the
     chip -> least squares for the measured response, a degree-1 trigonometric polynomial
     in each angle of the causal cone (HardwareSurrogateLayer);
  2. training (1 job per epoch): the <Z> of the 10 collocation points and of the 100 data
     points are measured on the QPU at the current angles; the forward pass uses the QPU
     values and the gradient goes through the polynomial (straight-through),
         q(phi) = poly(phi) + stopgrad(q_QPU(phi_c) - poly(phi_c)),
     with the same correction on the 5 stencil points of collocation point c. Every
     RECAL_EVERY epochs the polynomial is refitted with the calibration + all measurements;
  3. validation (1 job): u(x, TFIN) of the trained network running on the QPU, compared with
     the exact (Cole-Hopf) solution.

Fake QPU (default): the noise model of BACKEND simulated locally with Aer (FakeChainCircuit,
one n-qubit density-matrix circuit per chain, shots sampled), no IBM job. Real QPU (--real):
PackedCircuit in a Session -- not run in the verification of this script.

Usage (conda env `qiskit`):
    python Inverser_quantum_hardware_pinn_E110.py                     # fake: 3 configs x 3/7/10 qubits x seeds 0/1/2
    python Inverser_quantum_hardware_pinn_E110.py --config gradnorm_varpro --qubits 10 --seeds 0
    python Inverser_quantum_hardware_pinn_E110.py --real --config gradnorm_varpro --qubits 10 --seeds 0
Outputs: OUT_DIR/T_<TFIN>/<backend>/<config>/ and the calibrations in OUT_DIR/T_<TFIN>/<backend>/.
"""
import argparse
import json
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch as tc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantum_hardware"))
from qiskit_ibm_runtime import QiskitRuntimeService, Session  # noqa: E402

from Inverser_quantum_exact_pinn_E110 import (  # noqa: E402  -- the problem and the training
    CONFIGS, EPOCHS, N_COLLOCATION, NU_TRUE, QUBITS, SEEDS, TFIN, plot_nu_summary, problem_setup, train,
)
from Inverser_quantum_exact_pinn_paralelo import (  # noqa: E402  -- the model
    FIXED_LAYER, FIXED_LAYER_REPEATS, INPUT_ROTATION_AXIS, L, ExactQuantumLayer, build_model,
    exact_solution, plot_loss, plot_parameter_history, plot_solution, relative_l2_error, set_seed,
)
from Inverser_quantum_hardware_pinn import (  # noqa: E402  -- the hardware machinery
    FAKE_BACKENDS, MITIGATION, MITIGATIONS, HardwareSurrogateLayer, PackedCircuit, causal_cones,
    cnot_gate_list, find_chains, fit_surrogate, network_angles, network_output, screen_chains,
)
from main.functions import repeated_cnot_network_matrix  # noqa: E402
from regression_hardware_nqubits import INSTANCE  # noqa: E402
from run_quantum_hardware import FakeChainCircuit  # noqa: E402

# ---- QPU ----------------------------------------------------------------------
BACKEND = "ibm_berlin"            # the fake is FAKE_BACKENDS[BACKEND]
PRECISION = 1 / 32                # per expectation value -> 1024 shots per circuit
CAL_SAMPLES = 1024                # calibration samples per output (~ chains x parameter sets)
MAX_CHAIN_SCORE_PER_QUBIT = 0.05  # chains with a worse error score (find_chains) are dropped
CHAIN_OUTLIER = 3.0               # ... and so are those with calibration misfit > 3 x the median
RECAL_EVERY = 10                  # refit the polynomial with the training measurements
VAL_POINTS = 100                  # validation: x points at t = TFIN, one replica per chain
SESSION_MAX_TIME = "2h"           # --real: safety limit of the Session

OUT_DIR = os.path.join("data", "Inverso", "quantum_inverse_hardware_E110")


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


def run_one(config_name, n_qubits, seed, real=False, threads=0, epochs=EPOCHS):
    """One run: training with the QPU in the loop + validation on the QPU. Same initialization
    and collocation stream as Inverser_quantum_exact_pinn_E110.run_one."""
    tc.set_num_threads(1)
    t0 = time.time()
    cfg = CONFIGS[config_name]
    backend = get_backend(real)
    base_dir = os.path.join(OUT_DIR, f"T_{TFIN}", backend.name)
    chains, cal_angles, cal_evs = calibrate(backend, n_qubits, real, base_dir)
    M, cones = circuit_structure(n_qubits)
    qpu = make_qpu(backend, chains, n_qubits, real, threads)
    coefs, fit_rms, _ = fit_surrogate(cal_angles, cal_evs, cones, M)

    set_seed(seed)
    nu_init = float(np.random.random())
    model = build_model(n_qubits, TFIN, nu_init)
    model.quantum = HardwareSurrogateLayer(cones, coefs)
    job_ids, meas_angles, meas_evs = [], [cal_angles], [cal_evs]

    def measure(epoch, xt):
        """<Z> on the QPU at the stencil centres and data points -> offset of the 5 stencil
        points of each centre and of the data points."""
        if epoch and epoch % RECAL_EVERY == 0:
            model.quantum.set_coefs(fit_surrogate(np.concatenate(meas_angles), np.concatenate(meas_evs), cones, M)[0])
        centres = tc.cat([xt[:N_COLLOCATION], xt[5 * N_COLLOCATION:]])
        with tc.no_grad():
            angles = network_angles(model, centres[:, :1], centres[:, 1:])
            q_poly = model.quantum(angles)
        q_qpu, slot_angles, slot_evs = qpu.evaluate(angles.numpy(), job_ids, shift=epoch, verbose=False)
        meas_angles.append(slot_angles)
        meas_evs.append(slot_evs)
        delta = tc.as_tensor(q_qpu, dtype=q_poly.dtype) - q_poly
        return tc.cat([delta[:N_COLLOCATION].repeat(5, 1), delta[N_COLLOCATION:]])

    with (Session(backend=backend, max_time=SESSION_MAX_TIME) if real else nullcontext()) as session:
        if real:
            qpu.set_mode(session)
        hist = train(model, cfg, seed, problem_setup(), epochs=epochs, measure=measure)

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
    u_ref = exact_solution(x_val, TFIN, nu=NU_TRUE)

    nu_learned = float(model.learnable_parameter.detach())
    out = dict(config=config_name, n_qubits=n_qubits, seed=seed, backend=backend.name, nu_init=nu_init,
               nu=nu_learned, nu_rel_err=abs(nu_learned - NU_TRUE) / NU_TRUE,
               rel_l2_T_qpu=relative_l2_error(u_qpu, u_ref), rel_l2_T_measured_model=relative_l2_error(u_sur, u_ref),
               rel_l2_T_ideal_circuit=relative_l2_error(u_ideal, u_ref), chains=len(chains),
               fit_rms_calibration=float(np.mean(fit_rms)), n_jobs=len(job_ids), time_s=time.time() - t0)

    # ---------- outputs ----------
    out_dir = os.path.join(base_dir, config_name)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{N_COLLOCATION}_NQ{n_qubits}_{seed}_E{epochs}"
    tc.save(dict(state_dict=model.state_dict(), config=model.config, cfg=cfg, cones=cones,
                 coefs=[c.tolist() for c in model.quantum.coef.numpy()], chains=chains, result=out),
            os.path.join(out_dir, f"model_{tag}.pt"))
    with open(os.path.join(out_dir, f"run_{tag}.json"), "w") as f:
        json.dump(dict(out, cfg=cfg, job_ids=job_ids, chain_list=chains), f, indent=1)
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


def _run_job(job):
    return run_one(*job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", choices=tuple(CONFIGS) + ("all",), default="all")
    parser.add_argument("--qubits", type=int, nargs="+", default=list(QUBITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--real", action="store_true", help=f"QPU real {BACKEND} (padrão: o fake, local)")
    parser.add_argument("--workers", type=int, default=4, help="runs em paralelo no fake")
    args = parser.parse_args()
    names = tuple(CONFIGS) if args.config == "all" else (args.config,)

    backend = get_backend(args.real)
    base_dir = os.path.join(OUT_DIR, f"T_{TFIN}", backend.name)
    for n in args.qubits:   # once per qubit count, before the runs that share it
        calibrate(backend, n, args.real, base_dir)
    jobs = [(name, n, s) for name in names for n in args.qubits for s in args.seeds]
    if args.real:           # one Session per run, one after the other
        results = [run_one(name, n, s, real=True) for name, n, s in jobs]
    else:
        threads = max(1, (os.cpu_count() or 2) // args.workers)
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_run_job, [(name, n, s, False, threads) for name, n, s in jobs]))

    print(f"{backend.name}: NU={NU_TRUE}, T={TFIN}, N_col={N_COLLOCATION}, {EPOCHS} epochs -- learned nu (relative error)")
    for name in names:
        runs = [r for r in results if r["config"] == name]
        out_dir = os.path.join(base_dir, name)
        if set(args.qubits) == set(QUBITS) and set(args.seeds) == set(SEEDS):
            plot_nu_summary(out_dir)
        err = np.array([r["nu_rel_err"] for r in runs])
        l2 = np.array([r["rel_l2_T_qpu"] for r in runs])
        print(f"{name}: nu error median {100 * np.median(err):.2f}%, mean {100 * err.mean():.2f}%, "
              f"max {100 * err.max():.2f}% | rel. L2 of u(x,T) on the QPU, median {np.median(l2):.2e}")
        for n in args.qubits:
            print(f"  NQ{n:<2d} " + "  ".join(f"seed {r['seed']}: {r['nu']:.4f} ({100 * r['nu_rel_err']:.1f}%)"
                                           for r in runs if r["n_qubits"] == n))
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(dict(config=CONFIGS[name], backend=backend.name, nu_true=NU_TRUE, Tfin=TFIN,
                           n_collocation=N_COLLOCATION, epochs=EPOCHS, nu_rel_err_median=float(np.median(err)),
                           nu_rel_err_mean=float(err.mean()), nu_rel_err_max=float(err.max()),
                           rel_l2_T_qpu_median=float(np.median(l2)), runs=runs), f, indent=1)
