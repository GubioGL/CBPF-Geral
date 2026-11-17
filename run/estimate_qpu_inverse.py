"""Before a real-QPU run of the inverse problem (Inverser_quantum_hardware_pinn_E110.py --real): every setting of the run,
the plan of jobs (parameter sets, shots, circuit executions) and IBM's estimate of the QPU time. Nothing runs on the QPU.

  - Plan: the jobs of one run (calibration, one per epoch, validation), their parameter sets on the packed circuit (all the
    chains side by side, one circuit), and the shots and circuit executions of each job as the Estimator would build them
    with the mitigation preset (counted with qiskit-ibm-runtime's client-side `prepare`).
  - --dry-run: one dry-run job of each kind (calibration, epoch, validation) sent to IBM with the run's options:
    `EstimatorV2.run(..., dry_run=True)` validates the job without executing it on the QPU and returns mock data, and
    `job.usage_estimation["quantum_seconds"]` is IBM's estimate of its QPU time. Total = calibration + epochs x epoch +
    validation. Needs the IBM account saved on the machine (QiskitRuntimeService.save_account).
  - Wall time: the QPU time does not include the latency of each job (transfer, compilation, result); the reference is the
    real forward run on ibm_berlin (one job per epoch, 2048 shots): data/quantum_forward_hardware_real.

    python run/estimate_qpu_inverse.py --fake                  # offline: FakeAachen, plan and shots only
    python run/estimate_qpu_inverse.py                         # ibm_aachen: real chains, plan and shots
    python run/estimate_qpu_inverse.py --dry-run               # + IBM's QPU-time estimate (3 dry-run jobs)
Output (real backend only): data/Inverso_real/<backend>/qpu_estimate_T<T>_E<epochs>_<preset>.json
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
import glob
import json
import math
import os
import sys
import time


def _ensure_libgomp():
    """Linux pip venv: qiskit_aer and torch need their libgomp preloaded (as run/Inverser_runs_paralelo.py)."""
    site = glob.glob(os.path.join(sys.prefix, "lib", "python3.*", "site-packages"))
    libs = [p for s in site for pat in ("qiskit_aer.libs/libgomp*.so*", "torch/lib/libgomp.so*")
            for p in sorted(glob.glob(os.path.join(s, pat)))[:1]]
    if libs and not all(lib in os.environ.get("LD_PRELOAD", "") for lib in libs):
        os.environ["LD_PRELOAD"] = ":".join(libs)
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_libgomp()
import numpy as np  # noqa: E402

import Inverser_quantum_hardware_pinn_E110 as H  # noqa: E402
from Inverser_classical_pinn_E110 import CONFIG as CLASSICAL_CONFIG, adjustments_for  # noqa: E402
from Inverser_quantum_hardware_pinn import MITIGATIONS, PackedCircuit, find_chains, point_grid  # noqa: E402

FORWARD_REAL = os.path.join("data", "quantum_forward_hardware_real", "T_0.1", "run_10_NQ10_0_ibm_berlin_none_HW110.json")


def plan(B, n_points_epoch, epochs, cal_samples, val_points):
    """Parameter sets of each job on the packed circuit (B chains side by side)."""
    epoch_sets = max(point_grid(n_points_epoch, B, 1, s).shape[0] for s in range(B))
    return dict(calibration=(1, math.ceil(cal_samples / B)), epoch=(epochs, epoch_sets),
                validation=(1, point_grid(val_points, B, B).shape[0]))


def pub(qpu, n_sets, rng):
    vals = np.zeros((n_sets, qpu.n_params))
    vals[:, qpu.phi_idx] = rng.uniform(-np.pi, np.pi, (n_sets, qpu.B, qpu.n))
    return (qpu.isa, qpu.obs, vals[:, None, None, :])


def count(qpu, options, n_sets, backend):
    """Shots and circuit executions of one job, as the Estimator builds them with these options."""
    from qiskit_ibm_runtime.executor_estimator import Estimator as ClientEstimator
    from qiskit_ibm_runtime.executor_estimator.prepare import prepare
    prog, _ = prepare([pub(qpu, n_sets, np.random.default_rng(0))], ClientEstimator(mode=backend, options=options).options,
                      H.PRECISION, backend=backend)
    sizes = [it.size() for it in prog.items]
    return prog.shots * sum(sizes), sum(sizes)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", default="ibm_aachen", choices=list(H.FAKE_BACKENDS))
    p.add_argument("--fake", action="store_true", help="offline, on the fake of the backend (plan and shots only)")
    p.add_argument("--dry-run", action="store_true", help="IBM's QPU-time estimate: 3 dry-run jobs (nothing executes)")
    p.add_argument("--qubits", type=int, default=10)
    p.add_argument("--seed", type=int, default=1, help="best 200-epoch HQPINN run at T = 0.1 on FakeAachen (eps_nu 1.0%%)")
    p.add_argument("--tfin", type=float, default=0.1)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--mitigation", default=H.REAL_MITIGATION, choices=list(MITIGATIONS))
    p.add_argument("--val-points", type=int, default=H.REAL_VAL_POINTS)
    p.add_argument("--cal-samples", type=int, default=H.REAL_CAL_SAMPLES)
    p.add_argument("--packing", default=H.REAL_PACKING, choices=["greedy", "max"])
    args = p.parse_args()
    if args.fake and args.dry_run:
        p.error("--dry-run needs the real backend")

    real = not args.fake
    backend = H.get_backend(real, args.backend)
    options = MITIGATIONS[args.mitigation]
    adj = adjustments_for(args.tfin, args.nu)
    n_ic = (adj or {}).get("n_ic", 0) if not (adj or {}).get("hard_ic", True) else 0
    n = args.qubits
    if args.packing == "max":
        chains, scores = H.pack_chains(backend, n, H.MAX_CHAIN_SCORE_PER_QUBIT * n)
    else:
        chains, scores, _ = find_chains(backend, n)
        chains, scores = zip(*[(c, s) for c, s in zip(chains, scores) if s <= H.MAX_CHAIN_SCORE_PER_QUBIT * n])
    gates = H.cnot_gate_list(n, H.FIXED_LAYER, H.FIXED_LAYER_REPEATS)
    qpu = PackedCircuit(backend, chains, n, gates, H.PRECISION, False, options)
    jobs = plan(qpu.B, H.N_COLLOCATION + n_ic, args.epochs, args.cal_samples, args.val_points)

    print("=" * 100)
    print(f"Inverse problem on the QPU: settings ({'fake ' if args.fake else ''}{backend.name})")
    print("=" * 100)
    rows = [
        ("backend", f"{backend.name} ({backend.num_qubits} qubits, 2-qubit gate {qpu.g2})"),
        ("instance", H.INSTANCE or "default of the saved account"),
        ("problem", f"Burgers, nu_true = {args.nu}, T = {args.tfin}, L = 1, nu learned (varpro)"),
        ("model", f"HQPINN, n = {n} qubits, fixed layer {H.FIXED_LAYER} x {H.FIXED_LAYER_REPEATS}, RY inputs, <Z_i> outputs"),
        ("seed", f"{args.seed} (network init and nu_init as in data/Inverso_200, seed {args.seed})"),
        ("epochs", f"{args.epochs}, one QPU job per epoch, {H.CONFIG['inner_steps']} AdamW steps on each measurement"),
        ("collocation", f"{H.N_COLLOCATION} random points per epoch (residual and data){f', + {n_ic} IC points' if n_ic else ''}"),
        ("T adjustments", adj or "none (T = 0.1)"),
        ("training CONFIG", H.CONFIG),
        ("mitigation", f"{args.mitigation}: {options}"),
        ("shots", f"{round(1 / H.PRECISION ** 2)} per parameter set (precision {H.PRECISION:g})"),
        ("chains", f"{qpu.B} chains of {n} qubits ({args.packing} packing, score <= {H.MAX_CHAIN_SCORE_PER_QUBIT} x n, scores "
                   f"{min(scores):.3f}-{max(scores):.3f}; the calibration job then drops those with misfit > "
                   f"{H.CHAIN_OUTLIER} x median), side by side in one circuit of {qpu.n_qubits_used} qubits"),
        ("ISA circuit", f"depth {qpu.depth}, {qpu.n_2q} {qpu.g2} (expected {qpu.n_2q_expected})"),
        ("calibration", f"{args.cal_samples} samples per output, inside the Session; polynomial refit every {H.RECAL_EVERY} "
                        f"epochs with the training data"),
        ("validation", f"{args.val_points} x points at t = T, each measured on every chain"),
        ("session", f"one Session per run, max_time {H.SESSION_MAX_TIME}"),
        ("output", os.path.join("data", "Inverso_real", args.backend)),
    ]
    for k, v in rows:
        print(f"  {k:<16} {v}")
    print("\n  classical CONFIG (for comparison):", CLASSICAL_CONFIG)

    print("\nPlan of jobs (shots and circuit executions counted as the Estimator builds them)")
    total = dict(jobs=0, sets=0, shots=0, executions=0)
    per = {}
    for kind, (n_jobs, sets) in jobs.items():
        shots, execs = count(qpu, options, sets, backend)
        per[kind] = dict(jobs=n_jobs, parameter_sets=sets, shots=shots, executions=execs)
        for k, v in (("jobs", n_jobs), ("sets", n_jobs * sets), ("shots", n_jobs * shots), ("executions", n_jobs * execs)):
            total[k] += v
        print(f"  {kind:<12} {n_jobs:4d} job(s) x {sets:4d} parameter sets: {shots:9,d} shots, {execs:5d} executions per job")
    raw = sum(per[k]["jobs"] * per[k]["parameter_sets"] for k in per) * round(1 / H.PRECISION ** 2)
    print(f"  total        {total['jobs']} jobs, {total['sets']} parameter sets, {total['shots']:,d} shots "
          f"({100 * (total['shots'] / raw - 1):+.0f}% vs no mitigation), {total['executions']:,d} circuit executions")

    out = dict(backend=backend.name, fake=args.fake, mitigation=args.mitigation, options=options, seed=args.seed,
               packing=args.packing, cal_samples=args.cal_samples, val_points=args.val_points,
               tfin=args.tfin, nu=args.nu, epochs=args.epochs, qubits=n, chains=chains, config=H.CONFIG, adjustments=adj,
               isa=dict(depth=qpu.depth, n_2q=qpu.n_2q, qubits_used=qpu.n_qubits_used), jobs=per, total=total)

    if os.path.exists(FORWARD_REAL):
        f = json.load(open(FORWARD_REAL))
        s_epoch = f["training_time_s"] / f["jobs_planned"]["training"]["jobs"]
        s_set = f["calibration_time_s"] / f["jobs_planned"]["calibration"]["parameter_sets"]   # big jobs: per parameter set
        cal, val = s_set * per["calibration"]["parameter_sets"], s_set * per["validation"]["parameter_sets"]
        wall = cal + args.epochs * s_epoch + val
        out["wall_reference"] = dict(forward_s_per_epoch=s_epoch, forward_s_per_calibration_set=s_set, estimate_s=wall)
        print(f"\nWall time, scaled from the real forward run on {f['backend']} ({s_epoch:.1f} s per epoch job; calibration "
              f"{f['calibration_time_s']:.0f} s for {f['jobs_planned']['calibration']['parameter_sets']} parameter sets, "
              f"{s_set:.1f} s each):\n  calibration ~{cal / 60:.0f} min + {args.epochs} epochs ~{args.epochs * s_epoch / 60:.0f} min "
              f"+ validation ~{val / 60:.0f} min = about {wall / 3600:.1f} h (Session max_time {H.SESSION_MAX_TIME})")

    if args.dry_run:
        print("\nIBM dry run (nothing executes on the QPU): QPU-time estimate per job ...", flush=True)
        rng, est = np.random.default_rng(1), {}
        for kind, (n_jobs, sets) in jobs.items():
            job = qpu.estimator.run([pub(qpu, sets, rng)], precision=H.PRECISION, dry_run=True)
            job.result()
            for _ in range(150):      # the estimate can take a few minutes to appear
                q = job.usage_estimation.get("quantum_seconds")
                if q is not None:
                    break
                time.sleep(2)
                job._usage_estimation = {}
            est[kind] = dict(job_id=job.job_id(), quantum_seconds=q)
            print(f"  {kind:<12} job {job.job_id()}: {q} s of QPU per job", flush=True)
        if all(v["quantum_seconds"] is not None for v in est.values()):
            qs = sum(jobs[k][0] * est[k]["quantum_seconds"] for k in jobs)
            print(f"  total QPU time of the run: {qs:.0f} s = {qs / 60:.1f} min")
            est["total_quantum_seconds"] = qs
        out["dry_run"] = est

    if not args.fake:      # data/Inverso_real holds the real QPU only: the offline plan is printed, not saved
        path = os.path.join("data", "Inverso_real", args.backend)
        os.makedirs(path, exist_ok=True)
        name = f"qpu_estimate_T{args.tfin:g}_E{args.epochs}_{args.mitigation}.json"
        with open(os.path.join(path, name), "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(f"\n-> {os.path.join(path, name)}")
    print(f"\nRun (after checking the above):\n  python run/Inverser_quantum_hardware_pinn_E110.py --real --backend {args.backend} "
          f"--qubits {n} --seeds {args.seed} --tfin {args.tfin:g} --nu {args.nu:g} --epochs {args.epochs} "
          f"--mitigation {args.mitigation} --val-points {args.val_points} --cal-samples {args.cal_samples}")


if __name__ == "__main__":
    main()
