"""Inverse problem, many runs at once: classical PINN and HQPINN, each with its best optimizer configuration, in parallel
with a progress bar and a time estimate.

Training: exactly the one of the 110/500-epoch study (Inverser_classical_pinn_E110.train and
Inverser_quantum_hardware_pinn_E110.train), with each model's CONFIG (the best optimizer configuration found for it) and, at
T = 1, the same ADJUSTMENTS. nu_true = 0.01, N_col = 10 per epoch, T = 0.1 and 1. Only the epochs (the cosine decays
stretch over them), the sizes, the seeds and the quantum layer change, through a preset:

  ideal      20 000 epochs, n = 3/7/10, seeds 0-10, IDEAL (noiseless) circuit        -> data/Inverso/      (132 runs)
  aachen200  200 epochs, n = 10, seeds 0-9, circuit on FakeAachen (as data/Inverso_500) -> data/Inverso_200/  (40 runs)

    python run/Inverser_runs_paralelo.py --benchmark                       # a few epochs of each kind of run -> time of everything
    python run/Inverser_runs_paralelo.py                                   # preset ideal (resumable: runs already saved are skipped)
    python run/Inverser_runs_paralelo.py --preset aachen200 --benchmark
    python run/Inverser_runs_paralelo.py --preset aachen200
    python run/Inverser_runs_paralelo.py --models quantum --tfin 0.1 --workers 12

Outputs: <out>/classico/ and <out>/quantico_ideal/ (ideal) or <out>/quantico/ (emulator), each with T_<T>_nu_0.01/
(run_*.json, losshist_*.npz, validation_*.npz, model_*.pt and summary.json), and <out>/benchmark.json. On an emulator the
quantum runs are written by Inverser_quantum_hardware_pinn_E110.run_one, in the format of data/Inverso_500; the calibration
(chains and calibration measurements) was copied from data/Inverso_500 (since deleted) to data/Inverso_200/quantico, so the
chains are the same as there.
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
import heapq
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED


def _ensure_libgomp():
    """The quantum module imports qiskit_aer: in a pip venv on Linux it needs its libgomp and torch's preloaded (as
    run/env.sh). Re-executes this script once with them, so the user does not have to."""
    site = glob.glob(os.path.join(sys.prefix, "lib", "python3.*", "site-packages"))
    libs = [p for s in site for pat in ("qiskit_aer.libs/libgomp*.so*", "torch/lib/libgomp.so*")
            for p in sorted(glob.glob(os.path.join(s, pat)))[:1]]
    if libs and not all(lib in os.environ.get("LD_PRELOAD", "") for lib in libs):
        os.environ["LD_PRELOAD"] = ":".join(libs)
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_libgomp()

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

PRESETS = {
    "ideal": dict(epochs=20000, sizes=(3, 7, 10), seeds=tuple(range(11)), backend=None,
                  out=os.path.join("data", "Inverso"), bench_epochs=100),
    "aachen200": dict(epochs=200, sizes=(10,), seeds=tuple(range(10)), backend="ibm_aachen",
                      out=os.path.join("data", "Inverso_200"), bench_epochs=10,
                      calibration_from=os.path.join("data", "Inverso_500", "quantico")),
}
NU = 0.01
TFINS = (0.1, 1.0)
N_COL = 10
VAL_POINTS = 100
PREFIX = {"classical": "NE", "quantum": "NQ"}
TICK = 10                     # epochs between two progress messages of a worker


def folder(model, backend):
    return "classico" if model == "classical" else ("quantico" if backend else "quantico_ideal")


def case_dir(out, model, backend, tfin):
    return os.path.join(out, folder(model, backend), f"T_{tfin}_nu_{NU:g}")


def tag(model, size, seed, epochs):
    return f"{N_COL}_{PREFIX[model]}{size}_{seed}_E{epochs}"


def prepare_calibration(out, backend, sizes, source):
    """Emulator runs: the chains and calibration measurements of data/Inverso_500 (one per qubit count), copied, or measured
    here once, before the runs that share them (the workers then only read them)."""
    import Inverser_quantum_hardware_pinn_E110 as H
    target = os.path.join(out, "quantico")
    os.makedirs(target, exist_ok=True)
    be = H.get_backend(False, backend)
    for n in sizes:
        name = f"calibration_NQ{n}_{be.name}.npz"
        if source and not os.path.exists(os.path.join(target, name)) and os.path.exists(os.path.join(source, name)):
            shutil.copy2(os.path.join(source, name), target)
        H.calibrate(be, n, False, target)


def run_job(job, queue=None, save=True):
    """One run, in a worker process. Returns its result, with the measured seconds per epoch of the training."""
    import torch as tc
    tc.set_num_threads(1)
    model_kind, size, seed, tfin, epochs, backend, out, threads = job
    t0 = time.time()
    count, stamps = [0], []

    def progress():
        count[0] += 1
        if count[0] in (1, epochs):
            stamps.append(time.time())
        if queue is not None and count[0] % TICK == 0:
            queue.put(TICK)

    if model_kind == "quantum" and backend:
        import Inverser_quantum_hardware_pinn_E110 as M
        from Inverser_classical_pinn_E110 import adjustments_for
        adj = adjustments_for(tfin, NU)
        result = M.run_one(size, seed, real=False, threads=threads, tfin=tfin, nu_true=NU, cfg=M.CONFIG, epochs=epochs,
                           adj=adj, backend_name=backend, out_root=out, on_epoch=progress)
        result = dict(result, model=model_kind, epochs=epochs)
    else:
        if model_kind == "classical":
            import Inverser_classical_pinn_E110 as M
            from Inverser_classical_pinn_paralelo import build_model, exact_solution, predict, relative_l2_error, set_seed
            M.NU_TRUE = NU                     # the data of train() come from this module global
            adj = M.adjustments_for(tfin, NU)
            size_key, kw = "neurons", {}
        else:
            import Inverser_quantum_hardware_pinn_E110 as M
            from Inverser_classical_pinn_E110 import adjustments_for
            from Inverser_quantum_exact_pinn_paralelo import build_model, exact_solution, predict, relative_l2_error, set_seed
            adj = adjustments_for(tfin, NU)
            size_key, kw = "n_qubits", dict(measure=None, nu_true=NU)   # exact (noiseless) quantum layer
        set_seed(seed)
        nu_init = float(np.random.random())
        model = build_model(size, tfin, nu_init)
        model.hard_ic = (adj or {}).get("hard_ic", True)
        hist = M.train(model, seed, cfg=M.CONFIG, Tfin=tfin, n_col=N_COL, epochs=epochs, adj=adj, on_epoch=progress, **kw)
        x_val = np.linspace(0.0, 1.0, VAL_POINTS, endpoint=False).reshape(-1, 1)
        u_pred, u_ref = predict(model, x_val, tfin), exact_solution(x_val, tfin, nu=NU, L=1.0)
        nu_learned = float(model.learnable_parameter.detach())
        result = {size_key: size, "seed": seed, "model": model_kind, "Tfin": tfin, "nu_true": NU, "nu_init": nu_init,
                  "nu": nu_learned, "nu_rel_err": abs(nu_learned - NU) / NU, "rel_l2_T": relative_l2_error(u_pred, u_ref),
                  "epochs": epochs, "adjustments": adj, "cfg": M.CONFIG}
    if queue is not None and count[0] % TICK:
        queue.put(count[0] % TICK)
    # the training alone: imports, calibration fit and validation do not count per epoch
    result.update(time_s=time.time() - t0, s_per_epoch=(stamps[-1] - stamps[0]) / max(1, epochs - 1))
    if save and not (model_kind == "quantum" and backend):     # run_one has saved the emulator runs itself
        d, t = case_dir(out, model_kind, backend, tfin), tag(model_kind, size, seed, epochs)
        os.makedirs(d, exist_ok=True)
        tc.save(dict(state_dict=model.state_dict(), cfg=M.CONFIG, result=result), os.path.join(d, f"model_{t}.pt"))
        np.savez(os.path.join(d, f"losshist_{t}.npz"), **{k: np.array(v) for k, v in hist.items()})
        np.savez(os.path.join(d, f"validation_{t}.npz"), x=x_val, u_pred=u_pred, u_exact=u_ref)
    if save:
        path = os.path.join(case_dir(out, model_kind, backend, tfin), f"run_{tag(model_kind, size, seed, epochs)}.json")
        record = dict(json.load(open(path)), **result) if os.path.exists(path) else result   # run_one's keeps cfg, job ids, chains
        with open(path, "w") as f:
            json.dump(record, f, indent=1)    # written last: with the validation file, it marks the run as done
    return result


def is_done(out, m, backend, T, n, s, epochs):
    d, t = case_dir(out, m, backend, T), tag(m, n, s, epochs)
    return all(os.path.exists(os.path.join(d, f"{kind}_{t}.{ext}")) for kind, ext in (("run", "json"), ("validation", "npz")))


def summarize(out, model_kind, backend, tfin, epochs):
    d = case_dir(out, model_kind, backend, tfin)
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, f"run_*_E{epochs}.json")))]
    if not runs:
        return
    for r in runs:
        for k in ("job_ids", "chain_list"):
            r.pop(k, None)
    err = np.array([r["nu_rel_err"] for r in runs])
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(dict(nu_true=NU, Tfin=tfin, epochs=epochs, n_collocation=N_COL, model=model_kind, backend=backend or "ideal",
                       nu_rel_err_median=float(np.median(err)), nu_rel_err_mean=float(err.mean()),
                       nu_rel_err_max=float(err.max()), runs=runs), f, indent=1)


def makespan(durations, workers):
    """Wall time of running the durations on that many processes, longest first (each run goes to the first free one)."""
    free = [0.0] * workers
    for d in sorted(durations, reverse=True):
        heapq.heappush(free, heapq.heappop(free) + d)
    return max(free)


def hours(s):
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"


def pool(workers):
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))


def key(m, n, T):
    return f"{m}_n{n}_T{T}"


def benchmark(args, threads):
    """Every kind of run (model x size x T) for a few epochs, repeated with other seeds until all the processes are busy
    (the machine loaded as in the real run), then the time of the whole plan from the measured seconds per epoch.
    Nothing is written in the output folder but benchmark.json."""
    kinds = [(m, n, T) for m in args.models for n in args.sizes for T in args.tfin]
    jobs = [(m, n, i // len(kinds), T) for i, (m, n, T) in
            ((i, kinds[i % len(kinds)]) for i in range(max(len(kinds), args.workers)))]
    scratch = tempfile.mkdtemp(prefix="bench_", dir=args.out)
    try:
        if args.backend and "quantum" in args.models:
            shutil.copytree(os.path.join(args.out, "quantico"), os.path.join(scratch, "quantico"), dirs_exist_ok=True)
        print(f"benchmark: {len(kinds)} kinds of run, {len(jobs)} runs x {args.bench_epochs} epochs on {args.workers} processes")
        full = [(m, n, s, T, args.bench_epochs, args.backend, scratch, threads) for m, n, s, T in jobs]
        with pool(args.workers) as ex:
            results = list(tqdm(ex.map(run_job, full, [None] * len(full), [False] * len(full)), total=len(full),
                                desc="benchmark"))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    rate = {}
    for r in results:
        rate.setdefault(key(r["model"], r.get("neurons", r.get("n_qubits")), r["Tfin"]), []).append(r["s_per_epoch"])
    rate = {k: float(np.mean(v)) for k, v in rate.items()}
    with open(os.path.join(args.out, "benchmark.json"), "w") as f:
        json.dump(dict(workers=args.workers, bench_epochs=args.bench_epochs, backend=args.backend or "ideal",
                       s_per_epoch=rate), f, indent=1)
    print(f"\n{'run':<26}{'s/epoch':>9}{'one run':>12}")
    for k, v in sorted(rate.items()):
        print(f"{k:<26}{v:9.3f}{hours(v * args.epochs):>12}")
    plan = [rate[key(m, n, T)] * args.epochs for m in args.models for n in args.sizes for T in args.tfin for _ in args.seeds]
    for m in args.models:
        part = [rate[key(m, n, T)] * args.epochs for n in args.sizes for T in args.tfin for _ in args.seeds]
        print(f"{m}: {len(part)} runs, {hours(makespan(part, args.workers))} alone on {args.workers} processes")
    print(f"everything: {len(plan)} runs of {args.epochs} epochs on {args.workers} processes: about "
          f"{hours(makespan(plan, args.workers))}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset", default="ideal", choices=list(PRESETS), help="epochs, sizes, seeds, quantum layer, output")
    p.add_argument("--models", nargs="+", default=["quantum", "classical"], choices=["quantum", "classical"])
    p.add_argument("--sizes", type=int, nargs="+", help="neurons or qubits (default: the preset's)")
    p.add_argument("--seeds", type=int, nargs="+", help="default: the preset's")
    p.add_argument("--tfin", type=float, nargs="+", default=list(TFINS))
    p.add_argument("--epochs", type=int, help="default: the preset's")
    p.add_argument("--workers", type=int, default=12, help="processes in parallel (16 threads here; 12 left the machine usable)")
    p.add_argument("--benchmark", action="store_true", help="only measure, and estimate the time of the whole plan")
    p.add_argument("--bench-epochs", type=int, help="default: the preset's")
    args = p.parse_args()
    preset = PRESETS[args.preset]
    for k in ("sizes", "seeds", "epochs", "bench_epochs"):
        if getattr(args, k) is None:
            setattr(args, k, preset[k])
    args.backend, args.out = preset["backend"], preset["out"]
    os.makedirs(args.out, exist_ok=True)
    threads = max(1, (os.cpu_count() or 2) // args.workers)     # simulator threads of one emulator run
    if args.backend and "quantum" in args.models:
        prepare_calibration(args.out, args.backend, args.sizes, preset.get("calibration_from"))
    if args.benchmark:
        return benchmark(args, threads)

    jobs = [(m, n, s, T, args.epochs, args.backend, args.out, threads)
            for m in args.models for n in args.sizes for T in args.tfin for s in args.seeds
            if not is_done(args.out, m, args.backend, T, n, s, args.epochs)]
    bench = os.path.join(args.out, "benchmark.json")
    rate = json.load(open(bench))["s_per_epoch"] if os.path.exists(bench) else {}
    cost = lambda j: rate.get(key(j[0], j[1], j[3]), 1.0 if j[0] == "quantum" else 0.5) * j[4]   # unknown: a guess
    jobs.sort(key=cost, reverse=True)       # longest first: the slow runs do not end up alone at the end
    print(f"{len(jobs)} runs to do ({args.epochs} epochs each, quantum layer: {args.backend or 'ideal'}) on {args.workers} "
          f"processes -> {args.out}/")
    if jobs and rate and all(key(j[0], j[1], j[3]) in rate for j in jobs):
        print(f"estimate from benchmark.json: about {hours(makespan([cost(j) for j in jobs], args.workers))}")
    if not jobs:
        return
    manager = multiprocessing.get_context("spawn").Manager()
    queue = manager.Queue()
    total = sum(j[4] for j in jobs)
    with pool(args.workers) as ex, tqdm(total=total, unit="epoch", unit_scale=True, smoothing=0.05, desc="epochs") as bar:
        futures = {ex.submit(run_job, j, queue): j for j in jobs}
        done = 0
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            while not queue.empty():
                bar.update(queue.get())
            for f in finished:
                r = f.result()          # raises here if a run failed
                done += 1
                bar.set_postfix_str(f"runs {done}/{len(jobs)}; last {r['model']} n={r.get('neurons', r.get('n_qubits'))} "
                                    f"T={r['Tfin']} seed {r['seed']}: nu err {100 * r['nu_rel_err']:.2f}%")
    for m in args.models:
        for T in args.tfin:
            summarize(args.out, m, args.backend, T, args.epochs)
    print("done; summaries in", args.out)


if __name__ == "__main__":
    main()
