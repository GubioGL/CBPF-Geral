"""Proxy of the fake QPU for statistics (search_optimizer_ibm_faker.py's idea): the polynomial
calibrated on the fake plays the chip and every <Z> is the mean of 1024 +-1 shots.
Runs Inverser_quantum_hardware_pinn_E110.run_one unchanged, with make_qpu swapped.

usage: python hw_proxy.py <out_dir> <seeds: e.g. 200-219> [workers]
"""
import sys, json, time
sys.path.insert(0, ".")   # run from the repo root
import numpy as np
import torch as tc
from concurrent.futures import ProcessPoolExecutor
import Inverser_quantum_hardware_pinn_E110 as HW
from Inverser_quantum_hardware_pinn import point_grid

REAL_OUT = HW.OUT_DIR
SHOTS = int(np.ceil(HW.PRECISION ** -2))


class ProxyQPU:
    def __init__(self, cones, coefs, B, rng):
        self.truth, self.B, self.rng = HW.HardwareSurrogateLayer(cones, coefs), B, rng

    def evaluate(self, angles, job_ids, replicas=1, shift=0, verbose=True):
        a = np.repeat(angles, replicas, axis=0)
        with tc.no_grad():
            z = self.truth(tc.as_tensor(a, dtype=tc.get_default_dtype())).double().numpy()
        evs = 2.0 * self.rng.binomial(SHOTS, np.clip((1.0 + z) / 2.0, 0.0, 1.0)) / SHOTS - 1.0
        job_ids.append("proxy")
        return evs.reshape(len(angles), replicas, -1).mean(1), a, evs


def proxy_make_qpu(backend, chains, n_qubits, real, threads=0):
    cal = np.load(f"{REAL_OUT}/T_{HW.TFIN}/{backend.name}/calibration_NQ{n_qubits}_{backend.name}.npz")
    M, cones = HW.circuit_structure(n_qubits)
    coefs = HW.fit_surrogate(cal["angles"], cal["evs"], cones, M)[0]
    return ProxyQPU(cones, coefs, len(chains), np.random.default_rng(_NOISE_SEED[0]))


_NOISE_SEED = [0]


def job(args):
    out_dir, name, n, seed = args
    HW.OUT_DIR = out_dir
    HW.make_qpu = proxy_make_qpu
    _NOISE_SEED[0] = 1000 + seed
    return HW.run_one(name, n, seed)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    lo, hi = map(int, sys.argv[2].split("-"))
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    import shutil, os
    for n in HW.QUBITS:   # the fake calibrations, copied where run_one looks for them
        src = f"{REAL_OUT}/T_{HW.TFIN}/fake_berlin/calibration_NQ{n}_fake_berlin.npz"
        dst = f"{out_dir}/T_{HW.TFIN}/fake_berlin"
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)
    jobs = [(out_dir, name, n, s) for name in HW.CONFIGS for n in HW.QUBITS for s in range(lo, hi + 1)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(job, jobs))
    json.dump(res, open(f"{out_dir}/proxy_results.json", "w"), indent=1)
    for name in HW.CONFIGS:
        e = np.array([r["nu_rel_err"] for r in res if r["config"] == name])
        l2 = np.array([r["rel_l2_T_qpu"] for r in res if r["config"] == name])
        per = " ".join(f"NQ{n}:{100*np.median([r['nu_rel_err'] for r in res if r['config']==name and r['n_qubits']==n]):.2f}%" for n in HW.QUBITS)
        print(f"{name:16s} nu err median {100*np.median(e):.2f}% mean {100*e.mean():.2f}% max {100*e.max():.2f}% | {per} | relL2 QPU median {np.median(l2):.2e}")
    print(f"{len(jobs)} runs, {time.time()-t0:.0f}s")
