"""Inverse problem at T = 1 (nu = 0.01), 200 epochs, n = 10, one seed: does error suppression / mitigation change the HQPINN
run on FakeAachen? One configuration per call, so the configurations can run side by side.

Same training, CONFIG and T = 1 ADJUSTMENTS as data/Inverso_200 (Inverser_quantum_hardware_pinn_E110.run_one), on the
SAME 10 chains as data/Inverso_200 (its calibration file), so that only the quantum layer changes:

  ideal       exact circuit, no device noise, no shot noise (control: what is left without any noise)
  none        raw FakeAachen circuits = the data/Inverso_200 protocol (its calibration measurements, copied)
  trex        TREX (readout), via the client-side Estimator of qiskit-ibm-runtime >= 0.50 (FakeMitigatedCircuit)
  trex_twirl  TREX + gate and measurement twirling, no dynamical decoupling
  dd_twirl    dynamical decoupling + twirling, no TREX: the preset of the real-QPU run at T = 0.1 (REAL_MITIGATION)

The mitigated configurations measure their own calibration (same chains, same preset as the training). --replica k
writes a second run of the same configuration to <config>_r<k> (the shots and twirling are not seeded, so it measures
how much a single-seed result moves from one run to the next).

    python run/test_mitigation_T1.py --config none --seed 1
    python run/test_mitigation_T1.py --config trex --seed 1 --threads 2

Outputs: data/Inverso_T1_mitigacao/<config>/quantico/T_1.0_nu_0.01/ (as run_one writes them), or quantico_ideal/.
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
import os
import shutil
import sys


def _ensure_libgomp():
    """Linux pip venv: qiskit_aer and torch need their libgomp preloaded (as run/env.sh)."""
    site = glob.glob(os.path.join(sys.prefix, "lib", "python3.*", "site-packages"))
    libs = [p for s in site for pat in ("qiskit_aer.libs/libgomp*.so*", "torch/lib/libgomp.so*")
            for p in sorted(glob.glob(os.path.join(s, pat)))[:1]]
    if libs and not all(lib in os.environ.get("LD_PRELOAD", "") for lib in libs):
        os.environ["LD_PRELOAD"] = ":".join(libs)
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_libgomp()

import numpy as np  # noqa: E402

import Inverser_quantum_hardware_pinn_E110 as H  # noqa: E402
from Inverser_classical_pinn_E110 import adjustments_for  # noqa: E402

TFIN, NU, N_QUBITS, EPOCHS = 1.0, 0.01, 10, 200
BACKEND = "ibm_aachen"                    # -> FakeAachen
SOURCE_CAL = os.path.join("data", "Inverso_200", "quantico", "calibration_NQ10_fake_aachen.npz")
OUT = os.path.join("data", "Inverso_T1_mitigacao")

# TREX + twirling without dynamical decoupling: not one of MITIGATIONS. DD is left out because the Aer noise model has
# no coherent errors, so the emulator cannot measure it (select-error-mitigation skill, verification.md, Method B).
H.MITIGATIONS.setdefault("trex_twirl", dict(resilience_level=0, resilience=dict(measure_mitigation=True),
                                            twirling=dict(enable_gates=True, enable_measure=True)))
CONFIGS = ("ideal", "none", "trex", "trex_twirl", "dd_twirl")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, choices=CONFIGS)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=1, help="simulator threads")
    p.add_argument("--replica", type=int, default=0, help="k > 0: another run of the same configuration -> <config>_r<k>")
    args = p.parse_args()
    adj = adjustments_for(TFIN, NU)
    name = args.config + (f"_r{args.replica}" if args.replica else "")
    out_root = os.path.join(OUT, name)

    if args.config == "ideal":
        from Inverser_runs_paralelo import run_job
        result = run_job(("quantum", N_QUBITS, args.seed, TFIN, EPOCHS, None, out_root, args.threads))
    else:
        cal_dir = os.path.join(out_root, "quantico")
        os.makedirs(cal_dir, exist_ok=True)
        chains = np.load(SOURCE_CAL)["chains"].tolist()
        backend = H.get_backend(False, BACKEND)
        if args.config == "none":   # the very calibration measurements of data/Inverso_200
            target = os.path.join(cal_dir, os.path.basename(SOURCE_CAL))
            if not os.path.exists(target):
                shutil.copy2(SOURCE_CAL, target)
        else:                        # measured here with the preset, on the same chains (no search, no screening)
            H.calibrate(backend, N_QUBITS, False, cal_dir, args.config, chains=chains)
        result = H.run_one(N_QUBITS, args.seed, real=False, threads=args.threads, tfin=TFIN, nu_true=NU, cfg=H.CONFIG,
                           epochs=EPOCHS, adj=adj, backend_name=BACKEND, out_root=out_root, mitigation=args.config)
    print(json.dumps({k: result[k] for k in ("seed", "nu", "nu_rel_err") if k in result}
                     | {k: v for k, v in result.items() if k.startswith("rel_l2")}), flush=True)


if __name__ == "__main__":
    main()
