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
    ramping up from 0 over the first epochs; inner_steps AdamW steps per epoch, all on the
    epoch's points and its single QPU measurement (the straight-through correction is kept);
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

T = 1, nu = 0.01: the training has the adjustments found for the classical model
(ADJUSTMENTS of Inverser_classical_pinn_E110.py, data/Inverso_110/README.md) -- u(x,0) as a
soft loss on N_IC points at t = 0 per epoch instead of the hard constraint (their <Z> measured in the
epoch's job, with the stencil centres), 10 epochs with the data loss only and nu frozen, and the lr of
the network cut on plateaus instead of the cosine. --treino-antigo runs that case unadjusted.

Fake QPU (default): the noise model of BACKEND simulated locally with Aer (FakeChainCircuit),
no IBM job. Real QPU (--real): PackedCircuit in a Session -- not run in the verification.

Error suppression / mitigation (--mitigation none|trex|suppress|zne, the presets MITIGATIONS of
Inverser_quantum_hardware_pinn.py): on the real QPU they are options of the runtime Estimator
(default there: REAL_MITIGATION = dd_twirl). On the fake they are applied locally by the client-side Estimator of
qiskit-ibm-runtime >= 0.50 (FakeMitigatedCircuit), with the same options, so a preset can be tried
on the emulator before spending QPU time; default on the fake: none (raw circuits), as in the
recorded runs. The calibration, the training and the validation of a run always use the same
preset, and a mitigated fake run goes to its own folder (<out-dir>_<preset>).

Usage (conda env `qiskit`):
    python Inverser_quantum_hardware_pinn_E110.py                        # fake: 3/7/10 qubits x seeds 0..10
    python Inverser_quantum_hardware_pinn_E110.py --qubits 7 10
    python Inverser_quantum_hardware_pinn_E110.py --tfin 0.1                         # final time T = 0.1
    python Inverser_quantum_hardware_pinn_E110.py --tfin 0.1 --nu 0.01               # -> quantico/T_0.1_nu_0.01
    python Inverser_quantum_hardware_pinn_E110.py --nu 0.01                          # T = 1, adjusted -> quantico/T_1.0_nu_0.01
    python Inverser_quantum_hardware_pinn_E110.py --nu 0.01 --treino-antigo          # unadjusted -> .../treino_antigo
    python Inverser_quantum_hardware_pinn_E110.py --real --qubits 3 --seeds 0        # -> data/Inverso_real/<backend>/ (not over the fake runs)
    python Inverser_quantum_hardware_pinn_E110.py --nu 0.01 --epochs 500 --backend ibm_aachen --out-dir data/Inverso_500
    python Inverser_quantum_hardware_pinn_E110.py --nu 0.01 --tfin 0.1 --epochs 200 --backend ibm_aachen --qubits 10 \
        --seeds 0 --mitigation trex --out-dir data/Inverso_200                      # -> data/Inverso_200_trex
Outputs: calibrations (independent of T, shared) in OUT_DIR/quantico/; runs and summary.json in
OUT_DIR/quantico/T_<TFIN>/. The summary figures (nu_summary, comparison with the classical model) are
made by plot_inverse_E110.py.
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
import multiprocessing
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext


def _ensure_libgomp():
    """Linux pip venv: qiskit_aer and torch need their libgomp preloaded ("cannot allocate memory in static TLS
    block" otherwise). Re-executes the script once with them in LD_PRELOAD (as run/Inverser_runs_paralelo.py);
    nothing to do where the libraries are not found (Windows, conda)."""
    site = glob.glob(os.path.join(sys.prefix, "lib", "python3.*", "site-packages"))
    libs = [p for s in site for pat in ("qiskit_aer.libs/libgomp*.so*", "torch/lib/libgomp.so*")
            for p in sorted(glob.glob(os.path.join(s, pat)))[:1]]
    if libs and not all(lib in os.environ.get("LD_PRELOAD", "") for lib in libs):
        os.environ["LD_PRELOAD"] = ":".join(libs)
        os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":   # as a module, the importing script takes care of it
    _ensure_libgomp()

import matplotlib
matplotlib.use("Agg")  # headless backend -- safe across worker processes
import numpy as np
import torch as tc

sys.path.insert(0, str(_ROOT / "src" / "quantum_hardware"))
from qiskit_ibm_runtime import QiskitRuntimeService, Session  # noqa: E402

from Inverser_quantum_exact_pinn_paralelo import (  # noqa: E402  -- the model and the FD residual
    FIXED_LAYER, FIXED_LAYER_REPEATS, HT, HX, INPUT_ROTATION_AXIS, L, ExactQuantumLayer, build_model,
    exact_solution, initial_condition, plot_loss, plot_parameter_history, plot_solution, relative_l2_error,
    residual_terms, set_seed, stencil_inputs,
)
from Inverser_classical_pinn_E110 import ADJUSTED_CASES, adjustments_for  # noqa: E402  -- T = 1 adjustments
from Inverser_quantum_hardware_pinn import (  # noqa: E402  -- the hardware machinery
    FAKE_BACKENDS, MITIGATION, MITIGATIONS, HardwareSurrogateLayer, PackedCircuit, causal_cones,
    cnot_gate_list, find_chains, fit_surrogate, network_angles, network_output, screen_chains,
)
from main.functions import repeated_cnot_network_matrix  # noqa: E402
from regression_hardware_nqubits import INSTANCE  # noqa: E402
from run_quantum_hardware import FakeChainCircuit  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import ParameterVector  # noqa: E402
from qiskit.providers import BackendV2, Options  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from qiskit.transpiler import Target, generate_preset_pass_manager  # noqa: E402

# ---- problem --------------------------------------------------------------------
NU_TRUE = 0.1
TFIN = 1.0
N_COLLOCATION = 10        # random points per epoch: residual AND data
EPOCHS = 110
QUBITS = (3, 7, 10)
SEEDS = tuple(range(11))   # 0..10

# ---- training: "fixed_varpro" (data README, data/Inverso_110/README.md) -----------
CONFIG = dict(
    lr_net=3.07e-2, lr_net_min=1.2e-3, betas_net=(0.77, 0.99),   # AdamW, cosine decay
    weight_decay=6.4e-6, grad_clip=10.0,
    r_ramp=20,                # epochs over which the residual weight ramps up from 0
    w_d=85.8,                 # data weight (the residual weight is 1)
    lr_nu=0.97, lr_nu_min=0.128,   # varpro relaxation factor beta, cosine decay
    inner_steps=16,           # AdamW steps per epoch, all on the epoch's single QPU measurement
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
# --real: error suppression of the inverse runs (author's choice, 17/11/2026): dynamical decoupling + Pauli twirling,
# no TREX -- the same shots as the raw circuits (TREX adds 1024 shots per job and did not help on FakeAachen)
REAL_MITIGATION = "dd_twirl"
# --real, author's choices of 17/11/2026 (the emulator runs keep CAL_SAMPLES, VAL_POINTS and the greedy packing):
REAL_PACKING = "max"      # as many disjoint chains as fit on the chip (12 of 10 qubits on ibm_aachen), not the greedy 10
REAL_CAL_SAMPLES = 512    # calibration samples per output: on FakeAachen the polynomial fitted to 512 is within 3% of the
                          # shot-noise floor on held-out data (1024: 1%), and it is refit with the training data anyway
REAL_VAL_POINTS = 30      # validation: x points at t = T, each measured on every chain

OUT_DIR = os.path.join("data", "Inverso_110")


def case_tag(tfin, nu):
    """T_<tfin>, plus _nu_<nu> when nu is not the reference 0.1."""
    return f"T_{tfin}" if nu == 0.1 else f"T_{tfin}_nu_{nu:g}"


def results_dir(tfin, nu, old=False, out_root=OUT_DIR):
    """<out_root>/quantico/<case_tag>: the runs for one (T, nu); the unadjusted training of an adjusted
    case (old=True) goes to its subfolder treino_antigo/."""
    path = os.path.join(out_root, "quantico", case_tag(tfin, nu))
    return os.path.join(path, "treino_antigo") if old and (tfin, nu) in ADJUSTED_CASES else path


# ----------------------------------------------------------------------
# Training: AdamW for the network, varpro for nu
# ----------------------------------------------------------------------
def cosine(lr0, lr_min, progress):
    """Cosine decay lr0 -> lr_min, progress in [0, 1]."""
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress))


def exact_at(x, t, nu_true):
    """Exact solution u(x, t) (Cole-Hopf, nu = nu_true): the data, shaped like x."""
    return tc.tensor(exact_solution(x.numpy(), t.numpy(), nu=nu_true, L=L), dtype=x.dtype)


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


def train(model, seed, measure=None, cfg=CONFIG, Tfin=TFIN, nu_true=NU_TRUE, n_col=N_COLLOCATION, epochs=EPOCHS,
          adj=None, on_epoch=None, stop_at=None, resume=None, state_out=None):
    """Trains `model` for `epochs` epochs, one batch of n_col fresh random points per epoch.
    measure(epoch, xt, xt_ic), optional (QPU in the loop): gets the epoch's stencil points (and the
    IC points, or None) before the forward pass and returns the offsets added to the output of
    `model.quantum` for each (None for the IC when there are none).
    adj: ADJUSTMENTS (model.hard_ic must already be False when adj["hard_ic"] is), or None.
    stop_at: stop after this many epochs of the `epochs` schedule (the cosine decays keep the full
    length); resume: the state_out of a stopped call, to go on from there exactly as without the stop
    (model weights and nu restored by the caller); state_out: dict filled with the training state
    (optimizer, log nu, random generators, plateau, history, next epoch).
    Returns the history of the loss terms and of nu (from epoch 0, the resumed part included)."""
    adj = adj or {}
    n_ic, warmup, plateau = adj.get("n_ic", 0), adj.get("warmup", 0), adj.get("plateau")
    nu = model.learnable_parameter
    net_params = [p for k, p in model.named_parameters() if k != "learnable_parameter"]
    opt = tc.optim.AdamW(net_params, lr=cfg["lr_net"], betas=cfg["betas_net"], weight_decay=cfg["weight_decay"])
    log_nu = tc.log(nu.detach().clone())
    generator = tc.Generator().manual_seed(seed)
    ic_generator = tc.Generator().manual_seed(seed + 10_000)   # IC points: the collocation points stay the same
    lr_scale, best_loss, wait = 1.0, math.inf, 0                # plateau
    hist = {k: [] for k in ("loss_r", "loss_d", "nu")}
    start = 0
    if resume is not None:
        opt.load_state_dict(resume["optimizer"])
        log_nu = tc.tensor(resume["log_nu"], dtype=log_nu.dtype)
        generator.set_state(resume["generator"])
        ic_generator.set_state(resume["ic_generator"])
        lr_scale, best_loss, wait = resume["lr_scale"], resume["best_loss"], resume["wait"]
        hist = {k: list(v) for k, v in resume["hist"].items()}
        start = resume["epoch"]

    for epoch in range(start, min(epochs, stop_at or epochs)):
        progress = epoch / max(epochs - 1, 1)
        for g in opt.param_groups:
            g["lr"] = cfg["lr_net"] * lr_scale if plateau else cosine(cfg["lr_net"], cfg["lr_net_min"], progress)
        beta = cosine(cfg["lr_nu"], cfg["lr_nu_min"], progress)

        x_f = tc.rand((n_col, 1), generator=generator) * L
        t_f = HT + tc.rand((n_col, 1), generator=generator) * (Tfin - 2.0 * HT)
        xt = stencil_inputs(x_f, t_f, L, HX, HT)   # 5 blocks of n_col rows, the centres first
        xt_ic = None
        if n_ic:
            x_ic = tc.rand((n_ic, 1), generator=ic_generator) * L
            xt_ic, u_ic = tc.cat([x_ic, tc.zeros_like(x_ic)], dim=1), initial_condition(x_ic, L=L)
        offset = offset_ic = None
        if measure is not None:
            offset, offset_ic = measure(epoch, xt, xt_ic)
        u_data = exact_at(x_f, t_f, nu_true)

        ramp = 0.0 if epoch < warmup else min(1.0, (epoch + 1 - warmup) / cfg["r_ramp"])
        for _ in range(cfg.get("inner_steps", 1)):   # the extra steps reuse the epoch's QPU offsets
            if measure is not None:
                model.quantum.offset = offset
            a, b, loss_r, loss_d = loss_terms(model, xt, n_col, u_data)
            loss = ramp * loss_r + cfg["w_d"] * loss_d
            if n_ic:
                if measure is not None:
                    model.quantum.offset = offset_ic
                loss = loss + adj["w_ic"] * (model(xt_ic) - u_ic).abs().mean()
            opt.zero_grad()
            loss.backward(inputs=net_params)
            tc.nn.utils.clip_grad_norm_(net_params, cfg["grad_clip"])
            opt.step()
        if measure is not None:
            model.quantum.offset = None

        if plateau and epoch >= warmup:   # lr / factor when the epoch loss stalls for `patience` epochs
            epoch_loss = ramp * float(loss_r.detach()) + cfg["w_d"] * float(loss_d.detach())
            if epoch_loss < best_loss * (1.0 - plateau["threshold"]):
                best_loss, wait = epoch_loss, 0
            else:
                wait += 1
                if wait > plateau["patience"]:
                    lr_scale, wait = max(lr_scale / plateau["factor"], 0.01), 0

        if epoch >= warmup:               # nu stays at its initial value during the warmup
            nu_star = min(max(nu_minimizer(a, b), 1.0e-4), 10.0)
            log_nu += beta * (math.log(nu_star) - log_nu)
            with tc.no_grad():
                nu.copy_(tc.exp(log_nu))

        hist["loss_r"].append(float(loss_r.detach()))
        hist["loss_d"].append(float(loss_d.detach()))
        hist["nu"].append(float(nu.detach()))
        if on_epoch is not None:   # optional callback, once per epoch (run/Inverser_runs_paralelo.py: progress bar)
            on_epoch()
    if state_out is not None:
        state_out.update(epoch=len(hist["nu"]), optimizer=opt.state_dict(), log_nu=float(log_nu),
                         generator=generator.get_state(), ic_generator=ic_generator.get_state(), lr_scale=lr_scale,
                         best_loss=best_loss, wait=wait, hist={k: list(v) for k, v in hist.items()})
    return hist


# ----------------------------------------------------------------------
# QPU: chains, calibration
# ----------------------------------------------------------------------
def get_backend(real, name=BACKEND):
    if real:
        return QiskitRuntimeService(instance=INSTANCE).backend(name)
    return FAKE_BACKENDS[name]()


class ChainBackend(BackendV2):
    """The qubits `chain` of a fake backend, renumbered 0..n-1, with their gate, readout and T1/T2 data (the
    target restricted to them). AerSimulator.from_backend of it is the noise model of the full chip on those
    qubits: what the chain sees, since the Aer noise model has no crosstalk."""

    def __init__(self, backend, chain):
        super().__init__(name=f"{backend.name}_chain")
        src, idx = backend.target, {q: k for k, q in enumerate(chain)}
        self._target = Target(num_qubits=len(chain), dt=src.dt, granularity=src.granularity,
                              min_length=src.min_length, pulse_alignment=src.pulse_alignment,
                              acquire_alignment=src.acquire_alignment,
                              qubit_properties=[src.qubit_properties[q] for q in chain])
        for name in src.operation_names:
            props, op = src[name], src.operation_from_name(name)
            if props is None or None in props:          # global instruction (control flow, ...)
                if isinstance(op, type):
                    self._target.add_instruction(op, name=name)
                else:
                    self._target.add_instruction(op, {None: None}, name=name)
                continue
            sub = {tuple(idx[q] for q in qa): p for qa, p in props.items() if all(q in idx for q in qa)}
            if sub:
                self._target.add_instruction(op, sub, name=name)

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, *args, **kwargs):
        raise NotImplementedError("simulated through AerSimulator.from_backend")


class FakeMitigatedCircuit:
    """Stand-in for PackedCircuit with IBM's error suppression / mitigation applied locally: the client-side
    Estimator of qiskit-ibm-runtime >= 0.50 implements the runtime options (TREX, dynamical decoupling, Pauli
    twirling, ZNE) on a local simulator; `options` is one of MITIGATIONS, the same dict the real QPU gets.
    Given the whole fake chip, that Estimator simulates all its 156 qubits (~280 s per epoch job with trex);
    here each chain runs on its own ChainBackend (n qubits), with noisy statevector trajectories: the twirled
    circuits get few shots each (32 randomizations), where one density matrix per randomization was 3-5x slower.
    Epoch job of 10 chains (17/11/2026, one thread): none 3 s, trex 16 s, suppress 19 s. Checked against
    FakeChainCircuit with `none`: same error and contrast versus the ideal circuit (RMS ~0.03, slope ~0.99)."""

    evaluate = FakeChainCircuit.evaluate

    def __init__(self, backend, chains, n, gates, precision, options, threads=0):
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.executor_estimator import Estimator as ClientEstimator
        self.B, self.n, self.precision = len(chains), n, precision
        phi = ParameterVector("phi", n)
        qc = QuantumCircuit(n)
        for k in range(n):
            qc.ry(phi[k], k)
        for c, t in gates:
            qc.cx(c, t)
        self.isa, self.obs, self.phi_idx, self.estimators = [], [], [], []
        for chain in chains:
            sub = ChainBackend(backend, chain)
            isa = generate_preset_pass_manager(backend=sub, optimization_level=1,
                                               initial_layout=list(range(n))).run(qc)
            self.isa.append(isa)
            self.obs.append([SparsePauliOp.from_sparse_list([("Z", [k], 1.0)], num_qubits=n)
                             .apply_layout(isa.layout) for k in range(n)])
            order = list(isa.parameters)
            self.phi_idx.append([order.index(p) for p in phi])
            simulator = AerSimulator.from_backend(sub, method="statevector")   # noisy: one trajectory per shot
            simulator.set_options(max_parallel_threads=threads)
            self.estimators.append(ClientEstimator(mode=simulator, options=options))
        self.g2 = next(g for g in ("cz", "ecr", "cx") if g in backend.target.operation_names)
        self.n_qubits_used = self.B * n
        self.depth = max(isa.depth() for isa in self.isa)
        self.n_2q = sum(isa.count_ops().get(self.g2, 0) for isa in self.isa)
        self.n_2q_expected = self.B * len(gates)

    def run(self, angle_sets, job_ids, verbose=True):
        """angle_sets (T, B, n) -> <Z> (T, B, n). One local job per chain, counted as one job (one on the QPU)."""
        evs = []
        for b, (isa, obs, idx, est) in enumerate(zip(self.isa, self.obs, self.phi_idx, self.estimators)):
            vals = np.zeros((len(angle_sets), isa.num_parameters))
            vals[:, idx] = angle_sets[:, b]
            job = est.run([(isa, obs, vals[:, None, :])], precision=self.precision)
            evs.append(np.asarray(job.result()[0].data.evs))
            if b == 0:
                job_ids.append(job.job_id())
        return np.stack(evs, axis=1)


def mitigation_of(real, mitigation=None):
    """The preset in force: the given one, else REAL_MITIGATION on the real QPU and `none` on the fake."""
    return mitigation or (REAL_MITIGATION if real else "none")


def make_qpu(backend, chains, n_qubits, real, threads=0, mitigation=None):
    gates = cnot_gate_list(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS)
    mitigation = mitigation_of(real, mitigation)
    if real:
        return PackedCircuit(backend, chains, n_qubits, gates, PRECISION, False, MITIGATIONS[mitigation])
    if mitigation != "none":
        return FakeMitigatedCircuit(backend, chains, n_qubits, gates, PRECISION, MITIGATIONS[mitigation], threads)
    qpu = FakeChainCircuit(backend, chains, n_qubits, gates, PRECISION, None)
    qpu.estimator.backend.set_options(max_parallel_threads=threads)
    return qpu


def circuit_structure(n_qubits):
    gates = cnot_gate_list(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS)
    return repeated_cnot_network_matrix(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS), causal_cones(n_qubits, gates)


def pack_chains(backend, n_qubits, max_score):
    """As many disjoint chains of n qubits as fit on the chip (integer program over every chain of find_chains with
    score <= max_score), ties broken by the smaller total score; sorted best first. find_chains' greedy choice (best
    chain first, then the best disjoint one, ...) stops earlier: 10 chains of 10 qubits on ibm_aachen, against 12 here."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    _, _, ranked = find_chains(backend, n_qubits)
    ranked = [(s, c) for s, c in ranked if s <= max_score]
    S = np.array([s for s, _ in ranked])
    A = lil_matrix((backend.num_qubits, len(ranked)))
    for j, (_, c) in enumerate(ranked):
        for q in c:
            A[q, j] = 1
    res = milp(c=-(1.0 - S / (S.sum() + 1.0)), constraints=LinearConstraint(A.tocsr(), 0, 1),
               integrality=np.ones(len(ranked)), bounds=Bounds(0, 1), options=dict(time_limit=120))
    sel = sorted(np.flatnonzero(res.x > 0.5), key=lambda j: S[j])
    return [list(ranked[j][1]) for j in sel], S[sel].tolist()


def calibrate(backend, n_qubits, real, out_dir, mitigation=None, chains=None, session=None, samples=CAL_SAMPLES,
              packing="greedy"):
    """Chains of n qubits + calibration measurements (1 job), cached in out_dir. Chains with a
    bad error score or that do not follow the chip-average polynomial are dropped. The cache is
    valid for one mitigation preset only: the polynomial is fitted to what that preset returns.
    chains: use these chains as they are (no search, no screening), e.g. those of the unmitigated
    runs, so that a comparison of presets is on the same qubits. session: run the calibration job in it (real QPU).
    samples: calibration samples per output (parameter sets = samples / chains). packing: "greedy" (find_chains) or
    "max" (pack_chains)."""
    mitigation = mitigation_of(real, mitigation)
    path = os.path.join(out_dir, f"calibration_NQ{n_qubits}_{backend.name}.npz")
    if os.path.exists(path):
        cal = np.load(path)
        cached = str(cal["mitigation"]) if "mitigation" in cal else mitigation_of(real)   # older files: the default
        if cached != mitigation:
            raise SystemExit(f"{path} was measured with --mitigation {cached}, not {mitigation}: use another --out-dir.")
        return cal["chains"].tolist(), cal["angles"], cal["evs"]
    M, cones = circuit_structure(n_qubits)
    fixed = chains is not None
    if not fixed and packing == "max":
        chains, _ = pack_chains(backend, n_qubits, MAX_CHAIN_SCORE_PER_QUBIT * n_qubits)
    elif not fixed:
        chains, scores, _ = find_chains(backend, n_qubits)
        chains = [c for c, s in zip(chains, scores) if s <= MAX_CHAIN_SCORE_PER_QUBIT * n_qubits]
    qpu = make_qpu(backend, chains, n_qubits, real, mitigation=mitigation)
    if session is not None:
        qpu.set_mode(session)
    rng = np.random.default_rng(0)
    angle_sets = rng.uniform(-np.pi, np.pi, size=(math.ceil(samples / qpu.B), qpu.B, n_qubits))
    angles, evs = angle_sets.reshape(-1, n_qubits), qpu.run(angle_sets, [], verbose=False).reshape(-1, n_qubits)
    keep = np.ones(qpu.B, bool) if fixed else screen_chains(angles, evs, qpu.B, cones, M, CHAIN_OUTLIER)[0]
    rows = keep[np.arange(len(angles)) % qpu.B]
    chains = [c for c, k in zip(chains, keep) if k]
    os.makedirs(out_dir, exist_ok=True)
    np.savez(path, chains=np.array(chains), angles=angles[rows], evs=evs[rows], backend=backend.name, mitigation=mitigation)
    return chains, angles[rows], evs[rows]


# ----------------------------------------------------------------------
# One run: training with the QPU in the loop + validation on the QPU
# ----------------------------------------------------------------------
def run_one(n_qubits, seed, real=False, threads=0, tfin=TFIN, nu_true=NU_TRUE, cfg=CONFIG, epochs=EPOCHS, adj=None,
            backend_name=BACKEND, out_root=OUT_DIR, on_epoch=None, mitigation=None, val_points=VAL_POINTS,
            cal_samples=CAL_SAMPLES, packing="greedy", stop_at=None, resume=False):
    """Initialization as Inverser_quantum_exact_pinn_paralelo.run_one (set_seed,
    nu_init = np.random.random(), then the network). adj: ADJUSTMENTS or None (unadjusted).
    on_epoch: optional callback once per epoch (run/Inverser_runs_paralelo.py: progress bar).
    mitigation: preset of MITIGATIONS (None: REAL_MITIGATION on the real QPU, `none` on the fake).
    val_points, cal_samples, packing: validation points, calibration samples per output, chain packing
    (calibrate); the emulator runs use the defaults, the real QPU the REAL_* values (17/11/2026).
    stop_at: stop after this many epochs of the `epochs` schedule, validate (val_points) and save the whole state in
    checkpoint_<tag>.pt (the outputs of the partial run get the suffix _at<epochs done>; the run is not "done").
    resume: go on from checkpoint_<tag>.pt (same seed, epochs, chains, mitigation): the training continues exactly as
    without the stop (train's resume); the calibration is the cached one, so no calibration job is sent."""
    tc.set_num_threads(1)
    t0 = time.time()
    backend = get_backend(real, backend_name)
    out_dir = results_dir(tfin, nu_true, old=adj is None, out_root=out_root)
    os.makedirs(out_dir, exist_ok=True)
    mitigation = mitigation_of(real, mitigation)
    tag = f"{N_COLLOCATION}_NQ{n_qubits}_{seed}_E{epochs}"
    ckpt_path = os.path.join(out_dir, f"checkpoint_{tag}.pt")
    if resume and not os.path.exists(ckpt_path):
        raise SystemExit(f"--resume: {ckpt_path} not found")
    # real QPU: one Session for the whole run, the calibration job included (it does not wait in the queue twice)
    with (Session(backend=backend, max_time=SESSION_MAX_TIME) if real else nullcontext()) as session:
        chains, cal_angles, cal_evs = calibrate(backend, n_qubits, real, os.path.join(out_root, "quantico"), mitigation,
                                                session=session, samples=cal_samples, packing=packing)
        M, cones = circuit_structure(n_qubits)
        qpu = make_qpu(backend, chains, n_qubits, real, threads, mitigation)
        if real:
            qpu.set_mode(session)
        coefs, fit_rms, _ = fit_surrogate(cal_angles, cal_evs, cones, M)

        set_seed(seed)
        nu_init = float(np.random.random())
        model = build_model(n_qubits, tfin, nu_init)
        model.hard_ic = (adj or {}).get("hard_ic", True)
        model.quantum = HardwareSurrogateLayer(cones, coefs)
        job_ids, meas_angles, meas_evs = [], [cal_angles], [cal_evs]
        job_wall, t_before, resume_state, train_state = [], 0.0, None, {}
        if resume:
            ck = tc.load(ckpt_path, weights_only=False)
            for key, value in (("seed", seed), ("epochs", epochs), ("mitigation", mitigation), ("chains", chains)):
                if ck[key] != value:
                    raise SystemExit(f"--resume: {key} of the checkpoint is {ck[key]}, not {value}")
            model.load_state_dict(ck["model"])          # weights, nu and the polynomial coefficients
            nu_init, job_ids, job_wall, t_before = ck["nu_init"], ck["job_ids"], ck["job_wall"], ck["time_s"]
            meas_angles, meas_evs = ck["meas_angles"], ck["meas_evs"]
            resume_state = ck["train_state"]
            print(f"resuming {tag} at epoch {resume_state['epoch']} of {epochs} ({len(job_ids)} jobs so far)", flush=True)

        def measure(epoch, xt, xt_ic=None):
            """<Z> on the QPU at the stencil centres (and the IC points, same job) -> offset of their
            5 stencil points (and of the IC points, or None)."""
            if epoch and epoch % RECAL_EVERY == 0:
                model.quantum.set_coefs(fit_surrogate(np.concatenate(meas_angles), np.concatenate(meas_evs), cones, M)[0])
            points = xt[:N_COLLOCATION] if xt_ic is None else tc.cat([xt[:N_COLLOCATION], xt_ic])
            with tc.no_grad():
                angles = network_angles(model, points[:, :1], points[:, 1:])
                q_poly = model.quantum(angles)
            t_job = time.time()
            q_qpu, slot_angles, slot_evs = qpu.evaluate(angles.numpy(), job_ids, shift=epoch, verbose=False)
            job_wall.append(time.time() - t_job)
            meas_angles.append(slot_angles)
            meas_evs.append(slot_evs)
            delta = tc.as_tensor(q_qpu, dtype=q_poly.dtype) - q_poly
            return delta[:N_COLLOCATION].repeat(5, 1), (None if xt_ic is None else delta[N_COLLOCATION:])

        hist = train(model, seed, measure=measure, cfg=cfg, Tfin=tfin, nu_true=nu_true, epochs=epochs, adj=adj,
                     on_epoch=on_epoch, stop_at=stop_at, resume=resume_state, state_out=train_state)
        done = len(hist["nu"])

        # ---------- validation: the trained network running on the QPU ----------
        x_val = np.linspace(0.0, L, val_points, endpoint=False).reshape(-1, 1)
        x_t = tc.tensor(x_val, dtype=tc.get_default_dtype())
        t_t = tc.full_like(x_t, tfin)
        with tc.no_grad():
            angles = network_angles(model, x_t, t_t)
            u_sur = model(tc.cat([x_t, t_t], 1)).numpy()
            q_exact = ExactQuantumLayer(n_qubits, FIXED_LAYER, FIXED_LAYER_REPEATS, INPUT_ROTATION_AXIS)(angles)
            u_ideal = network_output(model, x_t, t_t, q_exact).numpy()
        t_job = time.time()
        q_qpu, _, _ = qpu.evaluate(angles.numpy(), job_ids, replicas=qpu.B, verbose=False)
        t_validation = time.time() - t_job
    with tc.no_grad():
        u_qpu = network_output(model, x_t, t_t, q_qpu).numpy()
    u_ref = exact_solution(x_val, tfin, nu=nu_true, L=L)

    nu_learned = float(model.learnable_parameter.detach())
    out = dict(n_qubits=n_qubits, seed=seed, backend=backend.name, mitigation=mitigation, Tfin=tfin, nu_true=nu_true,
               nu_init=nu_init, val_points=val_points, cal_samples=cal_samples, packing=packing,
               nu=nu_learned, nu_rel_err=abs(nu_learned - nu_true) / nu_true,
               rel_l2_T_qpu=relative_l2_error(u_qpu, u_ref), rel_l2_T_measured_model=relative_l2_error(u_sur, u_ref),
               rel_l2_T_ideal_circuit=relative_l2_error(u_ideal, u_ref), chains=len(chains),
               fit_rms_calibration=float(np.mean(fit_rms)), n_jobs=len(job_ids), adjustments=adj,
               epochs_done=done, epochs_schedule=epochs, resumed=bool(resume),
               epoch_job_wall_s_median=float(np.median(job_wall)) if job_wall else None,
               validation_job_wall_s=t_validation, time_s=t_before + time.time() - t0)

    # ---------- outputs ----------
    # the whole state, at a stop and at the end: to go on (--resume) or to validate again (--validate-only)
    tc.save(dict(model=model.state_dict(), train_state=train_state, meas_angles=meas_angles, meas_evs=meas_evs,
                 job_ids=job_ids, job_wall=job_wall, chains=chains, seed=seed, epochs=epochs, mitigation=mitigation,
                 nu_init=nu_init, time_s=out["time_s"], cfg=cfg, adjustments=adj, result=out), ckpt_path)
    print(f"epoch {done} of {epochs}: state saved in {ckpt_path}", flush=True)
    if done < epochs:      # stopped: the outputs of the partial run get the suffix _at<done>
        tag = f"{tag}_at{done}"
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
    plot_parameter_history(hist["nu"], nu_true,
                           title=f"Learned nu = {nu_learned:.5f} ({100 * out['nu_rel_err']:.2f}%) -- "
                                 f"{backend.name}, NQ={n_qubits}, seed {seed}",
                           save_path=os.path.join(out_dir, f"nu_history_{tag}.png"))
    plot_solution(x_val, {f"QPU ({backend.name})": u_qpu, "measured model": u_sur,
                          "same weights, ideal circuit": u_ideal}, ref=u_ref,
                  title=f"T={tfin}, NQ={n_qubits}, seed {seed}: rel. L2 QPU = {out['rel_l2_T_qpu']:.2e}",
                  save_path=os.path.join(out_dir, f"solution_{tag}.png"))
    return out


def validate_saved(n_qubits, seed, real=False, tfin=TFIN, nu_true=NU_TRUE, epochs=EPOCHS, adj=None, backend_name=BACKEND,
                   out_root=OUT_DIR, mitigation=None, val_points=VAL_POINTS):
    """The validation again, from checkpoint_<tag>.pt (written at a --stop-at and at the end of a run): the saved network
    and polynomial on the saved chains, one QPU job (job mode, no Session), val_points x points at t = T, each measured
    on every chain. Nothing of the run is overwritten: revalidation_<tag>_P<points>_<date-time>.json / .npz."""
    tc.set_num_threads(1)
    backend = get_backend(real, backend_name)
    out_dir = results_dir(tfin, nu_true, old=adj is None, out_root=out_root)
    mitigation = mitigation_of(real, mitigation)
    tag = f"{N_COLLOCATION}_NQ{n_qubits}_{seed}_E{epochs}"
    ck = tc.load(os.path.join(out_dir, f"checkpoint_{tag}.pt"), weights_only=False)
    if ck["mitigation"] != mitigation:
        raise SystemExit(f"--validate-only: the run used --mitigation {ck['mitigation']}, not {mitigation}")
    M, cones = circuit_structure(n_qubits)
    model = build_model(n_qubits, tfin, ck["nu_init"])
    model.hard_ic = (adj or {}).get("hard_ic", True)
    model.quantum = HardwareSurrogateLayer(cones, [np.zeros(3 ** len(c)) for c in cones])
    model.load_state_dict(ck["model"])
    qpu = make_qpu(backend, ck["chains"], n_qubits, real, mitigation=mitigation)
    x_val = np.linspace(0.0, L, val_points, endpoint=False).reshape(-1, 1)
    x_t = tc.tensor(x_val, dtype=tc.get_default_dtype())
    t_t = tc.full_like(x_t, tfin)
    with tc.no_grad():
        angles = network_angles(model, x_t, t_t)
        u_sur = model(tc.cat([x_t, t_t], 1)).numpy()
    job_ids = []
    q_qpu, _, _ = qpu.evaluate(angles.numpy(), job_ids, replicas=qpu.B, verbose=False)
    with tc.no_grad():
        u_qpu = network_output(model, x_t, t_t, q_qpu).numpy()
    u_ref = exact_solution(x_val, tfin, nu=nu_true, L=L)
    done = ck["train_state"]["epoch"]
    stamp = time.strftime("%Y%m%d-%H%M")
    name = f"revalidation_{tag}{'' if done == epochs else f'_at{done}'}_P{val_points}_{stamp}"
    nu = float(model.learnable_parameter.detach())
    out = dict(seed=seed, epochs_done=done, backend=backend.name, mitigation=mitigation, val_points=val_points,
               nu=nu, nu_rel_err=abs(nu - nu_true) / nu_true, rel_l2_T_qpu=relative_l2_error(u_qpu, u_ref),
               rel_l2_T_measured_model=relative_l2_error(u_sur, u_ref), job_ids=job_ids, date=stamp)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(out_dir, f"{name}.npz"), x=x_val, angles=angles.numpy(), q_qpu=q_qpu, u_qpu=u_qpu,
             u_measured_model=u_sur, u_exact=u_ref)
    print(f"revalidation ({val_points} points, epoch {done}): rel. L2 of u(x,T) with the QPU {out['rel_l2_T_qpu']:.4f}, "
          f"nu {nu:.5f} ({100 * out['nu_rel_err']:.2f}%) -> {os.path.join(out_dir, name)}.json", flush=True)
    return out


def summarize(out_dir, tfin, nu_true, adj=None, epochs=EPOCHS):
    """summary.json of every run saved in out_dir; returns the runs."""
    runs = sorted((json.load(open(p)) for p in glob.glob(os.path.join(out_dir, f"run_*_E{epochs}.json"))),
                  key=lambda r: (r["n_qubits"], r["seed"]))
    if not runs:          # e.g. a run stopped by --stop-at: only its partial outputs (_at<epoch>) exist
        return runs
    for r in runs:
        for k in ("cfg", "job_ids", "chain_list"):
            r.pop(k, None)
    err = np.array([r["nu_rel_err"] for r in runs])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(config=CONFIG, adjustments=adj, nu_true=nu_true, Tfin=tfin, n_collocation=N_COLLOCATION, epochs=epochs,
                       data="exact solution at the collocation points", nu_rel_err_median=float(np.median(err)),
                       nu_rel_err_mean=float(err.mean()), nu_rel_err_max=float(err.max()), runs=runs), f, indent=1)
    return runs


def _run_job(job):
    return run_one(*job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=list(QUBITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--real", action="store_true", help="QPU real (padrão: o fake, local)")
    parser.add_argument("--backend", default=BACKEND, choices=list(FAKE_BACKENDS), help=f"QPU (padrão {BACKEND})")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"épocas (padrão {EPOCHS})")
    parser.add_argument("--out-dir", default=OUT_DIR, help=f"pasta das saídas (padrão {OUT_DIR})")
    parser.add_argument("--workers", type=int, default=6, help="runs em paralelo no fake")
    parser.add_argument("--mitigation", choices=tuple(MITIGATIONS), default=None,
                        help=f"supressão/mitigação de erro (padrão: {REAL_MITIGATION} no QPU real, none no fake); "
                             "no fake, um preset diferente de none grava em <out-dir>_<preset>")
    parser.add_argument("--tfin", type=float, default=TFIN, help=f"tempo final T (padrão {TFIN})")
    parser.add_argument("--val-points", type=int, default=None,
                        help=f"--real: pontos de validação (padrão {REAL_VAL_POINTS}; o fake usa {VAL_POINTS})")
    parser.add_argument("--stop-at", type=int, default=None,
                        help="--real: para depois desta época (do cronograma de --epochs), valida e salva o estado")
    parser.add_argument("--resume", action="store_true", help="--real: continua do checkpoint salvo por --stop-at")
    parser.add_argument("--validate-only", action="store_true",
                        help="--real: só refaz a validação a partir do checkpoint salvo (um job, sem treino)")
    parser.add_argument("--cal-samples", type=int, default=None,
                        help=f"--real: amostras de calibração por saída (padrão {REAL_CAL_SAMPLES}; o fake usa {CAL_SAMPLES})")
    parser.add_argument("--nu", type=float, default=NU_TRUE, help=f"viscosidade verdadeira (padrão {NU_TRUE})")
    parser.add_argument("--treino-antigo", action="store_true",
                        help="nos casos ajustados (T = 1, nu = 0.01), o treino sem os ajustes (-> subpasta treino_antigo)")
    args = parser.parse_args()
    adj = adjustments_for(args.tfin, args.nu, args.treino_antigo)
    # a real-QPU run has the same file names as the emulator runs: never write it over them
    if args.real and args.out_dir == OUT_DIR:
        args.out_dir = os.path.join("data", "Inverso_real", args.backend)

    mitigation = mitigation_of(args.real, args.mitigation)
    # a mitigated fake run has the same file names as the raw one: it goes to its own folder
    if not args.real and mitigation != "none" and not args.out_dir.endswith(f"_{mitigation}"):
        args.out_dir = f"{args.out_dir}_{mitigation}"

    backend = get_backend(args.real, args.backend)
    out_dir = results_dir(args.tfin, args.nu, old=adj is None, out_root=args.out_dir)
    jobs = [(n, s) for n in args.qubits for s in args.seeds]
    if args.real and args.validate_only:
        for n, s in jobs:
            validate_saved(n, s, real=True, tfin=args.tfin, nu_true=args.nu, epochs=args.epochs, adj=adj,
                           backend_name=args.backend, out_root=args.out_dir, mitigation=mitigation,
                           val_points=args.val_points or REAL_VAL_POINTS)
        raise SystemExit(0)
    if args.real:           # one Session per run, one after the other; the calibration runs inside the first one
        for n, s in jobs:
            run_one(n, s, real=True, tfin=args.tfin, nu_true=args.nu, epochs=args.epochs, adj=adj,
                    backend_name=args.backend, out_root=args.out_dir, mitigation=mitigation,
                    val_points=args.val_points or REAL_VAL_POINTS, cal_samples=args.cal_samples or REAL_CAL_SAMPLES,
                    packing=REAL_PACKING, stop_at=args.stop_at, resume=args.resume)
        jobs = []
    for n in sorted({n for n, _ in jobs}):   # fake: once per qubit count, before the runs that share it
        calibrate(backend, n, False, os.path.join(args.out_dir, "quantico"), mitigation)
    if jobs:
        threads = max(1, (os.cpu_count() or 2) // args.workers)
        # spawn, not fork (the Linux default): the parent already holds OpenMP threads from the
        # calibration, and forked workers deadlock on their locks
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            list(pool.map(_run_job, [(n, s, False, threads, args.tfin, args.nu, CONFIG, args.epochs, adj,
                                      args.backend, args.out_dir, None, mitigation) for n, s in jobs]))

    runs = summarize(out_dir, args.tfin, args.nu, adj, args.epochs)
    print(f"{backend.name} (mitigation {mitigation}): NU={args.nu}, T={args.tfin}, N_col={N_COLLOCATION}, {args.epochs} epochs, "
          f"{'adjusted' if adj else 'unadjusted'} training -- learned nu (relative error)")
    for n in sorted({r["n_qubits"] for r in runs}):
        print(f"  NQ{n:<2d} " + "  ".join(f"seed {r['seed']}: {r['nu']:.4f} ({100 * r['nu_rel_err']:.1f}%)"
                                       for r in runs if r["n_qubits"] == n))
