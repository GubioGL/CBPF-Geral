import argparse
import json
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # headless backend -- safe across worker processes
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
from tqdm import tqdm

from main.functions import (
    cnot_cosine_expectations,
    cnot_cosine_product_expressions,
    repeated_cnot_network_matrix,
)

# ---- physical parameters ---------------------------------------------------
L, NU = 1.0, 0.01
# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3
# ---- collocation sampling ---------------------------------------------------
N_DATA = 100  # fixed data points at t=Tfin, needed to identify nu
COLLOCATION_DISTRIBUTION = "random"        # training collocation points
STOPPING_DISTRIBUTION = "uniform_exact"    # fixed grid used for the stopping/monitor loss
# ---- training ---------------------------------------------------------------
EPOCHS, LR = 20000, 1.0e-2
LBFGS_EPOCHS, LBFGS_LR = 1000, .5
# ---- architecture -------------------------------------------------------------
# pre -> quantum -> post; the trainable "net" bottleneck of the classical
# script is replaced by a fixed (parameter-free) exact CNOT-cosine feature
# map with N_QUBITS wires -- swept per job below, like INTRA_NEURON in the
# classical script.
PRE_NEURONS, PRE_DEPTH = 10, 1
POST_NEURONS, POST_DEPTH = 10, 1
ANGLE_SCALE = math.pi
FIXED_LAYER = "linear_cnot"        # see FIXED_LAYERS below for every option
FIXED_LAYER_REPEATS = 1
# The exact cosine-product formula only holds for Rx/Ry product encodings --
# Rz acting on |0> leaves <Z_i> independent of the angle, so it is rejected
# below rather than silently producing a degenerate feature map.
INPUT_ROTATION_AXIS = "ry"         # 'rx' or 'ry'

OUT_DIR = "data/quantum_inverse_exact"

# ---- parallelization --------------------------------------------------------
# Unlike a statevector simulation, the exact cosine-product feature map costs
# O(N_QUBITS**2) per batch element (not O(2**N_QUBITS)), so every job here is
# cheap regardless of N_QUBITS -- no "small"/"large" qubit-count split is
# needed, just one worker pool sized for whichever device is in use.
N_WORKERS_CPU = max(1, (os.cpu_count() or 1) - 1)
N_WORKERS_PER_GPU = 4  # concurrent worker processes sharing each GPU, by default


def set_seed(seed=42):
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)

def initial_condition(x, L=1.0):
    """u0(x) = sin(2*pi*x/L). Accepts a NumPy array or a Torch tensor."""
    if isinstance(x, tc.Tensor):
        return tc.sin(2.0 * math.pi * x / L)
    return np.sin(2.0 * np.pi * x / L)

# ----------------------------------------------------------------------
# Reference solution (finite differences, used only for comparison/plots)
# ----------------------------------------------------------------------
def solve_burgers_fd(L=1.0, Tfin=0.1, nu=0.01, Nx=512, dt=None, safety=0.5):
    """Explicit upwind finite-difference solution of u_t + u u_x = nu u_xx
    on a periodic domain [0, L).

    dt: time step. If None (default), it is computed automatically from the
    explicit-scheme stability limits -- dt <= dx^2/(2*nu) for the diffusion
    term and dt <= dx/max|u0| for upwind advection -- scaled by `safety`
    (<1, for margin). A fixed dt that ignores nu (e.g. a large nu with a
    dt tuned for a small one) makes the scheme diverge: u grows without
    bound, overflows, and turns into inf/nan (RuntimeWarnings)."""
    x = np.linspace(0.0, L, Nx, endpoint=False)
    dx = L / Nx
    u = initial_condition(x, L=L).copy()
    if dt is None:
        dt_diffusion = dx**2 / (2.0 * nu) if nu > 0 else np.inf
        dt_advection = dx / max(np.max(np.abs(u)), 1.0e-12)
        dt = safety * min(dt_diffusion, dt_advection)

    times = np.arange(0.0, Tfin + 0.5 * dt, dt)
    U_hist = np.zeros((len(times), Nx))
    U_hist[0] = u

    for n in range(1, len(times)):
        up, um = np.roll(u, -1), np.roll(u, 1)
        ux_back, ux_fwd = (u - um) / dx, (up - u) / dx
        ux = np.where(u >= 0.0, ux_back, ux_fwd)
        uxx = (up - 2.0 * u + um) / dx**2
        u = u + dt * (-u * ux + nu * uxx)
        U_hist[n] = u

    return x, times, U_hist

def reference_at_time(t_eval, t_ref, x_ref, U_ref, x_query, L=1.0):
    """Periodic interpolation of the FDM reference at the closest stored time."""
    idx = int(np.argmin(np.abs(t_ref - t_eval)))
    xp = np.concatenate([x_ref, [L]])
    fp = np.concatenate([U_ref[idx], [U_ref[idx][0]]])
    return np.interp(x_query.ravel(), xp, fp).reshape(-1, 1)

def reference_pde_residual(x_f, t_f, x_ref, t_ref, U_ref, L, nu, hx, ht):
    """MSE of the PDE residual evaluated on the FDM reference itself, at the
    same collocation points/stencil used by `train`'s loss. Used as the
    training's stopping target: once the PINN's loss reaches this value it
    matches the reference's own discretization error, and pushing it lower
    is just chasing noise in the baseline."""
    x_f = np.asarray(x_f, dtype=np.float64).reshape(-1)
    t_f = np.asarray(t_f, dtype=np.float64).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    t_ref = np.asarray(t_ref, dtype=np.float64)

    dx = x_ref[1] - x_ref[0]
    dt_ref = t_ref[1] - t_ref[0]
    nx, nt = len(x_ref), len(t_ref)

    def u_ref_interp(x, t):
        x = np.mod(x, L)
        t = np.clip(t, t_ref[0], t_ref[-1])

        fx = x / dx
        ix0 = np.floor(fx).astype(int) % nx
        ix1 = (ix0 + 1) % nx
        wx = fx - np.floor(fx)

        ft = (t - t_ref[0]) / dt_ref
        it0 = np.clip(np.floor(ft).astype(int), 0, nt - 2)
        it1 = it0 + 1
        wt = ft - it0

        u0 = U_ref[it0, ix0] * (1.0 - wx) + U_ref[it0, ix1] * wx
        u1 = U_ref[it1, ix0] * (1.0 - wx) + U_ref[it1, ix1] * wx
        return u0 * (1.0 - wt) + u1 * wt

    u_0 = u_ref_interp(x_f, t_f)
    u_xp, u_xm = u_ref_interp(x_f + hx, t_f), u_ref_interp(x_f - hx, t_f)
    u_tp, u_tm = u_ref_interp(x_f, t_f + ht), u_ref_interp(x_f, t_f - ht)

    u_t = (u_tp - u_tm) / (2.0 * ht)
    u_x = (u_xp - u_xm) / (2.0 * hx)
    u_xx = (u_xp - 2.0 * u_0 + u_xm) / (hx**2)
    return float(np.mean((u_t + u_0 * u_x - nu * u_xx) ** 2))

# ----------------------------------------------------------------------
# Collocation points
# ----------------------------------------------------------------------
def sample_collocation_points(n_points, L, T, ht, seed=None, distribution="random"):
    """x in [0,L), t in [ht, T-ht] (interior points, needed for centered
    differences).

    "random": n_points drawn uniformly at random; pass `seed` for a
    reproducible draw.
    "uniform_exact": a uniform grid of exactly n_points (n_t =
    round(sqrt(n_points)) values along t, n_x = ceil(n_points/n_t) along x),
    used here as the fixed validation/stopping set.
    """
    if distribution == "random":
        generator = tc.Generator().manual_seed(seed) if seed is not None else None
        x = tc.rand((n_points, 1), generator=generator) * L
        t = ht + tc.rand((n_points, 1), generator=generator) * (T - 2.0 * ht)
    elif distribution == "uniform_exact":
        n_t = max(1, round(math.sqrt(n_points)))
        n_x = max(1, math.ceil(n_points / n_t))
        x_lin = tc.linspace(0.0, L, n_x + 1)[:-1]
        t_lin = tc.linspace(ht, T - ht, n_t)
        x_grid, t_grid = tc.meshgrid(x_lin, t_lin, indexing="ij")
        x = x_grid.reshape(-1, 1)[:n_points]
        t = t_grid.reshape(-1, 1)[:n_points]
    else:
        raise ValueError("distribution must be 'random' or 'uniform_exact'.")
    return x, t

# ----------------------------------------------------------------------
# Exact (parameter-free) entangling layer, evaluated analytically via GF(2)
# binary algebra instead of a simulated circuit (main/functions.py).
# ----------------------------------------------------------------------
FIXED_LAYERS = (
    "linear_cnot", "reverse_linear_cnot", "ring_cnot", "brickwork_cnot",
    "symmetric_cnot_ladder",
)

class ExactQuantumLayer(nn.Module):
    """Parameter-free quantum feature layer evaluated exactly through GF(2)
    binary algebra (`cnot_cosine_expectations`/`repeated_cnot_network_matrix`
    in main/functions.py; see `run_quantumExact.ipynb`), instead of a
    simulated circuit. For an Rx/Ry product encoding followed by fixed CNOT
    layers,

        <Z_i> = product_j cos(angle_j)**M[i,j]

    where M is the repeated-CNOT binary matrix over GF(2). This is pure,
    differentiable Torch tensor arithmetic -- no statevector, no shots, no
    quantum-simulation library -- so it costs O(N_QUBITS**2) per batch
    element (not O(2**N_QUBITS)) and runs natively on GPU."""

    def __init__(self, n_qubits, fixed_layer="linear_cnot", fixed_layer_repeats=1,
                 input_rotation_axis="ry"):
        super().__init__()
        input_rotation_axis = str(input_rotation_axis).lower()
        if input_rotation_axis not in ("rx", "ry"):
            raise ValueError(
                "The exact cosine-product feature map supports only 'rx' and "
                "'ry' input rotations -- Rz acting on |0> leaves <Z_i> "
                "independent of the angle, so it cannot be used here."
            )
        self.n_qubits = int(n_qubits)
        self.input_rotation_axis = input_rotation_axis

        fixed_layer = str(fixed_layer).lower()
        if fixed_layer == "inverse_linear_cnot":
            fixed_layer = "reverse_linear_cnot"
        self.fixed_layer = fixed_layer
        self.fixed_layer_repeats = int(fixed_layer_repeats)

        binary_matrix = repeated_cnot_network_matrix(
            n_qubits=self.n_qubits, network_kind=self.fixed_layer,
            n_layers=self.fixed_layer_repeats,
        )
        # Persistent buffer: follows the model across .to(device) calls and
        # checkpoints like a parameter, but is never trained (fixed circuit).
        self.register_buffer(
            "cnot_matrix", tc.as_tensor(binary_matrix, dtype=tc.bool), persistent=True)

    def forward(self, angles):
        return cnot_cosine_expectations(angles, self.cnot_matrix)

    def cosine_expressions(self, angle_symbol="theta"):
        """Return the exact product-of-cosines formula for every output channel."""
        return cnot_cosine_product_expressions(
            self.cnot_matrix.detach().cpu().numpy().astype(np.uint8),
            angle_symbol=angle_symbol,
        )

class QuantumExactInversePINN(nn.Module):
    """Hybrid HPINN for the inverse problem: classical pre-net -> bounded
    rotation angles -> fixed (parameter-free) exact CNOT-cosine feature map
    -> classical post-net, with a learnable scalar `learnable_parameter`
    (nu). Same shape as `Inverser_quantum_pinn_paralelo.py`'s `QuantumPINN`;
    only the quantum bottleneck implementation changes (exact instead of
    simulated -- see `ExactQuantumLayer` above)."""

    def __init__(self, pre_neurons, pre_depth, n_qubits, post_neurons, post_depth,
                 L=1.0, Tfin=0.1, nu_init=0.1, angle_scale=math.pi,
                 fixed_layer="linear_cnot", fixed_layer_repeats=1,
                 input_rotation_axis="ry"):
        super().__init__()
        self.L, self.Tfin = float(L), float(Tfin)
        self.angle_scale = float(angle_scale)
        # Kwargs needed to reconstruct this model (cls(**config) must work);
        # `nu_init` is left out on purpose -- it only seeds the training run,
        # the loaded state_dict overwrites `learnable_parameter` anyway.
        self.config = dict(
            pre_neurons=pre_neurons, pre_depth=pre_depth, n_qubits=n_qubits,
            post_neurons=post_neurons, post_depth=post_depth, L=self.L, Tfin=self.Tfin,
            angle_scale=self.angle_scale, fixed_layer=fixed_layer,
            fixed_layer_repeats=fixed_layer_repeats, input_rotation_axis=input_rotation_axis,
        )

        def mlp(n_in, hidden, depth, n_out):
            layers = [nn.Linear(n_in, hidden), nn.Tanh()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            layers += [nn.Linear(hidden, n_out)]
            return nn.Sequential(*layers)

        self.pre = mlp(3, pre_neurons, pre_depth, n_qubits)
        self.quantum = ExactQuantumLayer(
            n_qubits=n_qubits, fixed_layer=fixed_layer,
            fixed_layer_repeats=fixed_layer_repeats,
            input_rotation_axis=input_rotation_axis,
        )
        self.post = mlp(n_qubits, post_neurons, post_depth, 1)
        self.learnable_parameter = nn.Parameter(tc.tensor(float(nu_init)))

    def forward(self, xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        features = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), t / self.Tfin],
            dim=1,
        )
        angles = self.angle_scale * tc.tanh(self.pre(features))
        q = self.quantum(angles).to(dtype=self.post[0].weight.dtype)
        out = self.post(q)
        return initial_condition(x, L=self.L) + (t / self.Tfin) * out  # hard-constrains u(x,0)=u0(x)

    def save(self, path, training_params=None):
        """Saves weights + config to `path` (state_dict + reconstruction
        kwargs), plus a human-readable `<path>_config.json` alongside it so
        the run configuration can be inspected without loading torch."""
        path = str(path)
        payload = {"class_name": type(self).__name__, "config": self.config, "state_dict": self.state_dict()}
        if training_params is not None:
            payload["training_params"] = training_params
        tc.save(payload, path)

        config_path = os.path.splitext(path)[0] + "_config.json"
        json_payload = {"class_name": type(self).__name__, "config": self.config}
        if training_params is not None:
            json_payload["training_params"] = training_params
        with open(config_path, "w") as f:
            json.dump(json_payload, f, indent=2)

    @classmethod
    def load(cls, path, map_location="cpu"):
        """Reconstructs the model from a checkpoint saved with `save`, ready
        for further training or inference."""
        checkpoint = tc.load(path, map_location=map_location, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model

def stencil(x, t, L, hx, ht):
    """Shifted neighbor points x+-hx, t+-ht for the centered-difference stencil."""
    return tc.remainder(x + hx, L), tc.remainder(x - hx, L), t + ht, t - ht

def stencil_inputs(x, t, L, hx, ht):
    """The 5 stencil points (x,t), (x+-hx,t), (x,t+-ht) stacked into a single
    (5N, 2) batch, so the network runs one forward pass instead of five --
    with networks this small the cost is per-call overhead, not arithmetic.
    Also accepts (S, N, 1) inputs (one point set per run) -> (S, 5N, 2)."""
    xp, xm, tp, tm = stencil(x, t, L, hx, ht)
    return tc.cat([tc.cat(pair, dim=-1) for pair in ((x, t), (xp, t), (xm, t), (x, tp), (x, tm))], dim=-2)

def residual_terms(u, n, hx, ht):
    """Splits a forward pass over `stencil_inputs` (first 5n rows of u, along
    the points axis) into (A, B), with the PDE residual
    u_t + u*u_x - nu*u_xx = A - nu*B."""
    u0, uxp, uxm, utp, utm = u.narrow(-2, 0, 5 * n).split(n, dim=-2)
    u_t = (utp - utm) / (2.0 * ht)
    u_x = (uxp - uxm) / (2.0 * hx)
    u_xx = (uxp - 2.0 * u0 + uxm) / hx**2
    return u_t + u0 * u_x, u_xx

def pde_residual(model, xt_stencil, n, hx, ht):
    """u_t + u*u_x - nu*u_xx at the n points stacked in `xt_stencil`, with
    nu = model.learnable_parameter."""
    a, b = residual_terms(model(xt_stencil), n, hx, ht)
    return a - model.learnable_parameter * b

def lbfgs_refine(model, xt_f, n_f, hist, best_state, best_loss, target_reached,
                 lbfgs_epochs, lbfgs_lr, target_loss=None, xt_val=None, n_val=None,
                 xt_data=None, u_data=None, hx=1.0e-3, ht=1.0e-3):
    """L-BFGS phase of `train`, run after AdamW on a single model: starts from
    `best_state` (the best AdamW snapshot) on the points AdamW ended on
    (xt_f, stacked by `stencil_inputs`), appends to `hist` and loads the best
    state found into `model`. Returns (best_loss, target_reached)."""

    def monitor():
        with tc.no_grad():
            if xt_val is not None:
                return float(tc.mean(pde_residual(model, xt_val, n_val, hx, ht) ** 2))
            return float(tc.mean(pde_residual(model, xt_f, n_f, hx, ht) ** 2))

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    params = [p for p in model.parameters() if p.requires_grad]
    model.load_state_dict(best_state)  # hand L-BFGS the best basin Adam found

    # Once u(x,t) already matches the reference's own precision, freeze
    # it and let L-BFGS refine only nu, to sharpen its last digits --
    # otherwise (target not yet reached) L-BFGS still refines everything.
    only_nu = target_reached
    if only_nu:
        for p in params:
            p.requires_grad_(p is model.learnable_parameter)
        lbfgs_params = [model.learnable_parameter]
    else:
        lbfgs_params = params

    lbfgs_opt = tc.optim.LBFGS(
        lbfgs_params, lr=lbfgs_lr, max_iter=300, history_size=100, line_search_fn="strong_wolfe")
    patience, tol, stall = 50, 1.0e-9, 0

    def closure():
        lbfgs_opt.zero_grad()
        # One forward pass over the stencil and data points together.
        u = model(xt_f if xt_data is None else tc.cat([xt_f, xt_data], dim=0))
        a, b = residual_terms(u, n_f, hx, ht)
        loss = tc.mean((a - model.learnable_parameter * b) ** 2)
        if xt_data is not None:
            loss = loss + tc.mean(abs(u[5 * n_f:] - u_data))
        loss.backward()
        return loss

    for _ in range(lbfgs_epochs):
        lbfgs_opt.step(closure)
        loss_value = monitor()
        hist["loss"].append(loss_value)
        hist["nu"].append(float(model.learnable_parameter.detach()))
        if loss_value < best_loss - tol:
            best_loss, best_state, stall = loss_value, snapshot(), 0
        else:
            stall += 1
        # In nu-only mode, u is already at the target precision, so this
        # would trip on the very first iteration and cut the refinement
        # short -- only apply it while still refining the whole network.
        if not only_nu and target_loss is not None and loss_value <= target_loss:
            target_reached = True
            break
        if stall >= patience:
            break

    for p in params:
        p.requires_grad_(True)  # restore full trainability before returning

    model.load_state_dict(best_state)  # never return worse than the best point found
    return best_loss, target_reached

def train(model, x_f, t_f, epochs, lr, resample_fn=None,
          lbfgs_epochs=1000, lbfgs_lr=0.5, target_loss=None,
          validation_points=None, data_points=None, L=1.0, hx=1.0e-3, ht=1.0e-3):
    """resample_fn: optional callable returning a fresh (x_f, t_f) pair,
    called every AdamW epoch (used to train on freshly drawn random points).
    validation_points: optional fixed (x, t) used to monitor/stop training
    instead of the (possibly resampled) training points.
    data_points: optional fixed (x_data, t_data, u_data) supervised targets,
    added to the training loss as MAE(model(x_data, t_data), u_data). Kept
    fixed across epochs, unlike x_f, t_f which may change every epoch.
    target_loss: stop as soon as the monitored (PDE-residual-only) loss
    reaches this value.
    Returns (model, hist), hist["nu"] holding the learned nu at every step,
    and the best (lowest monitored loss) snapshot loaded into `model`.

    All tensors (x_f, t_f, validation_points, data_points) must already be
    on the same device as `model` -- this function performs no device
    transfers itself."""

    n_f, xt_f = x_f.shape[0], stencil_inputs(x_f, t_f, L, hx, ht)
    xt_val = n_val = None
    if validation_points is not None:
        x_val, t_val = validation_points
        n_val, xt_val = x_val.shape[0], stencil_inputs(x_val, t_val, L, hx, ht)

    xt_data = u_data = None
    if data_points is not None:
        x_data, t_data, u_data = data_points
        xt_data = tc.cat([x_data, t_data], dim=1)

    def train_loss():
        """PDE-residual MAE + data MAE, from one forward pass over the stencil
        and data points together."""
        u = model(xt_f if xt_data is None else tc.cat([xt_f, xt_data], dim=0))
        a, b = residual_terms(u, n_f, hx, ht)
        loss = tc.mean(abs(a - model.learnable_parameter * b))
        if xt_data is not None:
            loss = loss + tc.mean(abs(u[5 * n_f:] - u_data))
        return loss

    def monitor():
        with tc.no_grad():
            if xt_val is not None:
                return float(tc.mean(pde_residual(model, xt_val, n_val, hx, ht) ** 2))
            return float(tc.mean(pde_residual(model, xt_f, n_f, hx, ht) ** 2))

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = tc.optim.AdamW(params, lr=lr, weight_decay=1.0e-4)

    best_state, best_loss = snapshot(), float("inf")
    hist = {"loss": [], "nu": []}
    target_reached = False

    for _ in range(epochs):
        if resample_fn is not None:
            x_f, t_f = resample_fn()
            xt_f = stencil_inputs(x_f, t_f, L, hx, ht)

        loss = train_loss()
        opt.zero_grad()
        loss.backward()
        tc.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        loss_value = monitor()
        hist["loss"].append(loss_value)
        hist["nu"].append(float(model.learnable_parameter.detach()))
        if loss_value < best_loss:
            best_loss, best_state = loss_value, snapshot()
        if target_loss is not None and loss_value <= target_loss:
            target_reached = True
            break

    if lbfgs_epochs > 0:
        best_loss, target_reached = lbfgs_refine(
            model, xt_f, n_f, hist, best_state, best_loss, target_reached,
            lbfgs_epochs, lbfgs_lr, target_loss=target_loss, xt_val=xt_val, n_val=n_val,
            xt_data=xt_data, u_data=u_data, hx=hx, ht=ht,
        )
    else:
        model.load_state_dict(best_state)  # never return worse than the best point found
    hist["target_reached"] = target_reached
    hist["best_loss"] = best_loss
    return model, hist

# ----------------------------------------------------------------------
# Batched AdamW phase: several independent runs trained together
# ----------------------------------------------------------------------
class BatchedQuantumExactPINN(nn.Module):
    """S independent QuantumExactInversePINNs (same architecture), evaluated
    together. Each Linear layer's weights are stacked along a leading run
    axis, so one batched matmul per layer serves all S networks; the fixed
    CNOT-cosine layer has no parameters and is the same for every run. Every
    run keeps its own weights and its own nu and still takes only (x, t) as
    input -- nothing is shared between runs. With networks this small the
    cost is per-call overhead, so S runs together cost little more than one."""

    def __init__(self, models):
        super().__init__()
        ref = models[0]
        self.L, self.Tfin, self.angle_scale = ref.L, ref.Tfin, ref.angle_scale
        self.register_buffer("cnot_matrix", ref.quantum.cnot_matrix.clone())
        self.linear_names, self.weights, self.biases = {}, nn.ParameterList(), nn.ParameterList()
        for block in ("pre", "post"):
            # (name in QuantumExactInversePINN, tanh after it?) for every Linear
            layers = []
            seq = getattr(ref, block)
            for i, layer in enumerate(seq):
                if not isinstance(layer, nn.Linear):
                    continue
                name = f"{block}.{i}"
                layers.append((name, len(self.weights), i + 1 < len(seq) and isinstance(seq[i + 1], nn.Tanh)))
                linears = [m.get_submodule(name) for m in models]
                self.weights.append(nn.Parameter(tc.stack([lin.weight.detach().T for lin in linears])))  # (S, in, out)
                self.biases.append(nn.Parameter(tc.stack([lin.bias.detach()[None, :] for lin in linears])))  # (S, 1, out)
            self.linear_names[block] = layers
        self.learnable_parameter = nn.Parameter(
            tc.stack([m.learnable_parameter.detach() for m in models]).view(-1, 1, 1))  # (S, 1, 1)

    def _mlp(self, block, h):
        for _, k, tanh_after in self.linear_names[block]:
            h = tc.baddbmm(self.biases[k], h, self.weights[k])
            if tanh_after:
                h = tc.tanh(h)
        return h

    def forward(self, xt):
        """xt: (S, M, 2) -- M points (x, t) for each of the S runs. Same
        computation as QuantumExactInversePINN.forward, run by run."""
        x, t = xt[..., 0:1], xt[..., 1:2]
        features = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), t / self.Tfin],
            dim=-1,
        )
        angles = self.angle_scale * tc.tanh(self._mlp("pre", features))
        S, M, n_qubits = angles.shape
        q = cnot_cosine_expectations(angles.reshape(S * M, n_qubits), self.cnot_matrix)
        out = self._mlp("post", q.reshape(S, M, n_qubits).to(dtype=self.weights[-1].dtype))
        return initial_condition(x, L=self.L) + (t / self.Tfin) * out  # hard-constrains u(x,0)=u0(x)

    def run_state_dict(self, i):
        """Run i's state in QuantumExactInversePINN's state_dict format, on the CPU."""
        sd = {
            "learnable_parameter": self.learnable_parameter[i, 0, 0].detach().cpu().clone(),
            "quantum.cnot_matrix": self.cnot_matrix.cpu().clone(),
        }
        for layers in self.linear_names.values():
            for name, k, _ in layers:
                sd[f"{name}.weight"] = self.weights[k][i].detach().T.contiguous().cpu()
                sd[f"{name}.bias"] = self.biases[k][i, 0].detach().cpu().clone()
        return sd

def train_batched_adam(model, x_f, t_f, epochs, lr, resample_fn=None, target_loss=None,
                       validation_points=None, data_points=None, L=1.0, hx=1.0e-3, ht=1.0e-3,
                       check_every=50):
    """AdamW phase of `train` for a BatchedQuantumExactPINN, applied to every
    run independently. Summing the per-run losses gives each run exactly its
    own gradient, and AdamW acts elementwise, so each run follows its own
    optimization. What `train` does globally is done per run here: gradient
    clipping, the best snapshot and early stopping (a run that reached
    `target_loss` stops recording and keeps its best state and the points it
    stopped on). The L-BFGS phase is not batched -- its line search and
    curvature history would couple the runs -- see `lbfgs_refine`.

    x_f, t_f: (S, N, 1), one point set per run; resample_fn returns the same
    shape. validation_points and data_points are shared by all runs, (N, 1).
    check_every: how often (steps) to check whether every run has already
    stopped, to end early -- only saves time, results don't depend on it.
    Returns (model, runs): `model` holds each run's best AdamW state, and
    runs[i] = dict(hist, best_loss, target_reached, stop_x, stop_t) with
    stop_x/stop_t (N, 1) the points run i ended on."""
    nu = model.learnable_parameter
    S, device = nu.shape[0], nu.device
    params = list(model.parameters())
    per_run = lambda p: (-1,) + (1,) * (p.dim() - 1)  # view to broadcast a (S,) tensor over p
    per_run_mean = lambda r: tc.mean(r, dim=(1, 2))  # (S, M, 1) -> (S,)

    n_f, xt_f = x_f.shape[-2], stencil_inputs(x_f, t_f, L, hx, ht)
    xt_val = n_val = None
    if validation_points is not None:
        x_val, t_val = validation_points
        n_val, xt_val = x_val.shape[-2], stencil_inputs(x_val, t_val, L, hx, ht).expand(S, -1, -1)

    xt_data = u_data = None
    if data_points is not None:
        x_data, t_data, u_data = data_points
        xt_data = tc.cat([x_data, t_data], dim=-1).expand(S, -1, -1)

    def train_loss():
        u = model(xt_f if xt_data is None else tc.cat([xt_f, xt_data], dim=1))
        a, b = residual_terms(u, n_f, hx, ht)
        loss = per_run_mean(abs(a - nu * b))
        if xt_data is not None:
            loss = loss + per_run_mean(abs(u[:, 5 * n_f:] - u_data))
        return loss.sum()

    def monitor():
        with tc.no_grad():
            xt, n = (xt_val, n_val) if xt_val is not None else (xt_f, n_f)
            a, b = residual_terms(model(xt), n, hx, ht)
            return per_run_mean((a - nu * b) ** 2)

    def clip_grad_norm_per_run(max_norm=1.0):
        """tc.nn.utils.clip_grad_norm_(max_norm), on each run's own gradient."""
        norms = tc.sqrt(sum(p.grad.pow(2).flatten(1).sum(1) for p in params))
        coef = tc.clamp(max_norm / (norms + 1.0e-6), max=1.0)
        for p in params:
            p.grad.mul_(coef.view(per_run(p)))

    opt = tc.optim.AdamW(params, lr=lr, weight_decay=1.0e-4)

    best_params = [p.detach().clone() for p in params]
    best_loss = tc.full((S,), float("inf"), device=device)
    active = tc.ones(S, dtype=tc.bool, device=device)  # still training
    n_steps = tc.full((S,), epochs, dtype=tc.long, device=device)  # steps taken
    stop_xt = xt_f.clone()  # each run's points at the step it stopped on
    loss_hist = tc.zeros(epochs, S, device=device)
    nu_hist = tc.zeros(epochs, S, device=device)

    for step in range(epochs):
        if resample_fn is not None:
            x_f, t_f = resample_fn()
            xt_f = stencil_inputs(x_f, t_f, L, hx, ht)

        loss = train_loss()
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_per_run(1.0)
        opt.step()

        loss_value = monitor()
        loss_hist[step], nu_hist[step] = loss_value, nu.detach().view(-1)
        improved = active & (loss_value < best_loss)
        best_loss = tc.where(improved, loss_value, best_loss)
        for bp, p in zip(best_params, params):
            bp.copy_(tc.where(improved.view(per_run(p)), p.detach(), bp))

        if target_loss is not None:
            reached = active & (loss_value <= target_loss)
            n_steps = tc.where(reached, step + 1, n_steps)
            stop_xt = tc.where(reached.view(-1, 1, 1), xt_f, stop_xt)
            active = active & ~reached
            if (step + 1) % check_every == 0 and not bool(active.any()):
                break
    target_reached = ~active
    stop_xt = tc.where(active.view(-1, 1, 1), xt_f, stop_xt)  # runs that never stopped: last points

    with tc.no_grad():
        for bp, p in zip(best_params, params):
            p.copy_(bp)  # best AdamW state of every run

    loss_hist, nu_hist = loss_hist.cpu().numpy(), nu_hist.cpu().numpy()
    n_steps, target_reached, best_loss = n_steps.tolist(), target_reached.tolist(), best_loss.tolist()
    runs = [
        dict(
            hist={"loss": loss_hist[:k, i].tolist(), "nu": nu_hist[:k, i].tolist()},
            best_loss=best_loss[i], target_reached=target_reached[i],
            stop_x=stop_xt[i, :n_f, 0:1], stop_t=stop_xt[i, :n_f, 1:2],
        )
        for i, k in enumerate(n_steps)
    ]
    return model, runs

def predict(model, x, t_val):
    """Evaluates the model on a spatial grid x at a fixed time t_val, on
    whichever device `model` currently lives on."""
    x = np.asarray(x).reshape(-1, 1)
    device = next(model.parameters()).device
    xt = tc.tensor(np.hstack([x, t_val * np.ones_like(x)]), dtype=tc.get_default_dtype(), device=device)
    model.eval()
    with tc.no_grad():
        return model(xt).detach().cpu().numpy().reshape(-1, 1)

def relative_l2_error(u_pred, u_ref):
    return float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_loss(histories, title="Training loss", save_path=None):
    plt.figure(figsize=(9, 4.5))
    for label, values in histories.items():
        plt.plot(np.asarray(values), linewidth=2.0, label=label)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.yscale("log")
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_parameter_history(nu_history, target, title="Learned nu", save_path=None):
    """nu_history: learned nu at every training step. target: the true nu,
    drawn as a horizontal reference line."""
    plt.figure(figsize=(9, 4.5))
    plt.plot(nu_history, linewidth=2.0, label="learned nu")
    plt.axhline(target, color="black", linestyle="--", linewidth=2.0, label=f"target nu = {target:g}")
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("nu", fontsize=14)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_solution(x, curves, ref=None, ref_label="FDM reference", title="",
                   save_path=None, collocation_x=None):
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.gca()
    if ref is not None:
        ax.plot(x.ravel(), ref.ravel(), linewidth=3.0, label=ref_label)
    for label, u in curves.items():
        ax.plot(x.ravel(), u.ravel(), "--", linewidth=2.0, label=label)
    if collocation_x is not None:
        collocation_x = np.asarray(collocation_x).ravel()
        rug_trans = ax.get_xaxis_transform()
        ax.scatter(
            collocation_x, np.full_like(collocation_x, 0.02), transform=rug_trans,
            marker="|", s=200, linewidth=1.2, color="black", clip_on=False,
            label=f"collocation points (N={collocation_x.size})",
        )
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u(x,t)", fontsize=14)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ----------------------------------------------------------------------
# Runs of the inverse problem (worker jobs)
# ----------------------------------------------------------------------
def build_model(n_qubits, Tfin, nu_init):
    return QuantumExactInversePINN(
        pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
        n_qubits=n_qubits, post_neurons=POST_NEURONS, post_depth=POST_DEPTH,
        L=L, Tfin=Tfin, nu_init=nu_init, angle_scale=ANGLE_SCALE,
        fixed_layer=FIXED_LAYER, fixed_layer_repeats=FIXED_LAYER_REPEATS,
        input_rotation_axis=INPUT_ROTATION_AXIS,
    )

def problem_setup(n_col, Tfin, device):
    """What a run needs that doesn't depend on its seed. Returns
    (ref, validation_points, data_points, target_loss), ref = (x_ref, t_ref,
    U_ref) the FDM reference, used only for comparison/plots and for the
    fixed data points at t=Tfin, never for the PDE-residual part of training."""
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_val, t_val = sample_collocation_points(
        n_col, L, Tfin, HT, distribution=STOPPING_DISTRIBUTION)

    # Inverse-problem data: the reference solution at the final time, on a
    # fixed subset of points -- unlike x_f, t_f, this set never changes
    # across epochs. Together with the hard-imposed initial condition, this
    # is what makes nu identifiable (the PDE residual alone can't pin it down).
    data_idx = np.linspace(0, len(x_ref) - 1, N_DATA).astype(int)
    to_tensor = lambda a: tc.tensor(a, dtype=tc.get_default_dtype(), device=device)
    x_data = to_tensor(x_ref[data_idx].reshape(-1, 1))
    t_data = tc.full_like(x_data, float(t_ref[-1]))
    u_data = to_tensor(U_ref[-1][data_idx].reshape(-1, 1))

    # Stop once the PINN's residual matches the reference's own discretization
    # error at these points -- pushing it lower would just chase FD noise.
    # Computed on CPU (numpy), before anything is moved to `device`.
    target_loss = reference_pde_residual(x_val, t_val, x_ref, t_ref, U_ref, L=L, nu=NU, hx=HX, ht=HT)

    return ((x_ref, t_ref, U_ref), (x_val.to(device), t_val.to(device)),
            (x_data, t_data, u_data), target_loss)

def save_run(model, hist, n_col, n_qubits, Tfin, seed, nu_init, target_loss, x_f, ref, device_str):
    """Checkpoint, loss/nu history and plots of one trained run (`model` on
    the CPU). x_f: the run's first collocation draw (drawn on the solution
    plot). Returns the learned nu."""
    x_ref, t_ref, U_ref = ref
    nu_learned = float(model.learnable_parameter.detach())

    out_dir = os.path.join(OUT_DIR, f"T_{Tfin}")
    tag = f"{n_col}_NQ{n_qubits}_{seed}"

    model.save(
        os.path.join(out_dir, f"model_{tag}.pt"),
        training_params=dict(
            L=L, TFIN=Tfin, NU=NU, HX=HX, HT=HT,
            N_COLLOCATION=n_col, N_DATA=N_DATA, SEED=seed,
            COLLOCATION_DISTRIBUTION=COLLOCATION_DISTRIBUTION,
            STOPPING_DISTRIBUTION=STOPPING_DISTRIBUTION,
            EPOCHS=EPOCHS, LR=LR, LBFGS_EPOCHS=LBFGS_EPOCHS, LBFGS_LR=LBFGS_LR,
            PRE_NEURONS=PRE_NEURONS, PRE_DEPTH=PRE_DEPTH,
            N_QUBITS=n_qubits, POST_NEURONS=POST_NEURONS, POST_DEPTH=POST_DEPTH,
            ANGLE_SCALE=ANGLE_SCALE, FIXED_LAYER=FIXED_LAYER,
            FIXED_LAYER_REPEATS=FIXED_LAYER_REPEATS, INPUT_ROTATION_AXIS=INPUT_ROTATION_AXIS,
            NU_INIT=nu_init, NU_LEARNED=nu_learned,
            TARGET_LOSS=target_loss, TARGET_REACHED=hist["target_reached"],
            STOPPED_AT_STEP=len(hist["loss"]), BEST_LOSS=hist["best_loss"],
            DEVICE=device_str,
        ),
    )

    # Full cost-function/nu history (every step, AdamW + L-BFGS), kept
    # separate from the checkpoint so the human-readable metadata above
    # stays small.
    np.savez(
        os.path.join(out_dir, f"losshist_{tag}.npz"),
        loss=np.array(hist["loss"]),
        nu=np.array(hist["nu"]),
        target_loss=target_loss,
        target_reached=hist["target_reached"],
    )

    plot_loss(
        {"Quantum Exact PINN": hist["loss"]}, title=f"Quantum Exact PINN (inverse) loss -- T={Tfin}",
        save_path=os.path.join(out_dir, f"loss_{tag}.png"),
    )
    plot_parameter_history(
        hist["nu"], NU, title=f"Learned viscosity nu -- N_col={n_col}, N_qubits={n_qubits}, T={Tfin}",
        save_path=os.path.join(out_dir, f"nu_history_{tag}.png"),
    )

    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_pred = predict(model, x_plot, Tfin)
    u_ref = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    plot_solution(
        x_plot, {"Quantum Exact PINN": u_pred}, ref=u_ref,
        title=f"Quantum Exact PINN vs FDM reference at T={Tfin}",
        save_path=os.path.join(out_dir, f"solution_{tag}.png"),
        collocation_x=x_f.detach().cpu().numpy(),
    )
    return nu_learned

def run_one(n_col, n_qubits, Tfin, run_id, device_str="cpu"):
    """A single run, trained on its own with `train`. The sweep below uses
    `run_group` (much faster); this is kept for single-run checks."""
    tc.set_num_threads(2)
    device = tc.device(device_str)

    SEED = run_id
    set_seed(SEED)
    # Drawn *after* set_seed so it is reproducible per (run_id) -- unlike the
    # serial script, where NU_INIT was drawn once at import time.
    nu_init = float(np.random.random())

    # Model weights are initialized on CPU (so init RNG is reproducible
    # regardless of `device`), then moved to the target device.
    model = build_model(n_qubits, Tfin, nu_init).to(device)

    x_f, t_f = sample_collocation_points(
        n_col, L, Tfin, HT, seed=SEED, distribution=COLLOCATION_DISTRIBUTION)
    resample_fn = lambda: tuple(
        tensor.to(device) for tensor in sample_collocation_points(
            n_col, L, Tfin, HT, distribution=COLLOCATION_DISTRIBUTION)
    )
    ref, validation_points, data_points, target_loss = problem_setup(n_col, Tfin, device)

    model, hist = train(
        model, x_f.to(device), t_f.to(device), epochs=EPOCHS, lr=LR, resample_fn=resample_fn,
        lbfgs_epochs=LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
        target_loss=target_loss, validation_points=validation_points,
        data_points=data_points, L=L, hx=HX, ht=HT,
    )

    # Move back to CPU: checkpoints stay loadable without a GPU, and every
    # plot/prediction below is small enough that CPU is plenty fast.
    nu_learned = save_run(model.to("cpu"), hist, n_col, n_qubits, Tfin, SEED, nu_init,
                          target_loss, x_f, ref, device_str)
    return n_col, n_qubits, Tfin, run_id, nu_learned

def run_group(n_col, n_qubits, Tfin, run_ids, device_str="cpu"):
    """Every seed in `run_ids` of one (n_col, n_qubits, Tfin) configuration
    (one worker job): the AdamW phase trains them all together as one
    BatchedQuantumExactPINN, then each run goes through the same L-BFGS
    phase as `train`, on its own. Per seed, the initial network, nu_init and
    first collocation draw are the same as run_one's. The points re-drawn
    every AdamW epoch come from one generator for the whole group (seeded
    with its first seed), so they are reproducible for the same group of
    seeds, not seed by seed. Returns [(run_id, nu_learned), ...]."""
    tc.set_num_threads(2)
    device = tc.device(device_str)

    nu_inits, models, x_first, t_first = [], [], [], []
    for seed in run_ids:
        set_seed(seed)
        nu_inits.append(float(np.random.random()))
        models.append(build_model(n_qubits, Tfin, nu_inits[-1]))  # initialized on CPU
        x, t = sample_collocation_points(
            n_col, L, Tfin, HT, seed=seed, distribution=COLLOCATION_DISTRIBUTION)
        x_first.append(x)
        t_first.append(t)
    model = BatchedQuantumExactPINN(models).to(device)
    x_f, t_f = tc.stack(x_first).to(device), tc.stack(t_first).to(device)  # (S, n_col, 1)

    if COLLOCATION_DISTRIBUTION == "random":
        generator = tc.Generator(device=device).manual_seed(run_ids[0])
        shape = (len(run_ids), n_col, 1)
        resample_fn = lambda: (
            tc.rand(shape, generator=generator, device=device) * L,
            HT + tc.rand(shape, generator=generator, device=device) * (Tfin - 2.0 * HT),
        )
    else:  # a fixed grid: re-drawing gives the same points every epoch
        resample_fn = None

    ref, validation_points, data_points, target_loss = problem_setup(n_col, Tfin, device)
    model, runs = train_batched_adam(
        model, x_f, t_f, epochs=EPOCHS, lr=LR, resample_fn=resample_fn,
        target_loss=target_loss, validation_points=validation_points,
        data_points=data_points, L=L, hx=HX, ht=HT,
    )

    x_val, t_val = validation_points
    xt_val = stencil_inputs(x_val, t_val, L, HX, HT)
    x_data, t_data, u_data = data_points
    xt_data = tc.cat([x_data, t_data], dim=1)

    results = []
    for i, (seed, run) in enumerate(zip(run_ids, runs)):
        run_model = build_model(n_qubits, Tfin, nu_inits[i])
        run_model.load_state_dict(model.run_state_dict(i))
        run_model = run_model.to(device)
        hist, best_loss, target_reached = run["hist"], run["best_loss"], run["target_reached"]
        if LBFGS_EPOCHS > 0:
            best_state = {k: v.detach().clone() for k, v in run_model.state_dict().items()}
            best_loss, target_reached = lbfgs_refine(
                run_model, stencil_inputs(run["stop_x"], run["stop_t"], L, HX, HT), n_col,
                hist, best_state, best_loss, target_reached, LBFGS_EPOCHS, LBFGS_LR,
                target_loss=target_loss, xt_val=xt_val, n_val=x_val.shape[0],
                xt_data=xt_data, u_data=u_data, hx=HX, ht=HT,
            )
        hist["target_reached"], hist["best_loss"] = target_reached, best_loss

        nu_learned = save_run(run_model.to("cpu"), hist, n_col, n_qubits, Tfin, seed, nu_inits[i],
                              target_loss, x_first[i], ref, device_str)
        results.append((seed, nu_learned))
    return results


if __name__ == "__main__":
    ################# PARALELIZAÇAO ######################################
    parser = argparse.ArgumentParser(
        description="Parallel sweep for the exact-CNOT-cosine quantum inverse PINN "
                     "(see run_quantumExact.ipynb). Cost does not grow exponentially "
                     "with N_QUBITS, so unlike Inverser_quantum_pinn_paralelo.py there "
                     "is a single worker pool for every job.")
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "auto"], default="cpu",
        help="'cpu' (default): run every job on CPU. 'cuda': run on GPU(s) -- see "
             "--gpu-ids/--workers to control how many worker processes share them. "
             "'auto': cuda if available, else cpu.")
    parser.add_argument(
        "--gpu-ids", type=str, default=None,
        help="Comma-separated CUDA device indices to round-robin jobs across "
             "(e.g. '0,1'). Only used with --device cuda/auto. Default: every "
             "GPU visible to torch.")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override the worker count (default: %d on CPU; %d per GPU when "
             "using --device cuda/auto)." % (N_WORKERS_CPU, N_WORKERS_PER_GPU))
    args = parser.parse_args()

    use_cuda = args.device == "cuda" or (args.device == "auto" and tc.cuda.is_available())
    if args.device == "cuda" and not tc.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False.")

    if use_cuda:
        gpu_ids = (
            [int(i) for i in args.gpu_ids.split(",")] if args.gpu_ids
            else list(range(tc.cuda.device_count()))
        )
        if not gpu_ids:
            raise RuntimeError("--device cuda/auto was requested but no CUDA devices were found.")
        n_workers = args.workers or (N_WORKERS_PER_GPU * len(gpu_ids))
        device_for = lambda job_index: f"cuda:{gpu_ids[job_index % len(gpu_ids)]}"
        device_summary = f"cuda ({len(gpu_ids)} GPU(s): {gpu_ids})"
    else:
        n_workers = args.workers or N_WORKERS_CPU
        n_workers = max(1, min(n_workers, os.cpu_count() or n_workers))
        device_for = lambda job_index: "cpu"
        device_summary = "cpu"

    SEEDS_NEW = range(10, 50)  # 10 seeds

    # Bloco A: varredura de N_COLLOCATION com N_QUBITS em [3, 7, 15]
    block_A = [
        (n_col, n_qubits, Tfin, run_id)
        for n_col in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for n_qubits in [3, 7]
        for Tfin in [0.1, 1]
        for run_id in SEEDS_NEW
    ]

    # Bloco B: varredura de N_QUBITS (1 a 10) com N_COLLOCATION fixo em 10 e 100
    block_B = [
        (n_col, n_qubits, Tfin, run_id)
        for n_col in [10, 100]
        for n_qubits in range(1, 11)
        for Tfin in [0.1, 1]
        for run_id in SEEDS_NEW
    ]

    jobs_by_key = {}
    for block in (block_A, block_B):
        for job in block:
            jobs_by_key[job] = job
    all_jobs = list(jobs_by_key.values())

    # Retomada apos interrupcao: pula jobs cujo checkpoint (.pt) ja existe em
    # disco, para nao re-treinar o que ja foi concluido antes do PC desligar.
    def _output_path(job):
        n_col, n_qubits, Tfin, run_id = job
        return os.path.join(OUT_DIR, f"T_{Tfin}", f"model_{n_col}_NQ{n_qubits}_{run_id}.pt")

    jobs_done = [job for job in all_jobs if os.path.exists(_output_path(job))]
    jobs = [job for job in all_jobs if not os.path.exists(_output_path(job))]

    print(f"Device: {device_summary}, workers={n_workers}")
    print(f"Ja concluidos (pulados): {len(jobs_done)} runs")
    print(f"Faltando (serao rodados agora): {len(jobs)} runs")

    for _, _, Tfin, _ in jobs:
        os.makedirs(os.path.join(OUT_DIR, f"T_{Tfin}"), exist_ok=True)

    # Em lote: as seeds que faltam de cada configuracao (n_col, n_qubits,
    # Tfin) sao treinadas juntas, num unico job (ver run_group).
    groups = {}
    for n_col, n_qubits, Tfin, run_id in jobs:
        groups.setdefault((n_col, n_qubits, Tfin), []).append(run_id)
    print(f"{len(groups)} grupos (n_col, n_qubits, Tfin)")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_group, n_col, n_qubits, Tfin, run_ids, device_for(i)): ((n_col, n_qubits, Tfin), run_ids)
            for i, ((n_col, n_qubits, Tfin), run_ids) in enumerate(groups.items())
        }
        progress = tqdm(total=len(jobs), desc="runs", unit="run")
        n_failed = 0
        for future in as_completed(futures):
            (n_col, n_qubits, Tfin), run_ids = futures[future]
            try:
                future.result()
            except Exception as exc:
                n_failed += len(run_ids)
                progress.write(
                    f"FAILED: N_COLLOCATION={n_col} N_QUBITS={n_qubits} "
                    f"Tfin={Tfin} run_ids={list(run_ids)} -> {exc!r}"
                )
            progress.update(len(run_ids))
            progress.set_postfix(failed=n_failed)
        progress.close()
        if n_failed:
            print(f"{n_failed} run(s) failed -- see FAILED lines above.")
