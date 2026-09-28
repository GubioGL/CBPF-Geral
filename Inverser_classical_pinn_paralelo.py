"""Parallelized version of Inverser_classical_pinn.py (inverse problem: learn
nu jointly with u(x,t)), following the same run_one / ProcessPoolExecutor
sweep structure as run_classical.py.
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # headless backend -- safe across worker processes
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
from tqdm import tqdm

# ---- physical parameters ---------------------------------------------------
L, NU = 1.0, 0.1
# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3
# ---- collocation sampling ---------------------------------------------------
N_DATA = 100  # fixed data points at t=Tfin, needed to identify nu
# ---- training ---------------------------------------------------------------
EPOCHS, LR = 20000, 1.0e-2               # phase 1: AdamW, network + nu trained jointly
REFINE_EPOCHS, REFINE_LR = 5000, 1.0e-3  # phase 2: Adam, network frozen, nu-only refinement
# ---- architecture (activation: tanh, see `mlp` in ClassicalPINN) ------------
PRE_NEURONS, PRE_DEPTH = 10, 1
INTRA_NEURONS, INTRA_DEPTH = 10, 1        # width/depth around the bottleneck, fixed
POST_NEURONS, POST_DEPTH = 10, 1

SAVE_EXTRAS = True  # also save the model checkpoint + full loss/nu history
OUT_DIR = "data/classical_inverse"

# ---- device -----------------------------------------------------------------
DEVICE = tc.device("cpu")

# ---- parallelization --------------------------------------------------------
# On GPU every worker process opens its own CUDA context (~300-500 MB of VRAM
# each), so only a few workers share the card; on CPU, use all cores but one.
# N_GPU_WORKERS = 12
N_WORKERS =12

def set_seed(seed=42):
    np.random.seed(seed)
    tc.manual_seed(seed)

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
    bound, overflows, and turns into inf/nan (the RuntimeWarnings seen when
    stepping the loop below)."""
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

class ClassicalPINN(nn.Module):
    def __init__(self, pre_neurons, pre_depth, intra_neuron, intra_neurons,
                 intra_depth, post_neurons, post_depth, L=1.0, Tfin=0.1, nu_init=.1):
        super().__init__()
        self.L, self.Tfin = float(L), float(Tfin)

        def mlp(n_in, hidden, depth, n_out):
            layers = [nn.Linear(n_in, hidden), nn.Tanh()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            layers += [nn.Linear(hidden, n_out)]
            return nn.Sequential(*layers)

        self.pre = mlp(3, pre_neurons, pre_depth, intra_neuron)
        self.net = mlp(intra_neuron, intra_neurons, intra_depth, intra_neuron)
        self.post = mlp(intra_neuron, post_neurons, post_depth, 1)
        self.learnable_parameter = nn.Parameter(tc.tensor(float(nu_init)))

    def forward(self, xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        features = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), t / self.Tfin],
            dim=1,
        )
        out = self.post(self.net(self.pre(features)))
        return initial_condition(x, L=self.L) + (t / self.Tfin) * out  # hard-constrains u(x,0)=u0(x)

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

def train(model, x_f, t_f, epochs, lr, resample_fn=None,
          refine_epochs=5000, refine_lr=1.0e-4, target_loss=None,
          validation_points=None, data_points=None, L=1.0, hx=1.0e-3, ht=1.0e-3):
    """Two-phase training, both phases with Adam(W) -- no L-BFGS.

    Phase 1 (AdamW, lr): all trainable parameters (network weights + nu)
    are updated jointly for up to `epochs` steps, stopping early once the
    monitored loss reaches `target_loss`.

    Phase 2 (Adam, refine_lr): the network is frozen (only
    model.learnable_parameter keeps requires_grad=True) and training
    continues on nu alone, for up to `refine_epochs` steps, with a smaller
    learning rate to refine its last digits now that u(x,t) is already a
    good fit. Uses the same (possibly stale) x_f, t_f phase 1 ended on --
    resample_fn is not called again in this phase.

    resample_fn: optional callable returning a fresh (x_f, t_f) pair,
    called every phase-1 epoch (used to train on freshly drawn random points).
    validation_points: optional fixed (x, t) used to monitor/stop training
    instead of the (possibly resampled) training points.
    data_points: optional fixed (x_data, t_data, u_data) supervised targets,
    added to the training loss as MSE(model(x_data, t_data), u_data). Kept
    fixed across epochs, unlike x_f, t_f which may change every epoch.
    target_loss: stop as soon as the monitored (PDE-residual-only) loss
    reaches this value.
    Returns (model, hist), hist["nu"] holding the learned nu at every step.
    `model` holds the best (lowest monitored loss) phase-1 network, with nu
    set to its phase-2 refined value (= hist["nu"][-1]) when refine_epochs > 0."""

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
        """PDE-residual MSE + data MSE, from one forward pass over the stencil
        and data points together."""
        u = model(xt_f if xt_data is None else tc.cat([xt_f, xt_data], dim=0))
        a, b = residual_terms(u, n_f, hx, ht)
        loss = tc.mean((a - model.learnable_parameter * b) ** 2)
        if xt_data is not None:
            loss = loss + tc.mean((u[5 * n_f:] - u_data) ** 2)
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

    # ---- Phase 1: AdamW, network + nu trained jointly ----------------------
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

    # ---- Phase 2: Adam, network frozen, nu-only refinement ------------------
    if refine_epochs > 0:
        model.load_state_dict(best_state)  # refine from the best basin phase 1 found

        # With the network frozen and the points fixed, the residual is affine
        # in nu -- r = A - nu*B with A, B constant -- so A and B are evaluated
        # once instead of re-running the network every step. The data loss
        # doesn't depend on nu (zero gradient), so it drops out here.
        nu = model.learnable_parameter
        with tc.no_grad():
            a_f, b_f = residual_terms(model(xt_f), n_f, hx, ht)
            a_mon, b_mon = residual_terms(model(xt_val), n_val, hx, ht) if xt_val is not None else (a_f, b_f)
        nu_opt = tc.optim.Adam([nu], lr=refine_lr)

        for _ in range(refine_epochs):
            loss = tc.mean((a_f - nu * b_f) ** 2)
            nu_opt.zero_grad()
            loss.backward()
            nu_opt.step()

            with tc.no_grad():
                loss_value = float(tc.mean((a_mon - nu * b_mon) ** 2))
            hist["loss"].append(loss_value)
            hist["nu"].append(float(nu.detach()))

        # The network in best_state is exactly the frozen one phase 2 ran on;
        # only nu changed, so keep the refined value instead of phase 1's.
        best_state["learnable_parameter"] = nu.detach().clone()

    model.load_state_dict(best_state)  # best phase-1 network (+ refined nu, if phase 2 ran)
    hist["target_reached"] = target_reached
    hist["best_loss"] = best_loss
    return model, hist

# ----------------------------------------------------------------------
# Batched training: several independent runs trained together
# ----------------------------------------------------------------------
class BatchedPINN(nn.Module):
    """S independent ClassicalPINNs (same architecture), evaluated together.
    Each Linear layer's weights are stacked along a leading run axis, so one
    batched matmul per layer serves all S networks. Every run keeps its own
    weights and its own nu and still takes only (x, t) as input -- nothing
    is shared between runs. With networks this small the cost is per-call
    overhead, so S runs together cost little more than one."""

    def __init__(self, models):
        super().__init__()
        ref = models[0]
        self.L, self.Tfin = ref.L, ref.Tfin
        # Same layer order/activations as ClassicalPINN.forward: post(net(pre(.))),
        # tanh after every Linear except the last of each block.
        self.layers = []  # (name in ClassicalPINN, tanh after it?)
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        for block in ("pre", "net", "post"):
            seq = getattr(ref, block)
            for i, layer in enumerate(seq):
                if not isinstance(layer, nn.Linear):
                    continue
                name = f"{block}.{i}"
                self.layers.append((name, i + 1 < len(seq) and isinstance(seq[i + 1], nn.Tanh)))
                linears = [m.get_submodule(name) for m in models]
                self.weights.append(nn.Parameter(tc.stack([lin.weight.detach().T for lin in linears])))  # (S, in, out)
                self.biases.append(nn.Parameter(tc.stack([lin.bias.detach()[None, :] for lin in linears])))  # (S, 1, out)
        self.learnable_parameter = nn.Parameter(
            tc.stack([m.learnable_parameter.detach() for m in models]).view(-1, 1, 1))  # (S, 1, 1)

    def forward(self, xt):
        """xt: (S, M, 2) -- M points (x, t) for each of the S runs."""
        x, t = xt[..., 0:1], xt[..., 1:2]
        h = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), tc.cos(2.0 * math.pi * x / self.L), t / self.Tfin],
            dim=-1,
        )
        for (_, tanh_after), w, b in zip(self.layers, self.weights, self.biases):
            h = tc.baddbmm(b, h, w)
            if tanh_after:
                h = tc.tanh(h)
        return initial_condition(x, L=self.L) + (t / self.Tfin) * h  # hard-constrains u(x,0)=u0(x)

    def run_state_dict(self, i):
        """Run i's weights in ClassicalPINN's state_dict format, on the CPU."""
        sd = {"learnable_parameter": self.learnable_parameter[i, 0, 0].detach().cpu().clone()}
        for (name, _), w, b in zip(self.layers, self.weights, self.biases):
            sd[f"{name}.weight"] = w[i].detach().T.contiguous().cpu()
            sd[f"{name}.bias"] = b[i, 0].detach().cpu().clone()
        return sd

def train_batched(model, x_f, t_f, epochs, lr, resample_fn=None,
                  refine_epochs=5000, refine_lr=1.0e-4, target_loss=None,
                  validation_points=None, data_points=None, L=1.0, hx=1.0e-3, ht=1.0e-3,
                  check_every=50):
    """`train` for a BatchedPINN: the same two phases, applied to every run
    independently. Summing the per-run losses gives each run exactly its own
    gradient, and Adam(W) acts elementwise, so each run follows its own
    optimization. What `train` does globally is done per run here: gradient
    clipping, the best snapshot, early stopping (a run that reached
    `target_loss` stops recording and keeps its best state and the points it
    stopped on) and phase 2.

    x_f, t_f: (S, N, 1), one point set per run; resample_fn returns the same
    shape. validation_points and data_points are shared by all runs, (N, 1).
    check_every: how often (steps) to check whether every run has already
    stopped, to end phase 1 early -- only saves time, results don't depend on it.
    Returns (model, hists), hists[i] the `train`-style hist of run i."""
    nu = model.learnable_parameter
    S, device = nu.shape[0], nu.device
    params = list(model.parameters())
    per_run = lambda p: (-1,) + (1,) * (p.dim() - 1)  # view to broadcast a (S,) tensor over p
    per_run_mse = lambda r: tc.mean(r ** 2, dim=(1, 2))  # (S, M, 1) -> (S,)

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
        loss = per_run_mse(a - nu * b)
        if xt_data is not None:
            loss = loss + per_run_mse(u[:, 5 * n_f:] - u_data)
        return loss.sum()

    def monitor():
        with tc.no_grad():
            xt, n = (xt_val, n_val) if xt_val is not None else (xt_f, n_f)
            a, b = residual_terms(model(xt), n, hx, ht)
            return per_run_mse(a - nu * b)

    def clip_grad_norm_per_run(max_norm=1.0):
        """tc.nn.utils.clip_grad_norm_(max_norm), on each run's own gradient."""
        norms = tc.sqrt(sum(p.grad.pow(2).flatten(1).sum(1) for p in params))
        coef = tc.clamp(max_norm / (norms + 1.0e-6), max=1.0)
        for p in params:
            p.grad.mul_(coef.view(per_run(p)))

    opt = tc.optim.AdamW(params, lr=lr, weight_decay=1.0e-4)

    best_params = [p.detach().clone() for p in params]
    best_loss = tc.full((S,), float("inf"), device=device)
    active = tc.ones(S, dtype=tc.bool, device=device)  # still in phase 1
    n_steps = tc.full((S,), epochs, dtype=tc.long, device=device)  # phase-1 steps taken
    stop_xt = xt_f.clone()  # each run's points at the step it stopped on
    loss_hist = tc.zeros(epochs, S, device=device)
    nu_hist = tc.zeros(epochs, S, device=device)

    # ---- Phase 1: AdamW, network + nu trained jointly ----------------------
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
            p.copy_(bp)  # best phase-1 state of every run

    # ---- Phase 2: Adam, network frozen, nu-only refinement ------------------
    # Same affine-in-nu shortcut as `train`: r = A - nu*B, A and B fixed.
    refine_loss = tc.zeros(refine_epochs, S, device=device)
    refine_nu = tc.zeros(refine_epochs, S, device=device)
    if refine_epochs > 0:
        with tc.no_grad():
            a_f, b_f = residual_terms(model(stop_xt), n_f, hx, ht)
            a_mon, b_mon = residual_terms(model(xt_val), n_val, hx, ht) if xt_val is not None else (a_f, b_f)
        nu_opt = tc.optim.Adam([nu], lr=refine_lr)

        for step in range(refine_epochs):
            loss = per_run_mse(a_f - nu * b_f).sum()
            nu_opt.zero_grad()
            loss.backward()
            nu_opt.step()

            with tc.no_grad():
                refine_loss[step], refine_nu[step] = per_run_mse(a_mon - nu * b_mon), nu.view(-1)
    # The model now holds each run's best phase-1 network + its refined nu.

    loss_hist, nu_hist = loss_hist.cpu().numpy(), nu_hist.cpu().numpy()
    refine_loss, refine_nu = refine_loss.cpu().numpy(), refine_nu.cpu().numpy()
    n_steps, target_reached, best_loss = n_steps.tolist(), target_reached.tolist(), best_loss.tolist()
    hists = [
        {
            "loss": np.concatenate([loss_hist[:k, i], refine_loss[:, i]]),
            "nu": np.concatenate([nu_hist[:k, i], refine_nu[:, i]]),
            "target_reached": target_reached[i],
            "best_loss": best_loss[i],
        }
        for i, k in enumerate(n_steps)
    ]
    return model, hists

def predict(model, x, t_val):
    """Evaluates the model on a spatial grid x at a fixed time t_val."""
    x = np.asarray(x).reshape(-1, 1)
    device = next(model.parameters()).device
    xt = tc.tensor(np.hstack([x, t_val * np.ones_like(x)]), dtype=tc.get_default_dtype(), device=device)
    model.eval()
    with tc.no_grad():
        return model(xt).cpu().numpy().reshape(-1, 1)

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
def build_model(intra_neuro, Tfin, nu_init):
    return ClassicalPINN(
        pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
        intra_neuron=intra_neuro, intra_neurons=INTRA_NEURONS, intra_depth=INTRA_DEPTH,
        post_neurons=POST_NEURONS, post_depth=POST_DEPTH,
        L=L, Tfin=Tfin, nu_init=nu_init,
    )

def problem_setup(n_col, Tfin):
    """What a run needs that doesn't depend on its seed. Returns
    (ref, validation_points, data_points, target_loss), ref = (x_ref, t_ref,
    U_ref) the FDM reference, used only for comparison/plots and for the
    fixed data points at t=Tfin, never for the PDE-residual part of training."""
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_val, t_val = sample_collocation_points(n_col, L, Tfin, HT, distribution="uniform_exact")

    # Inverse-problem data: the reference solution at the final time, on a
    # fixed subset of points -- unlike x_f, t_f, this set never changes
    # across epochs. Together with the hard-imposed initial condition, this
    # is what makes nu identifiable (the PDE residual alone can't pin it down).
    data_idx = np.linspace(0, len(x_ref) - 1, N_DATA).astype(int)
    to_tensor = lambda a: tc.tensor(a, dtype=tc.get_default_dtype(), device=DEVICE)
    x_data = to_tensor(x_ref[data_idx].reshape(-1, 1))
    t_data = tc.full_like(x_data, float(t_ref[-1]))
    u_data = to_tensor(U_ref[-1][data_idx].reshape(-1, 1))

    # Stop once the PINN's residual matches the reference's own discretization
    # error at these points -- pushing it lower would just chase FD noise.
    target_loss = reference_pde_residual(x_val, t_val, x_ref, t_ref, U_ref, L=L, nu=NU, hx=HX, ht=HT)

    return ((x_ref, t_ref, U_ref), (x_val.to(DEVICE), t_val.to(DEVICE)),
            (x_data, t_data, u_data), target_loss)

def save_run(model, hist, n_col, intra_neuro, Tfin, seed, nu_init, target_loss, x_f, ref):
    """Checkpoint, loss/nu history and plots of one trained run. x_f: the
    run's first collocation draw (drawn on the solution plot). Returns the
    learned nu."""
    x_ref, t_ref, U_ref = ref
    nu_learned = float(model.learnable_parameter.detach())

    out_dir = os.path.join(OUT_DIR, f"T_{Tfin}")
    tag = f"{n_col}_INTRA{intra_neuro}_{seed}"

    if SAVE_EXTRAS:
        tc.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},  # loadable without a GPU
                "training_params": dict(
                    L=L, TFIN=Tfin, NU=NU, HX=HX, HT=HT,
                    N_COLLOCATION=n_col, N_DATA=N_DATA, SEED=seed,
                    EPOCHS=EPOCHS, LR=LR, REFINE_EPOCHS=REFINE_EPOCHS, REFINE_LR=REFINE_LR,
                    PRE_NEURONS=PRE_NEURONS, PRE_DEPTH=PRE_DEPTH,
                    INTRA_NEURON=intra_neuro, INTRA_NEURONS=INTRA_NEURONS, INTRA_DEPTH=INTRA_DEPTH,
                    POST_NEURONS=POST_NEURONS, POST_DEPTH=POST_DEPTH,
                    NU_INIT=nu_init, NU_LEARNED=nu_learned,
                    TARGET_LOSS=target_loss, TARGET_REACHED=hist["target_reached"],
                    STOPPED_AT_STEP=len(hist["loss"]), BEST_LOSS=hist["best_loss"],
                ),
            },
            os.path.join(out_dir, f"model_{tag}.pt"),
        )

        # Full cost-function/nu history (every step, phase 1 + phase 2), kept
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
        {"Classical PINN": hist["loss"]}, title=f"Classical PINN (inverse) loss -- T={Tfin}",
        save_path=os.path.join(out_dir, f"loss_{tag}.png"),
    )
    plot_parameter_history(
        hist["nu"], NU, title=f"Learned viscosity nu -- N_col={n_col}, intra={intra_neuro}, T={Tfin}",
        save_path=os.path.join(out_dir, f"nu_history_{tag}.png"),
    )

    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_pred = predict(model, x_plot, Tfin)
    u_ref = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    plot_solution(
        x_plot, {"Classical PINN": u_pred}, ref=u_ref,
        title=f"Classical PINN vs FDM reference at T={Tfin}",
        save_path=os.path.join(out_dir, f"solution_{tag}.png"),
        collocation_x=x_f.cpu().numpy(),
    )
    return nu_learned

def run_one(n_col, intra_neuro, Tfin, run_id):
    """A single run, trained on its own with `train`. The sweep below uses
    `run_group` (much faster); this is kept for single-run checks."""
    tc.set_num_threads(1)

    SEED = run_id
    set_seed(SEED)
    # Drawn *after* set_seed so it is reproducible per (run_id) -- unlike the
    # serial script, where NU_INIT was drawn once at import time.
    nu_init = float(np.random.random())
    model = build_model(intra_neuro, Tfin, nu_init).to(DEVICE)

    # Points are drawn on the CPU (same RNG stream as the CPU-only version,
    # so the draws per seed don't change) and then moved to DEVICE.
    to_device = lambda pair: tuple(a.to(DEVICE) for a in pair)
    x_f, t_f = to_device(sample_collocation_points(n_col, L, Tfin, HT, seed=SEED, distribution="random"))
    resample_fn = lambda: to_device(sample_collocation_points(n_col, L, Tfin, HT, distribution="random"))

    ref, validation_points, data_points, target_loss = problem_setup(n_col, Tfin)
    model, hist = train(
        model, x_f, t_f, epochs=EPOCHS, lr=LR, resample_fn=resample_fn,
        refine_epochs=REFINE_EPOCHS, refine_lr=REFINE_LR,
        target_loss=target_loss, validation_points=validation_points,
        data_points=data_points, L=L, hx=HX, ht=HT,
    )
    nu_learned = save_run(model, hist, n_col, intra_neuro, Tfin, SEED, nu_init, target_loss, x_f, ref)
    return n_col, intra_neuro, Tfin, run_id, nu_learned

def run_group(n_col, intra_neuro, Tfin, run_ids):
    """Every seed in `run_ids` of one (n_col, intra_neuro, Tfin)
    configuration, trained together as one BatchedPINN (one worker job).
    Per seed, the initial network, nu_init and first collocation draw are the
    same as run_one's. The points re-drawn every epoch come from one
    generator for the whole group (seeded with its first seed), so they are
    reproducible for the same group of seeds, not seed by seed.
    Returns [(run_id, nu_learned), ...]."""
    tc.set_num_threads(1)

    nu_inits, models, x_first, t_first = [], [], [], []
    for seed in run_ids:
        set_seed(seed)
        nu_inits.append(float(np.random.random()))
        models.append(build_model(intra_neuro, Tfin, nu_inits[-1]))
        x, t = sample_collocation_points(n_col, L, Tfin, HT, seed=seed, distribution="random")
        x_first.append(x)
        t_first.append(t)
    model = BatchedPINN(models).to(DEVICE)
    x_f, t_f = tc.stack(x_first).to(DEVICE), tc.stack(t_first).to(DEVICE)  # (S, n_col, 1)

    generator = tc.Generator(device=DEVICE).manual_seed(run_ids[0])
    shape = (len(run_ids), n_col, 1)
    resample_fn = lambda: (
        tc.rand(shape, generator=generator, device=DEVICE) * L,
        HT + tc.rand(shape, generator=generator, device=DEVICE) * (Tfin - 2.0 * HT),
    )

    ref, validation_points, data_points, target_loss = problem_setup(n_col, Tfin)
    model, hists = train_batched(
        model, x_f, t_f, epochs=EPOCHS, lr=LR, resample_fn=resample_fn,
        refine_epochs=REFINE_EPOCHS, refine_lr=REFINE_LR,
        target_loss=target_loss, validation_points=validation_points,
        data_points=data_points, L=L, hx=HX, ht=HT,
    )

    results = []
    for i, seed in enumerate(run_ids):
        run_model = build_model(intra_neuro, Tfin, nu_inits[i])
        run_model.load_state_dict(model.run_state_dict(i))
        nu_learned = save_run(run_model, hists[i], n_col, intra_neuro, Tfin, seed, nu_inits[i],
                              target_loss, x_first[i], ref)
        results.append((seed, nu_learned))
    return results


if __name__ == "__main__":
    ################# PARALELIZAÇAO ######################################
    print(f"Device: {DEVICE} -- using {N_WORKERS} workers ({os.cpu_count()} CPUs available).")
    SEEDS_NEW = range(0, 50)  # 50 seeds

    # # Bloco A: varredura de N_COLLOCATION com intra_neuro em [3, 7, 15]
    # block_A = [
    #     (n_col, intra_neuro, Tfin, run_id)
    #     for n_col in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #     for intra_neuro in [3, 7, 15]
    #     for Tfin in [0.1, 1]
    #     for run_id in SEEDS_NEW
    # ]

    # Bloco B: varredura de intra_neuro com N_COLLOCATION fixo em 10 e 100
    block_B = [
        (n_col, intra_neuro, Tfin, run_id)
        for n_col in [10, 100]
        for intra_neuro in [10,11,12,13,14]
        for Tfin in [0.1, 1]
        for run_id in SEEDS_NEW
    ]

    jobs_by_key = {}
    for block in (block_B,):
        for job in block:
            jobs_by_key[job] = job
    jobs = list(jobs_by_key.values())

    # Retomada apos interrupcao: pula jobs cujo checkpoint (.pt) ja existe em
    # disco, para nao re-treinar o que ja foi concluido antes do PC desligar.
    def _output_path(job):
        n_col, intra_neuro, Tfin, run_id = job
        return os.path.join(OUT_DIR, f"T_{Tfin}", f"model_{n_col}_INTRA{intra_neuro}_{run_id}.pt")

    jobs_done = [job for job in jobs if os.path.exists(_output_path(job))]
    jobs = [job for job in jobs if not os.path.exists(_output_path(job))]
    print(f"Ja concluidos (pulados): {len(jobs_done)} runs")
    print(f"Faltando (serao rodados agora): {len(jobs)} runs")

    for _, _, Tfin, _ in jobs:
        os.makedirs(os.path.join(OUT_DIR, f"T_{Tfin}"), exist_ok=True)

    # Em lote: as seeds que faltam de cada configuracao (n_col, intra_neuro,
    # Tfin) sao treinadas juntas, num unico job (ver run_group).
    groups = {}
    for n_col, intra_neuro, Tfin, run_id in jobs:
        groups.setdefault((n_col, intra_neuro, Tfin), []).append(run_id)
    print(f"{len(groups)} grupos (n_col, intra_neuro, Tfin) em {N_WORKERS} workers")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_group, *key, run_ids): (key, run_ids) for key, run_ids in groups.items()}
        progress = tqdm(total=len(jobs), desc="runs", unit="run")
        n_failed = 0
        for future in as_completed(futures):
            (n_col, intra_neuro, Tfin), run_ids = futures[future]
            try:
                future.result()
            except Exception as exc:
                n_failed += len(run_ids)
                progress.write(
                    f"FAILED: N_COLLOCATION={n_col} intra_neuro={intra_neuro} "
                    f"Tfin={Tfin} run_ids={list(run_ids)} -> {exc!r}"
                )
            progress.update(len(run_ids))
            progress.set_postfix(failed=n_failed)
        progress.close()
        if n_failed:
            print(f"{n_failed} run(s) failed -- see FAILED lines above.")
