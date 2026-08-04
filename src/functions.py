import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
from tqdm import tqdm

def set_seed(seed=42):
    np.random.seed(seed)
    tc.manual_seed(seed)

def initial_condition(x, L=1.0):
    """u0(x) = sin(2*pi*x/L). Accepts a NumPy array or a Torch tensor."""
    if isinstance(x, tc.Tensor):
        return tc.sin(2.0 * math.pi * x / L)
    return np.sin(2.0 * np.pi * x / L)

# ----------------------------------------------------------------------
# Reference solution (pure numerical baseline, no neural network involved)
# ----------------------------------------------------------------------
def solve_burgers_fd(L=1.0, Tfin=0.1, nu=0.01, Nx=512, dt=1.0e-4, scheme="upwind"):
    """
    Explicit finite-difference solution of u_t + u u_x = nu u_xx on a
    periodic domain [0, L). Used as the ground-truth reference for every
    PINN comparison, and as the standalone numerical baseline.
    """
    x = np.linspace(0.0, L, Nx, endpoint=False)
    dx = L / Nx
    u = initial_condition(x, L=L).copy()

    times = np.arange(0.0, Tfin + 0.5 * dt, dt)
    U_hist = np.zeros((len(times), Nx), dtype=np.float64)
    U_hist[0] = u.copy()

    for n in range(1, len(times)):
        up, um = np.roll(u, -1), np.roll(u, 1)

        if scheme == "central":
            ux = (up - um) / (2.0 * dx)
        elif scheme == "upwind":
            ux_back = (u - um) / dx
            ux_fwd = (up - u) / dx
            ux = np.where(u >= 0.0, ux_back, ux_fwd)
        else:
            raise ValueError("scheme must be 'central' or 'upwind'.")

        uxx = (up - 2.0 * u + um) / dx**2
        u = u + dt * (-u * ux + nu * uxx)
        U_hist[n] = u.copy()

    return x, times, U_hist


def reference_at_time(t_eval, t_ref, x_ref, U_ref, x_query, L=1.0):
    """Periodic interpolation of the FDM reference at the closest stored time."""
    idx = int(np.argmin(np.abs(t_ref - t_eval)))
    xp = np.concatenate([x_ref, [L]])
    fp = np.concatenate([U_ref[idx], [U_ref[idx][0]]])
    u = np.interp(x_query.ravel(), xp, fp).reshape(-1, 1)
    return u, float(t_ref[idx])

# ----------------------------------------------------------------------
# Collocation points and physics-informed residual
# ----------------------------------------------------------------------
def sample_collocation_points(n_points, L, T, ht, seed=None):
    """x in [0,L), t in [ht, T-ht] (interior points, needed for centered differences).

    Points are laid out on a uniform grid (not sampled randomly): n_side = round(sqrt(n_points))
    values along each axis, combined via meshgrid, for n_side**2 total collocation points.
    """
    if T <= 2.0 * ht:
        raise ValueError("T must satisfy T > 2*ht.")

    n_side = max(1, round(math.sqrt(n_points)))
    x_lin = tc.linspace(0.0, L, int(n_side/2) + 1)[:-1]
    t_lin = tc.linspace(ht, T - ht, n_side)

    x_grid, t_grid = tc.meshgrid(x_lin, t_lin, indexing="ij")
    x = x_grid.reshape(-1, 1)
    t = t_grid.reshape(-1, 1)
    return x, t

# ----------------------------------------------------------------------
# Non-uniform mesh finite differences
#
# Both formulas fall back exactly to the standard symmetric central
# difference when h_L == h_R. They operate elementwise on tensors of
# shape (n_points, 1), so a whole (possibly scattered/non-uniform)
# collocation set is differentiated in one shot -- no Python loop over
# the mesh is needed, uniform or not.
# ----------------------------------------------------------------------
def nonuniform_first_derivative(f_left, f_center, f_right, h_L, h_R):
    """f'(x_i) via the 3-point asymmetric central difference on a non-uniform mesh."""
    return (
        (h_L / (h_R * (h_L + h_R))) * f_right
        + ((h_R - h_L) / (h_L * h_R)) * f_center
        - (h_R / (h_L * (h_L + h_R))) * f_left
    )

def nonuniform_second_derivative(f_left, f_center, f_right, h_L, h_R):
    """f''(x_i) via the 3-point central difference on a non-uniform mesh."""
    return (2.0 / (h_L + h_R)) * ((f_right - f_center) / h_R - (f_center - f_left) / h_L)

# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------
def train(model, x_f, t_f, loss_kwargs, epochs, lr, grad_clip=1.0,
          weight_decay=1.0e-6):
    """Minimizes the Burgers residual loss with AdamW + ReduceLROnPlateau."""
    L, nu, hx, ht = loss_kwargs["L"], loss_kwargs["nu"], loss_kwargs["hx"], loss_kwargs["ht"]
    x_p = tc.remainder(x_f + hx, L)
    x_m = tc.remainder(x_f - hx, L)
    t_p, t_m = t_f + ht, t_f - ht

    params = [p for p in model.parameters() if p.requires_grad]
    opt = tc.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=200, min_lr=1.0e-6
    )

    hist = {"loss": [], "residual_rms": []}
    #t0 = time.time()

    for _ in tqdm(range(epochs)):
        model.train()

        u_0 = model(tc.cat([x_f, t_f], dim=1))
        u_xp = model(tc.cat([x_p, t_f], dim=1))
        u_xm = model(tc.cat([x_m, t_f], dim=1))
        u_tp = model(tc.cat([x_f, t_p], dim=1))
        u_tm = model(tc.cat([x_f, t_m], dim=1))

        u_t = (u_tp - u_tm) / (2.0 * ht)
        u_x = (u_xp - u_xm) / (2.0 * hx)
        u_xx = (u_xp - 2.0 * u_0 + u_xm) / (hx**2)

        residual = u_t + u_0 * u_x - nu * u_xx
        loss = tc.mean(residual**2)

        opt.zero_grad()
        loss.backward()
        if grad_clip is not None:
            tc.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        loss_value = float(loss.detach())
        scheduler.step(loss_value)

        hist["loss"].append(loss_value)
        hist["residual_rms"].append(float(tc.sqrt(tc.mean(residual.detach() ** 2))))

    return model, hist

def train_autograd(model, x_f, t_f, loss_kwargs, epochs, lr, grad_clip=1.0,
                    weight_decay=1.0e-6, lbfgs_epochs=None, lbfgs_lr=1.0):
    """Same Burgers-residual training as `train`, but u_t, u_x, u_xx come from
    a single forward pass differentiated with autograd (torch.autograd.grad,
    create_graph=True) instead of finite differences on shifted collocation
    points. Only `nu` from `loss_kwargs` is used; hx/ht are not needed since
    no stencil is built.

    Training runs in two phases: `epochs` iterations of AdamW doing a broad,
    global search of parameter space (its lr decays via ReduceLROnPlateau as
    the loss stalls, so it actually settles instead of still bouncing around
    at a high lr when L-BFGS takes over), followed by `lbfgs_epochs`
    iterations of L-BFGS refining that region into a low-loss local minimum
    (defaults to `epochs` when not given). Set `lbfgs_epochs=0` to skip the
    second phase.

    The lowest-loss parameter snapshot seen across *both* phases is what
    gets loaded into `model` before returning -- not simply whichever
    iterate training happened to end on. This is what makes L-BFGS strictly
    help: even if a step lands somewhere much worse (a bad quasi-Newton
    step, or a noisy hand-off point from Adam), the snapshot rollback means
    the returned model can never be worse than the best point already
    found."""
    nu = loss_kwargs["nu"]

    x_f = x_f.clone().requires_grad_(True)
    t_f = t_f.clone().requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = tc.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # Decays lr as the loss plateaus: a larger lr early on drives the broad
    # global search, but without decay Adam is still bouncing around at that
    # same high lr on its last iteration, so L-BFGS would build its local
    # quadratic model from a noisy, badly-conditioned point instead of a
    # settled basin.
    scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(10, epochs // 10), min_lr=1.0e-4
    )

    hist = {"loss": [], "residual_rms": []}

    def compute_residual():
        u_0 = model(tc.cat([x_f, t_f], dim=1))

        u_x = tc.autograd.grad(u_0, x_f, grad_outputs=tc.ones_like(u_0), create_graph=True)[0]
        u_xx = tc.autograd.grad(u_x, x_f, grad_outputs=tc.ones_like(u_x), create_graph=True)[0]
        u_t = tc.autograd.grad(u_0, t_f, grad_outputs=tc.ones_like(u_0), create_graph=True)[0]

        return u_t + u_0 * u_x - nu * u_xx

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    best_state = snapshot()
    best_loss = float("inf")

    for _ in tqdm(range(epochs), desc="Adam"):
        model.train()

        residual = compute_residual()
        loss = tc.mean(residual**2)

        opt.zero_grad()
        loss.backward()
        if grad_clip is not None:
            tc.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        loss_value = float(loss.detach())
        scheduler.step(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_state = snapshot()

        hist["loss"].append(loss_value)
        hist["residual_rms"].append(float(tc.sqrt(tc.mean(residual.detach() ** 2))))

    if lbfgs_epochs is None:
        lbfgs_epochs = epochs

    if lbfgs_epochs > 0:
        # Hand L-BFGS the best basin Adam actually found, not whatever point
        # its last (possibly still-noisy) iterate landed on.
        model.load_state_dict(best_state)

        lbfgs_opt = tc.optim.LBFGS(
            params, lr=lbfgs_lr, max_iter=10, history_size=50,
            line_search_fn="strong_wolfe",
        )

        # Squared (not abs) residual: L-BFGS builds its Hessian approximation
        # from consecutive gradients, and the kink of |residual| at 0 makes
        # that curvature estimate unreliable, which is what drives loss
        # instability. Gradient clipping is skipped here for the same reason
        # -- it would distort the (s, y) pairs L-BFGS relies on. `strong_wolfe`
        # rejects any candidate step that doesn't give sufficient decrease,
        # but its bracketing/zoom search can still fall back to a non-improving
        # step if it can't find one inside its own iteration budget -- the
        # best-snapshot tracking below is what actually guarantees the final
        # model is never worse than the best point found.
        last = {}

        def closure():
            lbfgs_opt.zero_grad()
            residual = compute_residual()
            loss = tc.mean(residual**2)
            loss.backward()

            last["loss"] = float(loss.detach())
            last["residual_rms"] = float(tc.sqrt(tc.mean(residual.detach() ** 2)))
            return loss

        for _ in tqdm(range(lbfgs_epochs), desc="L-BFGS"):
            model.train()
            lbfgs_opt.step(closure)
            # Only the point the line search actually accepted this step is
            # logged -- `closure` is called several times per step to test
            # candidate step sizes, and logging every trial would make
            # rejected, never-applied candidates show up as fake spikes.
            loss_value = last["loss"]

            if math.isfinite(loss_value) and loss_value < best_loss:
                best_loss = loss_value
                best_state = snapshot()

            hist["loss"].append(loss_value)
            hist["residual_rms"].append(last["residual_rms"])

    # Always return the lowest-loss point seen across both phases, never
    # just whichever iterate the last epoch happened to land on.
    model.load_state_dict(best_state)
    return model, hist

def predict(model, x, t_val):
    """Evaluates the model on a spatial grid x at a fixed time t_val."""
    x = np.asarray(x).reshape(-1, 1)
    xt = tc.tensor(np.hstack([x, t_val * np.ones_like(x)]), dtype=tc.get_default_dtype())
    model.eval()
    with tc.no_grad():
        return model(xt).numpy().reshape(-1, 1)


def relative_l2_error(u_pred, u_ref):
    return float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_loss(histories, title="Training loss", save_path=None, show_cond=False):
    """histories: dict of label -> list of loss values."""
    plt.figure(figsize=(9, 4.5))
    for label, values in histories.items():
        #plt.plot(np.log10(np.asarray(values) + 1.0e-16), linewidth=2.0, label=label)
        plt.plot(np.asarray(values), linewidth=2.0, label=label)
    plt.xlabel("Epochs", fontsize=14)
    #plt.ylabel(r"$\log_{10}(\mathrm{loss})$", fontsize=14)
    plt.ylabel(r"$\mathrm{Loss}$", fontsize=14)
    plt.yscale("log")
    
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show_cond==True:
        plt.show()


def plot_solution(x, curves, ref=None, ref_label="FDM reference", title="", save_path=None,
                   show_cond=False, collocation_x=None):
    """
    x: spatial grid, shape (N,1).
    curves: dict of label -> predicted u array, shape (N,1). Used both for a
        single model's solution and for multi-model / multi-time comparisons.
    ref: optional reference array plotted as a thick solid line.
    collocation_x: optional array of the x-coordinates of the collocation
        points used for training; drawn as a rug of ticks along the x-axis
        (independent of the y-scale) so the point density is visible on top
        of the solution curve.
    """
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.gca()
    if ref is not None:
        ax.plot(x.ravel(), ref.ravel(), linewidth=3.0, label=ref_label)
    for label, u in curves.items():
        ax.plot(x.ravel(), u.ravel(), "--", linewidth=2.0, label=label)
    if collocation_x is not None:
        collocation_x = np.asarray(collocation_x).ravel()
        rug_trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
        ax.scatter(
            collocation_x, np.full_like(collocation_x, 0.02), transform=rug_trans,
            marker="|", s=200, linewidth=1.2, color="black", clip_on=False,
            label=f"collocation points (N={collocation_x.size})",
        )
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$u(x,t)$", fontsize=14)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show_cond==True:
        plt.show()
