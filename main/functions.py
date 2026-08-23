import glob
import json
import math
import os
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

def reference_pde_residual(x_f, t_f, x_ref, t_ref, U_ref, L, nu, hx, ht):
    """MSE of the PDE residual (u_t + u*u_x - nu*u_xx) evaluated on the FDM
    reference solution itself, at the same collocation points (x_f, t_f) and
    with the exact same finite-difference stencil (hx, ht) used by `train`'s
    loss. U_ref is bilinearly interpolated (periodic in x, linear in t) onto
    x_f, t_f and their +-hx/+-ht neighbors to build that residual.

    Used as a target/stopping value for the PINN training loss: since U_ref
    is itself only an approximate solution (its own FD scheme has nonzero
    truncation error), this residual is never exactly 0. It is the precision
    ceiling of the reference at this (hx, ht, N_COLLOCATION) -- once the
    PINN's residual loss reaches it, the PINN matches the reference's own
    discretization error at these points, and driving the loss lower is
    chasing noise in the baseline rather than the true PDE solution.
    """
    x_f = np.asarray(x_f, dtype=np.float64).reshape(-1)
    t_f = np.asarray(t_f, dtype=np.float64).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    t_ref = np.asarray(t_ref, dtype=np.float64)

    dx = x_ref[1] - x_ref[0]
    dt_ref = t_ref[1] - t_ref[0]
    nx = len(x_ref)
    nt = len(t_ref)

    def u_ref_interp(x, t):
        # Periodic bilinear interpolation of U_ref over its (x_ref, t_ref) grid.
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

        u00, u01 = U_ref[it0, ix0], U_ref[it0, ix1]
        u10, u11 = U_ref[it1, ix0], U_ref[it1, ix1]
        u0 = u00 * (1.0 - wx) + u01 * wx
        u1 = u10 * (1.0 - wx) + u11 * wx
        return u0 * (1.0 - wt) + u1 * wt

    u_0 = u_ref_interp(x_f, t_f)
    u_xp = u_ref_interp(x_f + hx, t_f)
    u_xm = u_ref_interp(x_f - hx, t_f)
    u_tp = u_ref_interp(x_f, t_f + ht)
    u_tm = u_ref_interp(x_f, t_f - ht)

    u_t = (u_tp - u_tm) / (2.0 * ht)
    u_x = (u_xp - u_xm) / (2.0 * hx)
    u_xx = (u_xp - 2.0 * u_0 + u_xm) / (hx**2)

    residual = u_t + u_0 * u_x - nu * u_xx
    return float(np.mean(residual**2))

# ----------------------------------------------------------------------
# Collocation points and physics-informed residual
# ----------------------------------------------------------------------
def sample_collocation_points(n_points, L, T, ht, seed=None, distribution="uniform"):
    """x in [0,L), t in [ht, T-ht] (interior points, needed for centered differences).

    distribution="uniform" (default): points are laid out on a uniform grid
    (not sampled randomly): n_side = round(sqrt(n_points)) values along each
    axis, combined via meshgrid, for n_side**2 total collocation points.

    distribution="random": n_points points drawn independently from a
    uniform distribution over the domain, x ~ U[0, L) and t ~ U[ht, T-ht].
    Pass `seed` for a reproducible draw. Leave `seed=None` to draw from the
    current (global) torch RNG state instead of a fresh local generator --
    that state advances on every call, so calling this repeatedly with
    seed=None (e.g. once per epoch via `train`'s `resample_fn`) yields a
    different set of random collocation points each time.
    """
    if T <= 2.0 * ht:
        raise ValueError("T must satisfy T > 2*ht.")

    if distribution == "uniform":
        n_side = max(1, round(math.sqrt(n_points)))
        x_lin = tc.linspace(0.0, L, int(n_side/2) + 1)[:-1]
        t_lin = tc.linspace(ht, T - ht, n_side)

        x_grid, t_grid = tc.meshgrid(x_lin, t_lin, indexing="ij")
        x = x_grid.reshape(-1, 1)
        t = t_grid.reshape(-1, 1)
    elif distribution == "random":
        generator = None
        if seed is not None:
            generator = tc.Generator().manual_seed(seed)
        x = tc.rand((n_points, 1), generator=generator) * L
        t = ht + tc.rand((n_points, 1), generator=generator) * (T - 2.0 * ht)
    else:
        raise ValueError("distribution must be 'uniform' or 'random'.")

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
          weight_decay=1.0e-6, resample_fn=None, lbfgs_epochs=None, lbfgs_lr=1.0,
          lbfgs_resample_fn=None, lbfgs_patience=50, lbfgs_tol=1.0e-9,
          target_loss=None):
    """Minimizes the Burgers residual loss with AdamW + ReduceLROnPlateau,
    followed by an L-BFGS refinement phase -- the same two-phase schedule as
    `train_autograd` (see its docstring for the rationale). The only
    difference from `train_autograd` is how u_t, u_x, u_xx are obtained:
    here they come from finite differences on shifted collocation points
    instead of autograd.

    resample_fn: optional zero-argument callable returning a fresh (x_f, t_f)
    pair, called at the start of every epoch during the AdamW phase to
    replace the collocation points used that epoch. Pass a closure over
    `sample_collocation_points(..., distribution="random")` (with seed left
    as None) to train on different random collocation points each epoch;
    leave as None to keep training on the fixed x_f, t_f passed in (e.g. a
    uniform grid).

    lbfgs_epochs: number of L-BFGS iterations for the second phase (defaults
    to `epochs` when not given). Set lbfgs_epochs=0 to skip it and recover
    the old AdamW-only behavior.

    lbfgs_resample_fn: optional zero-argument callable, same shape as
    resample_fn, called once per L-BFGS *outer* iteration (before that
    iteration's `lbfgs_opt.step(closure)`) to replace the collocation
    points used from then on. A single step() call still evaluates
    `closure` several times internally (the strong_wolfe bracketing/zoom
    search) and every one of those internal calls sees the same, just
    -resampled point set -- only the set changes *between* outer steps, so
    the line search itself is never fed a moving target. What does shift
    across outer steps is the (s, y) curvature pairs L-BFGS accumulates in
    its history, since they're now measured on slightly different
    objectives; this is a real tradeoff against a fully fixed point set,
    so prefer a lbfgs_resample_fn that draws a reasonably large batch (a
    handful of points relative to the model's parameter count lets L-BFGS
    drive the residual to ~0 by pure interpolation in one step, which is
    what produces a long flat tail on the loss curve rather than genuine
    convergence). Leave as None (default) to keep the old behavior: a
    single fixed point set for the whole L-BFGS phase (whatever x_f, t_f
    ended up being after the last AdamW epoch).

    lbfgs_patience, lbfgs_tol: early-stopping controls for the L-BFGS loop.
    If `best_loss` fails to improve by more than `lbfgs_tol` for
    `lbfgs_patience` consecutive outer iterations, the loop stops instead
    of burning through the rest of `lbfgs_epochs` -- this is what a flat
    loss tail actually means (L-BFGS has nothing left to descend) and there
    is no point paying for more iterations once it happens. Set
    lbfgs_patience=None to disable early stopping and always run the full
    lbfgs_epochs.

    target_loss: optional stopping target for the residual loss (e.g. from
    `reference_pde_residual`, the reference FDM solution's own discretization
    error at these collocation points). Training stops as soon as an
    iterate's loss reaches `<= target_loss` -- in the AdamW phase this also
    skips the L-BFGS phase entirely, since there is nothing left to refine
    once the PINN already matches the reference's own precision. This is the
    *first* stopping condition; whenever it is never reached, `epochs` (plus
    `lbfgs_epochs`) is what stops training, exactly as when target_loss is
    left at its default None.
    """
    L, nu, hx, ht = loss_kwargs["L"], loss_kwargs["nu"], loss_kwargs["hx"], loss_kwargs["ht"]

    def stencil_points(x_f, t_f):
        x_p = tc.remainder(x_f + hx, L)
        x_m = tc.remainder(x_f - hx, L)
        return x_p, x_m, t_f + ht, t_f - ht

    x_p, x_m, t_p, t_m = stencil_points(x_f, t_f)

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

    def compute_residual(x_f, x_p, x_m, t_f, t_p, t_m):
        u_0 = model(tc.cat([x_f, t_f], dim=1))
        u_xp = model(tc.cat([x_p, t_f], dim=1))
        u_xm = model(tc.cat([x_m, t_f], dim=1))
        u_tp = model(tc.cat([x_f, t_p], dim=1))
        u_tm = model(tc.cat([x_f, t_m], dim=1))

        u_t = (u_tp - u_tm) / (2.0 * ht)
        u_x = (u_xp - u_xm) / (2.0 * hx)
        u_xx = (u_xp - 2.0 * u_0 + u_xm) / (hx**2)

        return u_t + u_0 * u_x - nu * u_xx

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    best_state = snapshot()
    best_loss = float("inf")
    target_reached = False

    for _ in tqdm(range(epochs), desc="AdamW"):
        model.train()

        if resample_fn is not None:
            x_f, t_f = resample_fn()
            x_p, x_m, t_p, t_m = stencil_points(x_f, t_f)

        residual = compute_residual(x_f, x_p, x_m, t_f, t_p, t_m)
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

        # First stopping condition: the residual loss has already reached the
        # reference's own precision at these points -- no need for L-BFGS to
        # refine further, or to burn through the rest of `epochs`.
        if target_loss is not None and loss_value <= target_loss:
            target_reached = True
            break

    if lbfgs_epochs is None:
        lbfgs_epochs = epochs

    if lbfgs_epochs > 0 and not target_reached:
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
            residual = compute_residual(x_f, x_p, x_m, t_f, t_p, t_m)
            loss = tc.mean(residual**2)
            loss.backward()

            last["loss"] = float(loss.detach())
            last["residual_rms"] = float(tc.sqrt(tc.mean(residual.detach() ** 2)))
            return loss

        stall_count = 0
        for _ in tqdm(range(lbfgs_epochs), desc="L-BFGS"):
            model.train()

            if lbfgs_resample_fn is not None:
                x_f, t_f = lbfgs_resample_fn()
                x_p, x_m, t_p, t_m = stencil_points(x_f, t_f)

            lbfgs_opt.step(closure)
            # Only the point the line search actually accepted this step is
            # logged -- `closure` is called several times per step to test
            # candidate step sizes, and logging every trial would make
            # rejected, never-applied candidates show up as fake spikes.
            loss_value = last["loss"]

            if math.isfinite(loss_value) and loss_value < best_loss - lbfgs_tol:
                best_loss = loss_value
                best_state = snapshot()
                stall_count = 0
            else:
                stall_count += 1

            hist["loss"].append(loss_value)
            hist["residual_rms"].append(last["residual_rms"])

            if target_loss is not None and loss_value <= target_loss:
                target_reached = True
                break

            if lbfgs_patience is not None and stall_count >= lbfgs_patience:
                break

    hist["target_loss"] = target_loss
    hist["target_reached"] = target_reached

    # Always return the lowest-loss point seen across both phases, never
    # just whichever iterate the last epoch happened to land on.
    model.load_state_dict(best_state)
    return model, hist

def train_autograd(model, x_f, t_f, loss_kwargs, epochs, lr, grad_clip=1.0,
                    weight_decay=1.0e-6, resample_fn=None, lbfgs_epochs=None, lbfgs_lr=1.0,
                    lbfgs_resample_fn=None, lbfgs_patience=50, lbfgs_tol=1.0e-9):
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

    resample_fn: optional zero-argument callable returning a fresh (x_f, t_f)
    pair, called at the start of every epoch during the AdamW phase to
    replace the collocation points used that epoch. Pass a closure over
    `sample_collocation_points(..., distribution="random")` (with seed left
    as None) to train on different random collocation points each epoch;
    leave as None to keep training on the fixed x_f, t_f passed in (e.g. a
    uniform grid).

    lbfgs_resample_fn: optional zero-argument callable, same shape as
    resample_fn, called once per L-BFGS *outer* iteration (before that
    iteration's `lbfgs_opt.step(closure)`) to replace the collocation
    points used from then on. A single step() call still evaluates
    `closure` several times internally (the strong_wolfe bracketing/zoom
    search) and every one of those internal calls sees the same, just
    -resampled point set -- only the set changes *between* outer steps, so
    the line search itself is never fed a moving target. What does shift
    across outer steps is the (s, y) curvature pairs L-BFGS accumulates in
    its history, since they're now measured on slightly different
    objectives; this is a real tradeoff against a fully fixed point set,
    so prefer a lbfgs_resample_fn that draws a reasonably large batch (a
    handful of points relative to the model's parameter count lets L-BFGS
    drive the residual to ~0 by pure interpolation in one step, which is
    what produces a long flat tail on the loss curve rather than genuine
    convergence). Leave as None (default) to keep the old behavior: a
    single fixed point set for the whole L-BFGS phase (whatever x_f, t_f
    ended up being after the last AdamW epoch).

    lbfgs_patience, lbfgs_tol: early-stopping controls for the L-BFGS loop.
    If `best_loss` fails to improve by more than `lbfgs_tol` for
    `lbfgs_patience` consecutive outer iterations, the loop stops instead
    of burning through the rest of `lbfgs_epochs` -- this is what a flat
    loss tail actually means (L-BFGS has nothing left to descend) and there
    is no point paying for more iterations once it happens. Set
    lbfgs_patience=None to disable early stopping and always run the full
    lbfgs_epochs.

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

        if resample_fn is not None and _ % 10 == 0:  # Resample every 10 iterations
            x_f, t_f = resample_fn()
            x_f = x_f.clone().requires_grad_(True)
            t_f = t_f.clone().requires_grad_(True)

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

        stall_count = 0
        for _ in tqdm(range(lbfgs_epochs), desc="L-BFGS"):
            model.train()

            if lbfgs_resample_fn is not None:
                x_f, t_f = lbfgs_resample_fn()
                x_f = x_f.clone().requires_grad_(True)
                t_f = t_f.clone().requires_grad_(True)

            lbfgs_opt.step(closure)
            # Only the point the line search actually accepted this step is
            # logged -- `closure` is called several times per step to test
            # candidate step sizes, and logging every trial would make
            # rejected, never-applied candidates show up as fake spikes.
            loss_value = last["loss"]

            if math.isfinite(loss_value) and loss_value < best_loss - lbfgs_tol:
                best_loss = loss_value
                best_state = snapshot()
                stall_count = 0
            else:
                stall_count += 1

            hist["loss"].append(loss_value)
            hist["residual_rms"].append(last["residual_rms"])

            if lbfgs_patience is not None and stall_count >= lbfgs_patience:
                break

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
def asymmetric_yerr(mean, vmin, vmax):
    """Builds the (2, N) `yerr` array matplotlib's `errorbar()` expects for
    asymmetric error bars that reach down to `vmin` and up to `vmax` at each
    point, instead of a symmetric +/-std whisker.

    Use this (with `vmin`/`vmax` taken from the actual per-seed samples)
    instead of `yerr=std` whenever the error bars are drawn on a log-scale
    axis: mean +/- std routinely goes negative when std > mean, which is
    common with only a handful of seeds and a skewed, positive-only metric
    like MSE/MAE -- and a log axis cannot render that, so matplotlib clips
    the lower whisker all the way to the bottom of the plot instead of
    stopping below the mean. Because `vmin`/`vmax` come from samples that
    were actually observed (always > 0 for MSE/MAE), the lower whisker can
    never go negative or below the smallest point actually measured.
    """
    mean = np.asarray(mean, dtype=float)
    vmin = np.asarray(vmin, dtype=float)
    vmax = np.asarray(vmax, dtype=float)
    return np.vstack([mean - vmin, vmax - mean])


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
                   show_cond=False, collocation_x=None, colors=None, ref_color=None):
    """
    x: spatial grid, shape (N,1).
    curves: dict of label -> predicted u array, shape (N,1). Used both for a
        single model's solution and for multi-model / multi-time comparisons.
    ref: optional reference array plotted as a thick solid line.
    collocation_x: optional array of the x-coordinates of the collocation
        points used for training; drawn as a rug of ticks along the x-axis
        (independent of the y-scale) so the point density is visible on top
        of the solution curve.
    colors: optional dict of label -> matplotlib color, for curves that need
        a fixed color (e.g. a classical baseline drawn in black) instead of
        the default color cycle.
    ref_color: optional matplotlib color for the reference curve; defaults to
        None (matplotlib's default color cycle) when not given.
    """
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.gca()
    if ref is not None:
        ax.plot(x.ravel(), ref.ravel(), linewidth=3.0, label=ref_label, color=ref_color)
    for label, u in curves.items():
        color = None if colors is None else colors.get(label)
        ax.plot(x.ravel(), u.ravel(), "--", linewidth=2.0, label=label, color=color)
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
    #ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show_cond==True:
        plt.show()

# ----------------------------------------------------------------------
# plote_*.py helpers -- checkpoint discovery, error-vs-width statistics,
# and their companion plots. Every plote_*.py script imports these and
# keeps only its own parameters (paths, labels, colors) plus the call.
# `model_cls` (ClassicalPINN2 / QuantumPINNPennyLane) is passed in by the
# caller rather than imported here, since models.py/models_pennylane.py
# both import from this module -- importing them back would be circular.
# ----------------------------------------------------------------------
def find_classical_configs(t_dir, collocation_distribution=None):
    """Returns [(intra_neuron, run_id, pt_path), ...] for every classical
    Cmodel_NEURO* checkpoint in `t_dir` whose COLLOCATION_DISTRIBUTION
    matches `collocation_distribution` ("uniform" or "random"; leave as
    None to include every distribution found), reading intra_neuron from
    its companion `_config.json` (the run_id/sample index comes from the
    trailing `_<run_id>` in the filename)."""
    entries = []
    for config_path in glob.glob(os.path.join(t_dir, "Cmodel_*_NEURO*_config.json")):
        pt_path = config_path[: -len("_config.json")] + ".pt"
        if not os.path.exists(pt_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        distribution = cfg["training_params"].get("COLLOCATION_DISTRIBUTION")
        if collocation_distribution is not None and distribution != collocation_distribution:
            continue
        intra_neuron = cfg["config"]["intra_neuron"]
        run_id = int(os.path.basename(pt_path)[: -len(".pt")].rsplit("_", 1)[1])
        entries.append((intra_neuron, run_id, pt_path))
    return sorted(entries)


def find_quantum_configs(t_dir, input_rotation_axis="ry", collocation_distribution="random"):
    """Returns [(fixed_layer, n_qubits, seed, pt_path), ...] for every
    Qmodel_<collocation_distribution>_NQUBITS* checkpoint in `t_dir` whose
    input_rotation_axis matches, reading fixed_layer/n_qubits/seed from its
    companion `_config.json`."""
    entries = []
    pattern = f"Qmodel_{collocation_distribution}_NQUBITS*_config.json"
    for config_path in glob.glob(os.path.join(t_dir, pattern)):
        pt_path = config_path[: -len("_config.json")] + ".pt"
        if not os.path.exists(pt_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)["config"]
        if cfg.get("input_rotation_axis") != input_rotation_axis:
            continue
        entries.append((cfg["fixed_layer"], cfg["n_qubits"], cfg["seed"], pt_path))
    return sorted(entries)


def compute_classical_mean_errors(t_dir, model_cls, nu):
    """Loads every classical model in `t_dir`, evaluates each against the FDM
    reference at t=Tfin, and averages the MSE/MAE across the run_id samples
    (one per amostragem) for each INTRA_NEURO value. Returns (intra_neurons,
    mean_mse, mean_mae, n_samples, Tfin), one point per INTRA_NEURO. Used by
    the legacy `old/plote_classico.py`."""
    entries = find_classical_configs(t_dir)
    if not entries:
        return [], [], [], [], None

    first_model = model_cls.load(entries[0][2])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_neuron = {}
    for intra_neuron, _run_id, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_neuron.setdefault(intra_neuron, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    intra_neurons = sorted(errors_by_neuron)
    mse = [float(np.mean(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mae = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
    return intra_neurons, mse, mae, n_samples, Tfin


def plot_classical_mean_errors(intra_neurons, mse, mae, Tfin, save_path,
                                color_mse="#2a78d6", color_mae="#1baf7a"):
    """Companion plot for `compute_classical_mean_errors` (legacy, mean-only)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(intra_neurons, mse, "o-", color=color_mse, linewidth=2.0, markersize=6, label="MSE")
    ax.plot(intra_neurons, mae, "o-", color=color_mae, linewidth=2.0, markersize=6, label="MAE")
    ax.set_xlabel("INTRA_NEURO", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(intra_neurons)
    ax.set_title(f"Classical PINN mean error vs INTRA_NEURO (T = {Tfin})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_classical_mae_stats(t_dir, model_cls, nu):
    """Loads every classical model in `t_dir`, evaluates each against the FDM
    reference at t=Tfin, and returns per-INTRA_NEURO mean/std/min/max MAE
    across the run_id samples. Returns (intra_neurons, mae_mean, mae_std,
    mae_min, mae_max, n_samples, Tfin), one point per INTRA_NEURO.
    mae_min/mae_max (the extremes actually observed across run_ids) are what
    the plot uses for its error bars -- unlike mean +/- std, they can never
    go negative on the plot's log-scale y-axis. Used by the legacy
    `plote_classico_std copy.py` (MAE-only, ignores COLLOCATION_DISTRIBUTION)."""
    entries = find_classical_configs(t_dir)
    if not entries:
        return [], [], [], [], [], [], None

    first_model = model_cls.load(entries[0][2])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_neuron = {}
    for intra_neuron, _run_id, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_neuron.setdefault(intra_neuron, [])
        entry.append(float(np.mean(np.abs(u_pred - u_ref))))

    intra_neurons = sorted(errors_by_neuron)
    mae_mean = [float(np.mean(errors_by_neuron[n])) for n in intra_neurons]
    mae_std = [float(np.std(errors_by_neuron[n])) for n in intra_neurons]
    mae_min = [float(np.min(errors_by_neuron[n])) for n in intra_neurons]
    mae_max = [float(np.max(errors_by_neuron[n])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]) for n in intra_neurons]
    return intra_neurons, mae_mean, mae_std, mae_min, mae_max, n_samples, Tfin


def plot_classical_mae_std(intra_neurons, mae_mean, mae_min, mae_max, Tfin, save_path,
                            color_mae="#1baf7a"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    yerr = asymmetric_yerr(mae_mean, mae_min, mae_max)
    ax.errorbar(intra_neurons, mae_mean, yerr=yerr, fmt="o-", color=color_mae,
                linewidth=2.0, markersize=6, capsize=4, label="MAE (mean, [min,max])")
    ax.set_xlabel("INTRA_NEURO", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(intra_neurons)
    ax.set_title(f"Classical PINN MAE vs INTRA_NEURO (T = {Tfin}, legacy data)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_classical_error_stats_std(t_dir, model_cls, nu, collocation_distribution):
    """Loads every classical model in `t_dir` whose COLLOCATION_DISTRIBUTION
    matches `collocation_distribution`, evaluates each against the FDM
    reference at t=Tfin, and returns per-INTRA_NEURO mean/std/min/max MSE and
    MAE across the run_id samples. Returns (intra_neurons, mse_mean, mse_std,
    mse_min, mse_max, mae_mean, mae_std, mae_min, mae_max, n_samples, Tfin),
    one point per INTRA_NEURO. The _min/_max values (the extremes actually
    observed across run_ids) are what the plot uses for its error bars --
    unlike mean +/- std, they can never go negative on the plot's log-scale
    y-axis. Used by `plote_classico_std.py`."""
    entries = find_classical_configs(t_dir, collocation_distribution=collocation_distribution)
    if not entries:
        return [], [], [], [], [], [], [], [], [], [], None

    first_model = model_cls.load(entries[0][2])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_neuron = {}
    for intra_neuron, _run_id, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_neuron.setdefault(intra_neuron, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    intra_neurons = sorted(errors_by_neuron)
    mse_mean = [float(np.mean(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_std = [float(np.std(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_min = [float(np.min(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_max = [float(np.max(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mae_mean = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_std = [float(np.std(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_min = [float(np.min(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_max = [float(np.max(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
    return (intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, mae_min,
            mae_max, n_samples, Tfin)


def plot_classical_errors_std(stats_by_distribution, Tfin, save_path,
                               color_mse="#2a78d6", dist_styles=None):
    """`stats_by_distribution` maps COLLOCATION_DISTRIBUTION -> (intra_neurons,
    mse_mean, mse_min, mse_max) for that distribution; distributions with no
    models are omitted by the caller. `dist_styles` maps distribution name ->
    dict(fmt=..., markerfacecolor=...); defaults to solid filled markers for
    "uniform" and dashed hollow markers for "random", so both variants can
    share one set of axes."""
    if dist_styles is None:
        dist_styles = {
            "uniform": dict(fmt="o-", markerfacecolor=None),
            "random": dict(fmt="s--", markerfacecolor="none"),
        }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_neurons = set()
    for distribution, (intra_neurons, mse_mean, mse_min,
                        mse_max) in stats_by_distribution.items():
        all_neurons.update(intra_neurons)
        style = dist_styles.get(distribution, dict(fmt="o-", markerfacecolor=None))
        mse_yerr = asymmetric_yerr(mse_mean, mse_min, mse_max)
        ax.errorbar(intra_neurons, mse_mean, yerr=mse_yerr, fmt=style["fmt"], color=color_mse,
                    markerfacecolor=style["markerfacecolor"], linewidth=2.0, markersize=6,
                    capsize=4, label=f"MSE ({distribution})")
    ax.set_xlabel("INTRA_NEURO", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_neurons))
    ax.set_title(f"Classical PINN error vs INTRA_NEURO (T = {Tfin})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def find_classical_configs_by_n_collocation(t_dir, collocation_distribution=None, intra_neuron=None):
    """Returns [(n_collocation, run_id, pt_path), ...] for every classical
    Cmodel_NEURO* checkpoint in `t_dir` whose COLLOCATION_DISTRIBUTION
    matches `collocation_distribution` ("uniform" or "random"; leave as None
    to include every distribution found) and whose INTRA_NEURO architecture
    matches `intra_neuron` (leave as None to include every width found),
    reading N_COLLOCATION from each checkpoint's training_params. Mirrors
    `find_classical_configs`, but keyed by N_COLLOCATION (the collocation
    -sampling sweep in `run_classical.py`) instead of by INTRA_NEURO (the
    width sweep)."""
    entries = []
    for config_path in glob.glob(os.path.join(t_dir, "Cmodel_*_NEURO*_config.json")):
        pt_path = config_path[: -len("_config.json")] + ".pt"
        if not os.path.exists(pt_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        training_params = cfg.get("training_params", {})
        distribution = training_params.get("COLLOCATION_DISTRIBUTION")
        if collocation_distribution is not None and distribution != collocation_distribution:
            continue
        if intra_neuron is not None and cfg["config"].get("intra_neuron") != intra_neuron:
            continue
        n_collocation = training_params.get("N_COLLOCATION")
        if n_collocation is None:
            continue
        run_id = int(os.path.basename(pt_path)[: -len(".pt")].rsplit("_", 1)[1])
        entries.append((n_collocation, run_id, pt_path))
    return sorted(entries)


def compute_classical_error_stats_vs_collocation(t_dir, model_cls, nu, collocation_distribution,
                                                   intra_neuron=None):
    """Loads every classical model in `t_dir` matching `collocation_distribution`
    (and, when given, a fixed `intra_neuron` architecture width), evaluates
    each against the FDM reference at t=Tfin, and returns per-N_COLLOCATION
    mean/std/variance MSE and MAE across the run_id seeds. Returns
    (n_collocation, mse_mean, mse_std, mse_var, mae_mean, mae_std, mae_var,
    n_samples, Tfin), one point per N_COLLOCATION value actually found.
    Companion to `plot_classical_error_vs_collocation`; mirrors
    `compute_classical_error_stats_std`, but grouped by N_COLLOCATION
    instead of by INTRA_NEURO."""
    entries = find_classical_configs_by_n_collocation(
        t_dir, collocation_distribution=collocation_distribution, intra_neuron=intra_neuron)
    if not entries:
        return [], [], [], [], [], [], [], [], None

    first_model = model_cls.load(entries[0][2])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_n = {}
    for n_collocation, _run_id, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_n.setdefault(n_collocation, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    n_collocation_list = sorted(errors_by_n)
    mse_mean = [float(np.mean(errors_by_n[n]["mse"])) for n in n_collocation_list]
    mse_std = [float(np.std(errors_by_n[n]["mse"])) for n in n_collocation_list]
    mse_var = [float(np.var(errors_by_n[n]["mse"])) for n in n_collocation_list]
    mae_mean = [float(np.mean(errors_by_n[n]["mae"])) for n in n_collocation_list]
    mae_std = [float(np.std(errors_by_n[n]["mae"])) for n in n_collocation_list]
    mae_var = [float(np.var(errors_by_n[n]["mae"])) for n in n_collocation_list]
    n_samples = [len(errors_by_n[n]["mse"]) for n in n_collocation_list]
    return (n_collocation_list, mse_mean, mse_std, mse_var, mae_mean, mae_std, mae_var,
            n_samples, Tfin)


def plot_classical_error_vs_collocation(n_collocation, mse_mean, mse_var, Tfin, save_path,
                                         color_mse="#2a78d6"):
    """MSE (mean +/- variance across seeds) between the classical PINN and
    the FDM reference, plotted against N_COLLOCATION. Points sit at equally
    spaced positions on the x-axis regardless of the actual gap between
    consecutive N_COLLOCATION values (e.g. 100 -> 250 -> 500) -- only the
    tick label at each position shows the real value -- so the sampling
    density along the sweep doesn't visually distort the curve. Companion
    plot for `compute_classical_error_stats_vs_collocation`."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(n_collocation))
    ax.errorbar(positions, mse_mean, yerr=mse_var, fmt="o-", color=color_mse,
                linewidth=2.0, markersize=6, capsize=4, label="MSE (mean +/- variance)")
    ax.set_xlabel("N_COLLOCATION", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in n_collocation])
    ax.set_title(f"Classical PINN: MSE vs N_COLLOCATION (T = {Tfin})", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_classical_stopping_stats_vs_collocation(t_dir, collocation_distribution, intra_neuron=None):
    """Reads STOPPED_AT_STEP and TARGET_REACHED from every classical
    checkpoint's training_params in `t_dir` (no model loading needed --
    config json only), grouped by N_COLLOCATION. Returns (n_collocation,
    steps_mean, steps_std, steps_var, target_reached_frac, n_samples, Tfin),
    one point per N_COLLOCATION value found: how many AdamW+L-BFGS
    iterations training actually ran before hitting `target_loss` (or the
    EPOCHS+LBFGS_EPOCHS fallback), averaged over the run_id seeds, plus the
    fraction of seeds that reached `target_loss` rather than exhausting the
    epoch budget. Companion stat to `compute_classical_error_stats_vs_collocation`,
    read from the same checkpoints but reporting training cost instead of
    accuracy."""
    steps_by_n = {}
    Tfin = None
    for config_path in glob.glob(os.path.join(t_dir, "Cmodel_*_NEURO*_config.json")):
        with open(config_path) as f:
            cfg = json.load(f)
        training_params = cfg.get("training_params", {})
        distribution = training_params.get("COLLOCATION_DISTRIBUTION")
        if collocation_distribution is not None and distribution != collocation_distribution:
            continue
        if intra_neuron is not None and cfg["config"].get("intra_neuron") != intra_neuron:
            continue
        n_collocation = training_params.get("N_COLLOCATION")
        stopped_at_step = training_params.get("STOPPED_AT_STEP")
        if n_collocation is None or stopped_at_step is None:
            continue
        Tfin = cfg["config"].get("Tfin", Tfin)
        entry = steps_by_n.setdefault(n_collocation, {"steps": [], "reached": []})
        entry["steps"].append(stopped_at_step)
        entry["reached"].append(bool(training_params.get("TARGET_REACHED")))

    n_collocation_list = sorted(steps_by_n)
    steps_mean = [float(np.mean(steps_by_n[n]["steps"])) for n in n_collocation_list]
    steps_std = [float(np.std(steps_by_n[n]["steps"])) for n in n_collocation_list]
    steps_var = [float(np.var(steps_by_n[n]["steps"])) for n in n_collocation_list]
    target_reached_frac = [float(np.mean(steps_by_n[n]["reached"])) for n in n_collocation_list]
    n_samples = [len(steps_by_n[n]["steps"]) for n in n_collocation_list]
    return n_collocation_list, steps_mean, steps_std, steps_var, target_reached_frac, n_samples, Tfin


def plot_classical_stopping_steps_vs_collocation(n_collocation, steps_mean, steps_var, Tfin, save_path,
                                                   color_mean="#2a78d6", color_var="#e34948"):
    """Number of training steps (AdamW + L-BFGS) until the stopping
    criterion was hit, vs N_COLLOCATION -- mean on the left axis, variance
    across the run_id seeds on the right axis (plotted separately, not as an
    error bar: mean and variance of a step *count* live on very different
    scales -- steps vs steps^2 -- so overlaying them as mean +/- variance
    would put the whisker far outside the visible range). Points sit at
    equally spaced x positions, same convention as
    `plot_classical_error_vs_collocation`. Companion plot for
    `compute_classical_stopping_stats_vs_collocation`."""
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(n_collocation))

    line_mean = ax1.plot(positions, steps_mean, "o-", color=color_mean, linewidth=2.0,
                          markersize=6, label="Média de passos")
    ax1.set_xlabel("N_COLLOCATION", fontsize=14)
    ax1.set_ylabel("Passos até a parada (média)", fontsize=14, color=color_mean)
    ax1.tick_params(axis="y", labelcolor=color_mean)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([str(n) for n in n_collocation])
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    line_var = ax2.plot(positions, steps_var, "s--", color=color_var, linewidth=2.0,
                         markersize=6, label="Variância de passos")
    ax2.set_ylabel("Variância dos passos", fontsize=14, color=color_var)
    ax2.tick_params(axis="y", labelcolor=color_var)

    lines = line_mean + line_var
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    #ax1.set_yscale("log")
    #ax1.set_title(f"Classical PINN: passos até a parada vs N_COLLOCATION (T = {Tfin})", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_quantum_error_stats_std(t_dir, model_cls, nu, input_rotation_axis="ry"):
    """Loads every quantum model in `t_dir`, evaluates each against the FDM
    reference at t=Tfin, and returns per-(FIXED_LAYER, N_QUBITS) mean/std/
    min/max MAE across the SEED samples. Returns (stats, Tfin), where stats
    is {fixed_layer: (n_qubits, mae_mean, mae_std, mae_min, mae_max,
    n_samples)}, one entry per FIXED_LAYER found. mae_min/mae_max (the
    extremes actually observed across seeds) are what the plot uses for its
    error bars -- unlike mean +/- std, they can never go negative on the
    plot's log-scale y-axis. Used by `plote_quantico_std.py`."""
    entries = find_quantum_configs(t_dir, input_rotation_axis=input_rotation_axis)
    if not entries:
        return {}, None

    first_model = model_cls.load(entries[0][3])
    L, Tfin = first_model.L, first_model.Tfin

    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    errors_by_layer = {}
    for fixed_layer, n_qubits, _seed, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        by_n_qubits = errors_by_layer.setdefault(fixed_layer, {})
        by_n_qubits.setdefault(n_qubits, []).append(float(np.mean(np.abs(u_pred - u_ref))))

    stats = {}
    for fixed_layer, by_n_qubits in errors_by_layer.items():
        n_qubits_list = sorted(by_n_qubits)
        mae_mean = [float(np.mean(by_n_qubits[n])) for n in n_qubits_list]
        mae_std = [float(np.std(by_n_qubits[n])) for n in n_qubits_list]
        mae_min = [float(np.min(by_n_qubits[n])) for n in n_qubits_list]
        mae_max = [float(np.max(by_n_qubits[n])) for n in n_qubits_list]
        n_samples = [len(by_n_qubits[n]) for n in n_qubits_list]
        stats[fixed_layer] = (n_qubits_list, mae_mean, mae_std, mae_min, mae_max, n_samples)
    return stats, Tfin


def plot_quantum_errors_std(stats, Tfin, save_path, colors=None):
    if colors is None:
        colors = ["#2a78d6", "#e34948", "#eb6834", "#4a3aa7", "#1baf7a", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_n_qubits = set()
    for (fixed_layer, (n_qubits, mae_mean, _mae_std, mae_min, mae_max, _n_samples)), color in zip(
            sorted(stats.items()), colors):
        all_n_qubits.update(n_qubits)
        yerr = asymmetric_yerr(mae_mean, mae_min, mae_max)
        ax.errorbar(n_qubits, mae_mean, yerr=yerr, fmt="o-", color=color,
                    linewidth=2.0, markersize=6, capsize=4, label=fixed_layer)
    ax.set_xlabel("Number of qubits", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_n_qubits))
    #ax.set_title(f"Quantum PINN MAE vs number of qubits (T = {Tfin})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_quantum_errors_comparativo(t_dir, model_cls, x_plot, u_ref, Tfin,
                                        input_rotation_axis, collocation_distribution):
    """Per-(FIXED_LAYER, N_QUBITS) MSE/MAE stats against a caller-supplied
    `u_ref` grid, used by `plote_comparativo.py` to compare quantum models
    against the classical baseline on the same x_plot/u_ref/Tfin."""
    entries = find_quantum_configs(
        t_dir, input_rotation_axis=input_rotation_axis, collocation_distribution=collocation_distribution
    )

    errors_by_layer = {}
    for fixed_layer, n_qubits, _seed, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        by_n = errors_by_layer.setdefault(fixed_layer, {})
        entry = by_n.setdefault(n_qubits, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    stats = {}
    for fixed_layer, by_n in errors_by_layer.items():
        n_list = sorted(by_n)
        mse_mean = [float(np.mean(by_n[n]["mse"])) for n in n_list]
        mse_std = [float(np.std(by_n[n]["mse"])) for n in n_list]
        mse_min = [float(np.min(by_n[n]["mse"])) for n in n_list]
        mse_max = [float(np.max(by_n[n]["mse"])) for n in n_list]
        mae_mean = [float(np.mean(by_n[n]["mae"])) for n in n_list]
        mae_std = [float(np.std(by_n[n]["mae"])) for n in n_list]
        n_samples = [len(by_n[n]["mse"]) for n in n_list]
        stats[fixed_layer] = (n_list, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples)
    return stats


def compute_classical_errors_comparativo(t_dir, model_cls, x_plot, u_ref, Tfin, collocation_distribution):
    """Per-INTRA_NEURO MSE/MAE stats against a caller-supplied `u_ref` grid,
    used by `plote_comparativo.py` to compare classical models against the
    quantum ones on the same x_plot/u_ref/Tfin."""
    entries = find_classical_configs(t_dir, collocation_distribution=collocation_distribution)
    if not entries:
        return None

    errors_by_neuron = {}
    for intra_neuron, _run_id, pt_path in entries:
        model = model_cls.load(pt_path)
        u_pred = predict(model, x_plot, Tfin)
        entry = errors_by_neuron.setdefault(intra_neuron, {"mse": [], "mae": []})
        entry["mse"].append(float(np.mean((u_pred - u_ref) ** 2)))
        entry["mae"].append(float(np.mean(np.abs(u_pred - u_ref))))

    intra_neurons = sorted(errors_by_neuron)
    mse_mean = [float(np.mean(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_std = [float(np.std(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_min = [float(np.min(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mse_max = [float(np.max(errors_by_neuron[n]["mse"])) for n in intra_neurons]
    mae_mean = [float(np.mean(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    mae_std = [float(np.std(errors_by_neuron[n]["mae"])) for n in intra_neurons]
    n_samples = [len(errors_by_neuron[n]["mse"]) for n in intra_neurons]
    return intra_neurons, mse_mean, mse_std, mse_min, mse_max, mae_mean, mae_std, n_samples


def plot_comparison(quantum_stats, classical_data, Tfin, save_path,
                     color_classical="black", colors_quantum=None, fallback_colors=None):
    if colors_quantum is None:
        colors_quantum = {"cz_ring": "#2a78d6", "ring_cnot": "#1baf7a"}
    if fallback_colors is None:
        fallback_colors = ["#eb6834", "#4a3aa7", "#e87ba4"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_widths = set()

    fallback = iter(fallback_colors)
    for fixed_layer, (n_qubits, mse_mean, _mse_std, mse_min, mse_max, _mae_mean, _mae_std, _n_samples) in sorted(
        quantum_stats.items()
    ):
        color = colors_quantum.get(fixed_layer, next(fallback, None))
        all_widths.update(n_qubits)
        yerr = asymmetric_yerr(mse_mean, mse_min, mse_max)
        ax.errorbar(n_qubits, mse_mean, yerr=yerr, fmt="o-", color=color, linewidth=2.0,
                    markersize=6, capsize=4, label=f"Quantum PINN ({fixed_layer})")

    if classical_data is not None:
        intra_neurons, mse_mean, _mse_std, mse_min, mse_max, _mae_mean, _mae_std, _n_samples = classical_data
        all_widths.update(intra_neurons)
        yerr = asymmetric_yerr(mse_mean, mse_min, mse_max)
        ax.errorbar(intra_neurons, mse_mean, yerr=yerr, fmt="s--", color=color_classical,
                    linewidth=2.0, markersize=6, capsize=4, label="Classical PINN")

    ax.set_xlabel("Width (N_QUBITS / INTRA_NEURO)", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks(sorted(all_widths))
    #ax.set_title(f"Quantum vs classical PINN mean MSE vs width (T = {Tfin})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
