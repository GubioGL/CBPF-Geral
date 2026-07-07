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
    """x in [0,L), t in [ht, T-ht] (interior points, needed for centered differences)."""
    if T <= 2.0 * ht:
        raise ValueError("T must satisfy T > 2*ht.")
    gen = None
    if seed is not None:
        gen = tc.Generator(device="cpu")
        gen.manual_seed(seed)
    x = L * tc.rand((n_points, 1), generator=gen)
    tt = ht + (T - 2.0 * ht) * tc.rand((n_points, 1), generator=gen)
    return x, tt


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------
def train(model, x_f, t_f, loss_kwargs, epochs=400, lr=2.0e-4, grad_clip=1.0,
          weight_decay=1.0e-6, print_every=50):
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
    t0 = time.time()

    for ep in tqdm(range(epochs)):
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

        #if print_every and (ep == 0 or (ep + 1) % print_every == 0 or ep == epochs - 1):
        #    print(f"epoch {ep+1:5d}/{epochs} | loss={loss_value:.6e} | "
        #          f"r_rms={hist['residual_rms'][-1]:.3e} | time={time.time()-t0:.1f}s")

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
        plt.plot(np.log10(np.asarray(values) + 1.0e-16), linewidth=2.0, label=label)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(r"$\log_{10}(\mathrm{loss})$", fontsize=14)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show_cond==True:
        plt.show()


def plot_solution(x, curves, ref=None, ref_label="FDM reference", title="", save_path=None, show_cond=False):
    """
    x: spatial grid, shape (N,1).
    curves: dict of label -> predicted u array, shape (N,1). Used both for a
        single model's solution and for multi-model / multi-time comparisons.
    ref: optional reference array plotted as a thick solid line.
    """
    plt.figure(figsize=(9, 4.5))
    if ref is not None:
        plt.plot(x.ravel(), ref.ravel(), linewidth=3.0, label=ref_label)
    for label, u in curves.items():
        plt.plot(x.ravel(), u.ravel(), "--", linewidth=2.0, label=label)
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$u(x,t)$", fontsize=14)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show_cond==True:
        plt.show()
