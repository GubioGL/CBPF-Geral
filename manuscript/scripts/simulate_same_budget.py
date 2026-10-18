"""Noise ladder at the hardware budget: the HQPINN trained with EXACTLY the protocol of the
ibm_berlin run (110 AdamW epochs, 10 collocation points per epoch, same lr schedule, same
initialisation and point stream per seed), but with a simulated quantum layer:

  ideal  : <Z_i> = prod_j cos(phi_j)^M_ij on every stencil node (noiseless, infinite shots);
  shots  : the stencil centres get a binomial shot-noise sample of the ideal value with
           S = 1024 shots, the neighbours and the gradient use the ideal formula
           (the same straight-through estimator as on the QPU, with a perfect surrogate).

Cheap (CPU seconds, no Aer). Seeds 0-4 (seed 0 = the seed of the hardware runs).
Output: manuscript/cache/same_budget_sims.npz
"""
import numpy as np
import torch as tc
import torch.nn as nn

from paper_common import CACHE_DIR, HT, L, NU, rel_l2

import run_quantum_hardware as rqh  # noqa: E402
from main.functions import (  # noqa: E402
    cnot_cosine_expectations, reference_at_time, set_seed, solve_burgers_fd,
)

N, N_COL, TFIN, EPOCHS, N_EVAL, SHOTS = 10, 10, 0.1, 110, 200, 1024
SEEDS = range(5)


class ExactLayer(nn.Module):
    """Ideal quantum layer with the `offset` hook used by run_quantum_hardware.qpu_step."""

    def __init__(self, M):
        super().__init__()
        self.register_buffer("M", tc.as_tensor(M, dtype=tc.bool))
        self.offset = None

    def forward(self, angles):
        q = cnot_cosine_expectations(angles, self.M)
        return q if self.offset is None else q + self.offset


def train(seed, mode, xt_plot, u_ref):
    set_seed(seed)
    model = rqh.build_model(N, TFIN)
    model.quantum = ExactLayer(model.cnot_matrix)
    gen = tc.Generator().manual_seed(seed)
    rng = np.random.default_rng(10_000 + seed)
    opt = tc.optim.AdamW(model.parameters(), lr=rqh.LR0, weight_decay=rqh.WEIGHT_DECAY)

    def mse():
        with tc.no_grad():
            return float(np.mean((model(xt_plot).numpy() - u_ref) ** 2))

    curve = [mse()]
    for epoch in range(1, EPOCHS + 1):
        for group in opt.param_groups:
            group["lr"] = rqh.learning_rate(epoch)
        model.train()
        x_f = tc.rand((N_COL, 1), generator=gen) * L
        t_f = HT + tc.rand((N_COL, 1), generator=gen) * (TFIN - 2.0 * HT)
        xt = rqh.stencil_inputs(x_f, t_f)
        if mode == "ideal":
            loss = rqh.residual_loss(model, xt)
            opt.zero_grad()
            loss.backward()
            tc.nn.utils.clip_grad_norm_(model.parameters(), rqh.GRAD_CLIP)
            opt.step()
        else:
            with tc.no_grad():
                angles = model.angles(xt[:N_COL])
                y = model.quantum(angles).numpy().astype(np.float64)
            p0 = np.clip((1.0 + y) / 2.0, 0.0, 1.0)
            q_shot = 2.0 * rng.binomial(SHOTS, p0) / SHOTS - 1.0
            rqh.qpu_step(model, opt, xt, angles, q_shot)
        curve.append(mse())
    with tc.no_grad():
        u = model(xt_plot).numpy()
    return np.array(curve), u.ravel()


def main():
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=TFIN, nu=NU)
    x_plot = np.linspace(0.0, L, N_EVAL).reshape(-1, 1)
    u_ref, _ = reference_at_time(TFIN, t_ref, x_ref, U_ref, x_plot, L=L)
    xt_plot = tc.tensor(np.hstack([x_plot, np.full_like(x_plot, TFIN)]), dtype=tc.get_default_dtype())
    out = {}
    for mode in ("ideal", "shots"):
        curves, us = zip(*(train(s, mode, xt_plot, u_ref) for s in SEEDS))
        out[f"{mode}_mse"] = np.stack(curves)
        out[f"{mode}_u"] = np.stack(us)
        out[f"{mode}_rel_l2"] = np.array([rel_l2(u, u_ref) for u in us])
        print(f"{mode}: final MSE per seed " + ", ".join(f"{c[-1]:.2e}" for c in curves)
              + " | rel. L2 " + ", ".join(f"{100 * r:.1f}%" for r in out[f"{mode}_rel_l2"]))
    out.update(seeds=np.array(list(SEEDS)), x=x_plot.ravel(), u_reference=u_ref.ravel())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_DIR / "same_budget_sims.npz", **out)
    print(f"saved {CACHE_DIR / 'same_budget_sims.npz'}")


if __name__ == "__main__":
    main()
