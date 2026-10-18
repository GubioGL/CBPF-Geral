"""Classical controls at the hardware budget (110 AdamW epochs, 10 collocation points per epoch, T = 0.1).

Same loop as run_classical_vs_qpu.py (same initialisation, same point stream per seed, stencil
(1/512, 1e-3), weight decay 1e-6, clip 1); only the learning rate and the middle block change:

  1. LR sweep  : ClassicalPINN2 (tanh), widths n = 3, 10, seeds 0-4, initial LR eta0 in
                 {1e-3, 3e-3, 1e-2, 3e-2, 1e-1} with
                   step   -- the schedule of the QPU run scaled to eta0,
                             eta_k = max(eta0 - 0.1 eta0 floor((k-1)/10), 0.1 eta0)
                             (eta0 = 0.1 is the QPU schedule itself);
                   cosine -- eta_k = eta0/10 + (eta0 - eta0/10) (1 + cos(pi (k-1)/109)) / 2.
  2. sine block: ClassicalPINN2(seno=True), n = 10, on the same grid; the rows quoted in the paper
                 are the QPU schedule and the best tanh schedule for n = 10.

"Best" = lowest median relative L2 over the seeds, selected on the test error itself (the FDM
reference), which favours the classical model. Metrics: MSE and relative L2 of u(x, 0.1) against
the FDM reference on 200 points. Cheap (CPU, about a minute, no QPU or Aer job).

    C:/Users/Gubio/anaconda3/envs/qiskit/python.exe manuscript/scripts/controls_hardware_budget.py [--force]

Output: manuscript/cache/controls_hardware_budget.npz (reused unless --force); prints the summary.
"""
import json
import math
import sys

import numpy as np
import torch as tc

from paper_common import CACHE_DIR, CVQ_DIR, TRIVIAL_MSE, load_npz, mse, real_file, rel_l2

import run_classical_vs_qpu as cvq  # noqa: E402  (protocol of the QPU run, read from its log)
from main.functions import reference_at_time, set_seed, solve_burgers_fd  # noqa: E402
from main.models import ClassicalPINN2  # noqa: E402

CACHE = CACHE_DIR / "controls_hardware_budget.npz"
LRS = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
SCHEDULES = ("step", "cosine")
WIDTHS = (3, 10)
SEEDS = range(5)
SINE_WIDTH = 10
N_EVAL = cvq.cfg["n_evaluation_points"]


def learning_rate(schedule, eta0, epoch):
    if schedule == "step":
        return cvq.learning_rate(epoch) * (eta0 / cvq.cfg["lr0"])
    eta_min = eta0 / 10.0
    return eta_min + 0.5 * (eta0 - eta_min) * (1.0 + math.cos(math.pi * (epoch - 1) / (cvq.EPOCHS - 1)))


def train(seed, width, seno, schedule, eta0, xt_plot):
    """run_classical_vs_qpu.run_one without the file output; returns u(x, T) at the last epoch."""
    set_seed(seed)
    model = ClassicalPINN2(
        pre_neurons=cvq.arch["pre_neurons"], pre_depth=cvq.arch["pre_depth"],
        intra_neuron=width, intra_neurons=cvq.INTRA_NEURONS, intra_depth=cvq.INTRA_DEPTH,
        post_neurons=cvq.arch["post_neurons"], post_depth=cvq.arch["post_depth"], L=cvq.L, Tfin=cvq.TFIN,
        seno=seno)
    generator = tc.Generator().manual_seed(seed)
    opt = tc.optim.AdamW(model.parameters(), lr=eta0, weight_decay=cvq.cfg["weight_decay"])
    for epoch in range(1, cvq.EPOCHS + 1):
        for group in opt.param_groups:
            group["lr"] = learning_rate(schedule, eta0, epoch)
        x_f = tc.rand((cvq.N_COL, 1), generator=generator) * cvq.L
        t_f = cvq.HT + tc.rand((cvq.N_COL, 1), generator=generator) * (cvq.TFIN - 2.0 * cvq.HT)
        loss = cvq.residual_loss(model, cvq.stencil_inputs(x_f, t_f))
        opt.zero_grad()
        loss.backward()
        tc.nn.utils.clip_grad_norm_(model.parameters(), cvq.cfg["grad_clip"])
        opt.step()
    with tc.no_grad():
        u = model(xt_plot).numpy().ravel()
    return u, sum(p.numel() for p in model.parameters())


def compute():
    x_ref, t_ref, U_ref = solve_burgers_fd(L=cvq.L, Tfin=cvq.TFIN, nu=cvq.NU)
    x_plot = np.linspace(0.0, cvq.L, N_EVAL).reshape(-1, 1)
    u_ref, _ = reference_at_time(cvq.TFIN, t_ref, x_ref, U_ref, x_plot, L=cvq.L)
    u_ref = np.ravel(u_ref)
    xt_plot = tc.tensor(np.hstack([x_plot, np.full_like(x_plot, cvq.TFIN)]), dtype=tc.get_default_dtype())

    shape = (len(SCHEDULES), len(LRS), len(SEEDS))
    tanh_u = np.zeros((len(SCHEDULES), len(LRS), len(WIDTHS), len(SEEDS), N_EVAL))
    sine_u = np.zeros(shape + (N_EVAL,))
    n_params = {}
    for i, schedule in enumerate(SCHEDULES):
        for j, eta0 in enumerate(LRS):
            for k, width in enumerate(WIDTHS):
                for s, seed in enumerate(SEEDS):
                    tanh_u[i, j, k, s], n_params[("tanh", width)] = train(seed, width, False, schedule, eta0, xt_plot)
            for s, seed in enumerate(SEEDS):
                sine_u[i, j, s], n_params[("sine", SINE_WIDTH)] = train(seed, SINE_WIDTH, True, schedule, eta0, xt_plot)
            print(f"  {schedule:6s} eta0={eta0:.0e} done", flush=True)

    metrics = lambda U: (np.apply_along_axis(mse, -1, U, u_ref), np.apply_along_axis(rel_l2, -1, U, u_ref))
    tanh_mse, tanh_rel = metrics(tanh_u)
    sine_mse, sine_rel = metrics(sine_u)
    out = dict(
        lrs=LRS, schedules=np.array(SCHEDULES), widths=np.array(WIDTHS), seeds=np.array(list(SEEDS)),
        sine_width=SINE_WIDTH, x=x_plot.ravel(), u_reference=u_ref,
        tanh_u=tanh_u, tanh_mse=tanh_mse, tanh_rel_l2=tanh_rel,   # [schedule, lr, width, seed]
        sine_u=sine_u, sine_mse=sine_mse, sine_rel_l2=sine_rel,   # [schedule, lr, seed]
        tanh_n_params=np.array([n_params[("tanh", w)] for w in WIDTHS]),
        sine_n_params=n_params[("sine", SINE_WIDTH)],
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    print(f"saved {CACHE}")
    return out


def best_index(rel):
    """(schedule, lr) with the lowest median over the last axis (seeds)."""
    return np.unravel_index(np.argmin(np.median(rel, axis=-1)), rel.shape[:2])


def fmt(mse_values, rel_values):
    r = 100 * np.asarray(rel_values)
    above = np.mean(np.asarray(mse_values) > TRIVIAL_MSE[0.1])
    return (f"{np.median(mse_values):9.2e}  {np.median(r):5.1f}%  ({r.min():4.1f}-{r.max():4.1f}%)"
            f"  {100 * above:3.0f}%")


def check_reproduction(d):
    """step schedule at eta0 = 0.1 is the run of run_classical_vs_qpu.py: compare with its logs."""
    i, j = SCHEDULES.index("step"), int(np.argmin(np.abs(d["lrs"] - cvq.cfg["lr0"])))
    diffs = []
    for k, width in enumerate(WIDTHS):
        for s, seed in enumerate(SEEDS):
            log = json.loads((CVQ_DIR / f"run_{cvq.N_COL}_NEURO{cvq.INTRA_DEPTH}{cvq.INTRA_NEURONS}{width}"
                                        f"_tanh_{seed}_E{cvq.EPOCHS}.json").read_text())
            diffs.append(abs(log["relative_l2"] - d["tanh_rel_l2"][i, j, k, s]))
    print(f"reproduction of data/Forward/classical_vs_qpu (step, eta0 = 0.1, n = {WIDTHS}, seeds 0-4): "
          f"max |delta rel. L2| = {max(diffs):.1e}")


def summary(d):
    head = f"{'schedule':8s} {'eta0':>6s}  {'MSE med':>9s}  {'eps_L2 med (range)':>21s}  >triv"
    print("\nrelative L2 and MSE of u(x, 0.1) vs FDM, 200 points; medians over seeds 0-4;"
          " >triv = runs with MSE above the no-evolution level u = sin(2 pi x)")
    for k, width in enumerate(d["widths"]):
        print(f"\n[tanh middle block, n = {width}, {d['tanh_n_params'][k]} parameters]\n{head}")
        rel, err = d["tanh_rel_l2"][:, :, k], d["tanh_mse"][:, :, k]
        for i, schedule in enumerate(d["schedules"]):
            for j, eta0 in enumerate(d["lrs"]):
                print(f"{schedule:8s} {eta0:6.0e}  {fmt(err[i, j], rel[i, j])}")
        i, j = best_index(rel)
        print(f"best: {d['schedules'][i]} eta0={d['lrs'][j]:.0e} -> {fmt(err[i, j], rel[i, j])}"
              f"  per seed: {', '.join(f'{100 * v:.1f}' for v in rel[i, j])}%")

    k10 = list(d["widths"]).index(d["sine_width"])
    bi, bj = best_index(d["tanh_rel_l2"][:, :, k10])
    qi, qj = list(d["schedules"]).index("step"), int(np.argmin(np.abs(d["lrs"] - 0.1)))
    print(f"\n[sine middle block (seno=True), n = {d['sine_width']}, {d['sine_n_params']} parameters]\n{head}")
    for i, schedule in enumerate(d["schedules"]):
        for j, eta0 in enumerate(d["lrs"]):
            tag = "  <- QPU schedule" if (i, j) == (qi, qj) else ""
            tag += "  <- best tanh n=10" if (i, j) == (bi, bj) else ""
            print(f"{schedule:8s} {eta0:6.0e}  {fmt(d['sine_mse'][i, j], d['sine_rel_l2'][i, j])}{tag}")
    for name, (i, j) in (("QPU schedule", (qi, qj)), ("best tanh n=10 schedule", (bi, bj))):
        print(f"sine, {name}: per seed {', '.join(f'{100 * v:.1f}' for v in d['sine_rel_l2'][i, j])}%")
    i, j = best_index(d["sine_rel_l2"])
    print(f"best sine schedule: {d['schedules'][i]} eta0={d['lrs'][j]:.0e} -> "
          f"{fmt(d['sine_mse'][i, j], d['sine_rel_l2'][i, j])}")

    sims, real = load_npz(CACHE_DIR / "same_budget_sims.npz"), json.loads(real_file("run", "json").read_text())
    print(f"\nfor comparison: HQPINN ideal circuit, QPU schedule, seeds 0-4: median "
          f"{100 * np.median(sims['ideal_rel_l2']):.1f}% ({', '.join(f'{100 * v:.1f}' for v in sims['ideal_rel_l2'])}%);"
          f" ibm_berlin run (seed 0): {100 * real['relative_l2']:.1f}%")


def main():
    d = load_npz(CACHE) if CACHE.exists() and "--force" not in sys.argv else compute()
    check_reproduction(d)
    summary(d)


if __name__ == "__main__":
    main()
