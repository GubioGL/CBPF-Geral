"""Offline replay of the 110-epoch trainings -> per-epoch error of u(x, T) against the FDM.

No QPU and no Aer job is run. The hardware trainings are deterministic given what was
recorded: the initial weights (params_initial_*.pt), the calibration job (calibration_*.npz)
and, for every epoch, the collocation points and the <Z> measured at the stencil centres
(history_*.npz). Feeding the recorded measurements back into the same AdamW step
(run_quantum_hardware.qpu_step) and the same polynomial refits reproduces the weight
trajectory; the script checks this against the saved final model. At every epoch the
network is evaluated on the 200-point grid at t = T with
  * the calibrated surrogate in force at that epoch (what the training sees), and
  * the ideal circuit <Z_i> = prod_j cos(phi_j)^M_ij (same weights, no noise).
The QPU itself only evaluated the final network (validation job), so the QPU-evaluated
error exists for the last epoch only.

The classical control (run_classical_vs_qpu.py, same protocol, widths 1..10, seeds 0-4) is
re-run in the same way (it is deterministic) and checked against its saved MSE.

Output: manuscript/cache/learning_curves.npz
"""
import time

import numpy as np
import torch as tc

from paper_common import (CACHE_DIR, CVQ_DIR, HT, HX, L, NU, fake_file, load_json, load_npz,
                          real_file)

import run_quantum_hardware as rqh  # noqa: E402  (model, loss, AdamW step, lr schedule)
from Inverser_quantum_hardware_pinn import (  # noqa: E402
    HardwareSurrogateLayer, causal_cones, cnot_gate_list, fit_surrogate, point_grid,
)
from main.functions import (  # noqa: E402
    cnot_cosine_expectations, reference_at_time, repeated_cnot_network_matrix,
    sample_collocation_points, solve_burgers_fd,
)
from main.models import ClassicalPINN2  # noqa: E402

N, N_COL, TFIN, EPOCHS, N_EVAL = 10, 10, 0.1, 110, 200


def reference():
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=TFIN, nu=NU)
    x_plot = np.linspace(0.0, L, N_EVAL).reshape(-1, 1)
    u_ref, _ = reference_at_time(TFIN, t_ref, x_ref, U_ref, x_plot, L=L)
    xt_plot = tc.tensor(np.hstack([x_plot, np.full_like(x_plot, TFIN)]), dtype=tc.get_default_dtype())
    return x_plot, u_ref, xt_plot


def replay_quantum(kind):
    """kind = 'real' (ibm_berlin) or 'fake' (FakeBerlin)."""
    f = real_file if kind == "real" else fake_file
    hist, cal = load_npz(f("history")), load_npz(f("calibration"))
    init = tc.load(f("params_initial", "pt"), weights_only=False)
    final = tc.load(f("model", "pt"), weights_only=False)
    B = int(cal["chains"].shape[0])

    gates = cnot_gate_list(N, rqh.FIXED_LAYER, rqh.FIXED_LAYER_REPEATS)
    M = repeated_cnot_network_matrix(N, rqh.FIXED_LAYER, rqh.FIXED_LAYER_REPEATS)
    cones = causal_cones(N, gates)
    x_plot, u_ref, xt_plot = reference()
    xt_mon = rqh.stencil_inputs(*sample_collocation_points(N_COL, L, TFIN, HT, distribution="uniform_exact"))

    model = rqh.build_model(N, TFIN)
    model.load_state_dict(init["state_dict"])
    cal_angles, cal_evs = cal["angles"], cal["evs"]
    coefs, _, _ = fit_surrogate(cal_angles, cal_evs, cones, M)
    model.quantum = HardwareSurrogateLayer(cones, coefs)
    opt = tc.optim.AdamW(model.parameters(), lr=rqh.LR0, weight_decay=rqh.WEIGHT_DECAY)

    def evaluate():
        model.eval()
        with tc.no_grad():
            angles = model.angles(xt_plot)
            u_sur = model(xt_plot).numpy()
            u_id = model.from_features(xt_plot, cnot_cosine_expectations(angles, model.cnot_matrix)).numpy()
        return u_sur, u_id

    out = dict(mse_surrogate=[], mse_ideal=[], loss_qpu=[], loss_monitor=[], rms_qpu_vs_poly=[],
               angle_mismatch=[], u_surrogate=[], u_ideal=[])
    u_sur, u_id = evaluate()
    out["u_surrogate"].append(u_sur.ravel())
    out["u_ideal"].append(u_id.ravel())
    out["mse_surrogate"].append(np.mean((u_sur - u_ref) ** 2))
    out["mse_ideal"].append(np.mean((u_id - u_ref) ** 2))
    acc_angles, acc_evs = [], []
    for epoch in range(1, EPOCHS + 1):
        for group in opt.param_groups:
            group["lr"] = rqh.learning_rate(epoch)
        model.train()
        x_f = tc.as_tensor(hist["x_f"][epoch - 1])
        t_f = tc.as_tensor(hist["t_f"][epoch - 1])
        xt = rqh.stencil_inputs(x_f, t_f)
        with tc.no_grad():
            angles = model.angles(xt[:N_COL])
        # the history stores the measured slots in (parameter set, chain) order: undo it
        grid = point_grid(N_COL, B, 1, shift=epoch)
        order = grid[grid >= 0]
        q_qpu = np.empty((N_COL, N))
        q_qpu[order] = hist["evs"][epoch - 1]
        meas_angles = np.empty((N_COL, N), dtype=np.float32)
        meas_angles[order] = hist["angles"][epoch - 1]
        out["angle_mismatch"].append(float(np.max(np.abs(meas_angles - angles.numpy()))))
        with tc.no_grad():
            q_poly = model.quantum(angles).numpy()
        out["rms_qpu_vs_poly"].append(float(np.sqrt(np.mean((q_qpu - q_poly) ** 2))))
        out["loss_qpu"].append(rqh.qpu_step(model, opt, xt, angles, q_qpu))
        acc_angles.append(hist["angles"][epoch - 1])
        acc_evs.append(hist["evs"][epoch - 1])
        if rqh.RECAL_EVERY and epoch % rqh.RECAL_EVERY == 0:
            coefs, _, _ = fit_surrogate(np.concatenate([cal_angles, *acc_angles]),
                                        np.concatenate([cal_evs, *acc_evs]), cones, M)
            model.quantum.set_coefs(coefs)
        with tc.no_grad():
            out["loss_monitor"].append(float(rqh.residual_loss(model, xt_mon)))
        u_sur, u_id = evaluate()
        out["u_surrogate"].append(u_sur.ravel())
        out["u_ideal"].append(u_id.ravel())
        out["mse_surrogate"].append(np.mean((u_sur - u_ref) ** 2))
        out["mse_ideal"].append(np.mean((u_id - u_ref) ** 2))

    state = model.state_dict()
    weight_err = max(float((state[k] - v).abs().max()) for k, v in final["state_dict"].items()
                     if k in state and state[k].dtype.is_floating_point)
    loss_err = float(np.max(np.abs(np.array(out["loss_qpu"]) - hist["loss_qpu"]) / np.abs(hist["loss_qpu"])))
    print(f"{kind}: max |weights replay - saved| = {weight_err:.2e} | max rel. loss diff = {loss_err:.2e} | "
          f"max angle mismatch = {max(out['angle_mismatch']):.2e} | final MSE surrogate "
          f"{out['mse_surrogate'][-1]:.3e}, ideal {out['mse_ideal'][-1]:.3e}")
    res = {f"{kind}_{k}": np.array(v) for k, v in out.items()}
    res[f"{kind}_weight_err"] = np.array(weight_err)
    res[f"{kind}_loss_err"] = np.array(loss_err)
    return res, x_plot, u_ref


def replay_classical():
    """run_classical_vs_qpu.run_one, with u(x, T) evaluated after every epoch."""
    real = load_json(real_file("run", "json"))
    cfg, arch = real["training_config"], real["architecture"]
    x_plot, u_ref, xt_plot = reference()
    widths, seeds = range(1, 11), range(5)
    curves = np.zeros((len(widths), len(seeds), EPOCHS + 1))
    loss = np.zeros((len(widths), len(seeds), EPOCHS))
    final_err = 0.0
    for a, width in enumerate(widths):
        for b, seed in enumerate(seeds):
            tc.manual_seed(seed)
            np.random.seed(seed)
            model = ClassicalPINN2(pre_neurons=arch["pre_neurons"], pre_depth=arch["pre_depth"],
                                   intra_neuron=width, intra_neurons=10, intra_depth=1,
                                   post_neurons=arch["post_neurons"], post_depth=arch["post_depth"],
                                   L=L, Tfin=TFIN, seno=False)
            gen = tc.Generator().manual_seed(seed)
            opt = tc.optim.AdamW(model.parameters(), lr=cfg["lr0"], weight_decay=cfg["weight_decay"])

            def err():
                with tc.no_grad():
                    return float(np.mean((model(xt_plot).numpy() - u_ref) ** 2))

            curves[a, b, 0] = err()
            for epoch in range(1, EPOCHS + 1):
                for group in opt.param_groups:
                    group["lr"] = rqh.learning_rate(epoch)
                x_f = tc.rand((N_COL, 1), generator=gen) * L
                t_f = HT + tc.rand((N_COL, 1), generator=gen) * (TFIN - 2.0 * HT)
                lval = rqh.residual_loss(model, rqh.stencil_inputs(x_f, t_f))
                opt.zero_grad()
                lval.backward()
                tc.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
                loss[a, b, epoch - 1] = float(lval.detach())
                curves[a, b, epoch] = err()
            saved = load_json(CVQ_DIR / f"run_10_NEURO110{width}_tanh_{seed}_E110.json")["mse"]
            final_err = max(final_err, abs(curves[a, b, -1] - saved) / saved)
    print(f"classical: max rel. |final MSE replay - saved| = {final_err:.2e}")
    return dict(classical_mse=curves, classical_loss=loss, classical_widths=np.array(list(widths)),
                classical_seeds=np.array(list(seeds)), classical_final_err=np.array(final_err))


def main():
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind in ("real", "fake"):
        res, x_plot, u_ref = replay_quantum(kind)
        out.update(res)
    out.update(replay_classical())
    out.update(x=x_plot.ravel(), u_reference=u_ref.ravel(), epochs=np.arange(EPOCHS + 1))
    np.savez(CACHE_DIR / "learning_curves.npz", **out)
    print(f"saved {CACHE_DIR / 'learning_curves.npz'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
