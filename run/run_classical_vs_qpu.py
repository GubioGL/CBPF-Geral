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

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")  # no window: figures are only saved
import matplotlib.pyplot as plt

import numpy as np
import torch as tc

from main.functions import (
    plot_loss, plot_solution, reference_at_time, relative_l2_error, sample_collocation_points,
    set_seed, solve_burgers_fd,
)
from main.models import ClassicalPINN2, QuantumExactPINN

ROOT = _ROOT
REAL_RUN = ROOT / "data" / "quantum_forward_hardware_real" / "T_0.1" / "run_10_NQ10_0_ibm_berlin_none_HW110.json"
OUT_DIR = ROOT / "data" / "Forward" / "classical_vs_qpu"
INTRA_NEURON_LIST = range(1, 11)   # width that plays the role of the number of qubits
SEEDS = range(5)                   # the seeds of the fake-QPU batch run (seed 0 = the real run)
INTRA_DEPTH, INTRA_NEURONS = 1, 10

# same protocol as the real-QPU run: everything comes from its log
real = json.loads(REAL_RUN.read_text())
cfg, arch = real["training_config"], real["architecture"]
N_COL, TFIN = real["n_collocation"], real["Tfin"]
L, NU, HX, HT = cfg["L"], cfg["nu"], cfg["hx"], cfg["ht"]
EPOCHS = cfg["epochs"]


def learning_rate(epoch):
    """LR of `epoch` (1-based), as run_quantum_hardware.learning_rate."""
    return max(cfg["lr0"] - cfg["lr_step"] * ((epoch - 1) // cfg["lr_every"]), cfg["lr_min"])


def stencil_inputs(x_f, t_f):
    """5 blocks of n rows (centre, x+hx, x-hx, t+ht, t-ht), as run_quantum_hardware."""
    x_p, x_m = tc.remainder(x_f + HX, L), tc.remainder(x_f - HX, L)
    return tc.cat([tc.cat([x_f, t_f], 1), tc.cat([x_p, t_f], 1), tc.cat([x_m, t_f], 1),
                   tc.cat([x_f, t_f + HT], 1), tc.cat([x_f, t_f - HT], 1)])


def residual_loss(model, xt_stencil):
    u_0, u_xp, u_xm, u_tp, u_tm = model(xt_stencil).chunk(5)
    u_t = (u_tp - u_tm) / (2.0 * HT)
    u_x = (u_xp - u_xm) / (2.0 * HX)
    u_xx = (u_xp - 2.0 * u_0 + u_xm) / HX**2
    return tc.mean((u_t + u_0 * u_x - NU * u_xx) ** 2)


def run_one(seed, intra_neuron, x_plot, xt_plot, u_ref, xt_mon, out_dir):
    set_seed(seed)                                  # same initialization order as build_model
    model = ClassicalPINN2(
        pre_neurons=arch["pre_neurons"], pre_depth=arch["pre_depth"],
        intra_neuron=intra_neuron, intra_neurons=INTRA_NEURONS, intra_depth=INTRA_DEPTH,
        post_neurons=arch["post_neurons"], post_depth=arch["post_depth"], L=L, Tfin=TFIN, seno=False)
    generator = tc.Generator().manual_seed(seed)    # same collocation points as the QPU run
    opt = tc.optim.AdamW(model.parameters(), lr=cfg["lr0"], weight_decay=cfg["weight_decay"])

    with tc.no_grad():
        hist = dict(lr=[], loss_train=[], loss_monitor=[float(residual_loss(model, xt_mon))], x_f=[], t_f=[])
    for epoch in range(1, EPOCHS + 1):
        lr = learning_rate(epoch)
        for group in opt.param_groups:
            group["lr"] = lr
        x_f = tc.rand((N_COL, 1), generator=generator) * L
        t_f = HT + tc.rand((N_COL, 1), generator=generator) * (TFIN - 2.0 * HT)
        loss = residual_loss(model, stencil_inputs(x_f, t_f))
        opt.zero_grad()
        loss.backward()
        tc.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        with tc.no_grad():
            loss_mon = float(residual_loss(model, xt_mon))
        for key, value in dict(lr=lr, loss_train=float(loss.detach()), loss_monitor=loss_mon,
                               x_f=x_f.numpy(), t_f=t_f.numpy()).items():
            hist[key].append(value)

    with tc.no_grad():
        u = model(xt_plot).numpy()                  # last epoch, as the QPU run

    tag = f"{N_COL}_NEURO{INTRA_DEPTH}{INTRA_NEURONS}{intra_neuron}_tanh_{seed}_E{EPOCHS}"
    path = lambda kind, ext: out_dir / f"{kind}_{tag}.{ext}"
    model.save(str(path("model", "pt")), training_params=dict(seed=seed, real_run=str(REAL_RUN)))
    np.savez(path("history", "npz"), **{k: np.array(v) for k, v in hist.items()})
    np.savez(path("validation", "npz"), x=x_plot, u=u, u_reference=u_ref)
    log = dict(
        seed=seed, intra_neuron=intra_neuron, intra_neurons=INTRA_NEURONS, intra_depth=INTRA_DEPTH,
        activation="tanh", n_trainable_params=sum(p.numel() for p in model.parameters()),
        architecture=arch, training_config=cfg,
        mse=float(np.mean((u - u_ref) ** 2)), mae=float(np.mean(np.abs(u - u_ref))),
        relative_l2=relative_l2_error(u, u_ref), final_loss_monitor=hist["loss_monitor"][-1],
        real_qpu_run=str(REAL_RUN), real_qpu_mse=real["mse"])
    path("run", "json").write_text(json.dumps(log, indent=2))
    plot_solution(x_plot, {f"classical (intra_neuron={intra_neuron})": u}, ref=u_ref,
                  save_path=path("validation", "png"), collocation_x=hist["x_f"][-1])
    plot_loss({"training batch": hist["loss_train"], "fixed grid": hist["loss_monitor"][1:]},
              title=f"Loss -- intra_neuron={intra_neuron}, seed {seed}, T={TFIN}", save_path=path("loss", "png"))
    plt.close("all")
    return log


def main():
    out_dir = OUT_DIR / f"T_{TFIN}"
    out_dir.mkdir(parents=True, exist_ok=True)
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=TFIN, nu=NU)
    x_plot = np.linspace(0.0, L, cfg["n_evaluation_points"]).reshape(-1, 1)
    u_ref, _ = reference_at_time(TFIN, t_ref, x_ref, U_ref, x_plot, L=L)
    xt_plot = tc.tensor(np.hstack([x_plot, np.full_like(x_plot, TFIN)]), dtype=tc.get_default_dtype())
    xt_mon = stencil_inputs(*sample_collocation_points(N_COL, L, TFIN, HT, distribution="uniform_exact"))
    runs = [run_one(seed, intra_neuron, x_plot, xt_plot, u_ref, xt_mon, out_dir)
            for intra_neuron in INTRA_NEURON_LIST for seed in SEEDS]

    # classical: every run + statistics over the seeds per intra_neuron; quantum: the real QPU run
    quantum = QuantumExactPINN(n_qubits=real["n_qubits"], pre_neurons=arch["pre_neurons"], pre_depth=arch["pre_depth"],
                               post_neurons=arch["post_neurons"], post_depth=arch["post_depth"],
                               fixed_layer=real["network"], fixed_layer_repeats=real["n_cnot_layers"])
    classical = []
    for n in INTRA_NEURON_LIST:
        group = [r for r in runs if r["intra_neuron"] == n]
        row = dict(intra_neuron=n, n_trainable_params=group[0]["n_trainable_params"])
        for metric in ("mse", "mae", "relative_l2"):
            values = [r[metric] for r in group]
            row.update({f"{metric}_mean": float(np.mean(values)), f"{metric}_std": float(np.std(values)),
                        f"{metric}_median": float(np.median(values)), f"{metric}_min": float(np.min(values)),
                        f"{metric}_max": float(np.max(values))})
        row["per_seed"] = {r["seed"]: dict(mse=r["mse"], mae=r["mae"], relative_l2=r["relative_l2"],
                                           final_loss_monitor=r["final_loss_monitor"]) for r in group}
        classical.append(row)
    summary = dict(
        seeds=list(SEEDS), epochs=EPOCHS, n_collocation=N_COL, Tfin=TFIN, classical=classical,
        quantum_real_qpu=dict(backend=real["backend"], seed=real["seed"], n_qubits=real["n_qubits"],
                              n_trainable_params=sum(p.numel() for p in quantum.parameters()),  # the circuit has none
                              mse=real["mse"], mae=real["mae"], relative_l2=real["relative_l2"],
                              final_loss_monitor=real["final_loss_monitor"]))
    (out_dir / f"summary_{N_COL}_NEURO{INTRA_DEPTH}{INTRA_NEURONS}_tanh_seeds{SEEDS[0]}-{SEEDS[-1]}_E{EPOCHS}.json"
     ).write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
