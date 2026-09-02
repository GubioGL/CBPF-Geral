import math,os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from main.functions import (
    plot_loss,
    plot_solution,
    predict,
    reference_at_time,
    reference_pde_residual,
    sample_collocation_points,
    set_seed,
    solve_burgers_fd,
    relative_l2_error,
    train_autograd,)

from main.models_pennylane import QuantumPINNPennyLane
#from main.config import L, TFIN as Tfin, NU, HX, HT, N_COLLOCATION, SEED, EPOCHS, LR, PRE_NEURONS, PRE_DEPTH, POST_NEURONS, POST_DEPTH,SAVE_EXTRAS


# ---- physical parameters ---------------------------------------------------
L,  NU = 1.0 , 0.01
# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3
# ---- collocation sampling --------------------------------------------------
N_COLLOCATION = 100
N_COLLOCATION_LBFGS = N_COLLOCATION
COLLOCATION_DISTRIBUTION = "random"  # "uniform" ou "random"
if COLLOCATION_DISTRIBUTION not in ("uniform", "random"):
    raise ValueError(
        f"COLLOCATION_DISTRIBUTION must be 'uniform' or 'random', got {COLLOCATION_DISTRIBUTION!r}."
    )
# ---- training -------------------------------------------------------------
EPOCHS, LR = 20000, 1.0e-3
LBFGS_EPOCHS, LBFGS_LR = 1000, .5
LBFGS_PATIENCE, LBFGS_TOL = 100, 1.0e-9
# ---- architecture and training ---------------------------------------------
HIDDEN_NEURONS, HIDDEN_DEPTH = 10, 1
PRE_NEURONS, PRE_DEPTH = HIDDEN_NEURONS,HIDDEN_DEPTH
POST_NEURONS, POST_DEPTH =  HIDDEN_NEURONS,HIDDEN_DEPTH
SAVE_EXTRAS = True  # set True to also save the model checkpoint and loss plot

ANGLE_SCALE = math.pi
FIXED_LAYER = "brickwork_cnot"#"cz_ring" # ring_cnot , 
FIXED_LAYER_REPEATS = 1
INPUT_ROTATION_AXIS = "ry"
# N_QUBITS = 3  # estrutura=0 alimenta o quantum direto com o input bruto (sin,cos,tau) -> exige n_qubits=3
ESTRUTURAS = 1
Tfin = 1
# SEED = 52
#N_QUBITS =5

# Maximum number of seeds trained concurrently (one OS process each). Capped
# by the CPU count so we don't oversubscribe the machine.
MAX_WORKERS = min(5, os.cpu_count() or 1)


def run_one(FIXED_LAYER, N_QUBITS, SEED):
    """Train a single (FIXED_LAYER, N_QUBITS, SEED) combination end to end.

    Runs in its own process (see ProcessPoolExecutor below), so every value
    it needs is passed in or recomputed here rather than read from module
    globals set by a different process.
    """
    # Avoid CPU oversubscription: each worker process would otherwise try to
    # spawn its own torch thread pool, thrashing when several run at once.
    import torch as _torch
    _torch.set_num_threads(1)

    out_dir = f"data/quantum/T_{Tfin}"
    os.makedirs(out_dir, exist_ok=True)

    # mesmos pontos de colocacao para as 3 estruturas, comparacao justa
    x_f, t_f = sample_collocation_points(
        N_COLLOCATION, L, Tfin, HT,
        seed=SEED, distribution=COLLOCATION_DISTRIBUTION)

    ############################### Numerico #############################
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

    # Stopping target: the reference FDM solution's own discretization error,
    # i.e. the PDE residual noise floor of the baseline itself. Evaluated on a
    # uniform grid of collocation points regardless of COLLOCATION_DISTRIBUTION
    # -- unlike x_f, t_f (the actual training points, resampled randomly every
    # epoch when COLLOCATION_DISTRIBUTION="random"), this target must stay
    # fixed for the whole run, so it is computed once on its own independent
    # point set rather than tied to whatever points training happens to be
    # using at any given step.
    x_f_uniform, t_f_uniform = sample_collocation_points(
        N_COLLOCATION, L, Tfin, HT, distribution="uniform")
    target_loss = reference_pde_residual(
        x_f_uniform, t_f_uniform, x_ref, t_ref, U_ref, L=L, nu=NU, hx=HX, ht=HT)

    loss_histories = {}
    predictions = {}
    errors = {}
    set_seed(SEED)
    label = f"estrutura={ESTRUTURAS}"
    model = QuantumPINNPennyLane(
        n_qubits = N_QUBITS,
        pre_neurons = PRE_NEURONS,
        pre_depth = PRE_DEPTH,
        post_neurons=POST_NEURONS,
        post_depth=POST_DEPTH,
        angle_scale=ANGLE_SCALE,
        fixed_layer=FIXED_LAYER,
        fixed_layer_repeats=FIXED_LAYER_REPEATS,
        input_rotation_axis=INPUT_ROTATION_AXIS,
        seed=SEED,
        L=L,
        Tfin=Tfin,
        estrutura = ESTRUTURAS,
    )

    resample_fn = None
    lbfgs_resample_fn = None
    if COLLOCATION_DISTRIBUTION == "random":
        resample_fn = lambda: sample_collocation_points(
            N_COLLOCATION, L, Tfin, HT, distribution="random")
        lbfgs_resample_fn = lambda: sample_collocation_points(
            N_COLLOCATION_LBFGS, L, Tfin, HT, distribution="random")

    model, hist = train_autograd(
        model, x_f, t_f,
        loss_kwargs=dict(L=L, Tfin=Tfin, nu=NU, hx=HX, ht=HT),
        epochs=EPOCHS, lr=LR, resample_fn=resample_fn,
        lbfgs_epochs=LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
        lbfgs_resample_fn=lbfgs_resample_fn,
        lbfgs_patience=LBFGS_PATIENCE, lbfgs_tol=LBFGS_TOL,
        target_loss=target_loss,
    )
    loss_histories[label] = hist["loss"]

    u_pred = predict(model, x_plot, Tfin)
    predictions[label] = u_pred
    errors[label] = relative_l2_error(u_pred, u_ref)

    if SAVE_EXTRAS:
        model.save(
            f"{out_dir}/Qmodel_{COLLOCATION_DISTRIBUTION}_NQUBITS{N_QUBITS}_SEED{SEED}_{FIXED_LAYER}_{INPUT_ROTATION_AXIS}.pt",
            training_params=dict(
                NU=NU, HX=HX, HT=HT, N_COLLOCATION=N_COLLOCATION, SEED=SEED,
                COLLOCATION_DISTRIBUTION=COLLOCATION_DISTRIBUTION,
                EPOCHS=EPOCHS, LR=LR, LBFGS_EPOCHS=LBFGS_EPOCHS, LBFGS_LR=LBFGS_LR,
                HIDDEN_NEURONS=HIDDEN_NEURONS, HIDDEN_DEPTH=HIDDEN_DEPTH,
                TARGET_LOSS=target_loss, TARGET_REACHED=hist["target_reached"],
                STOPPED_AT_STEP=len(hist["loss"]),
            ),
        )

    ######################### PLOTS ###################################
    plot_loss(
        loss_histories,
        title=f"Quantum HPINN training loss por estrutura (PennyLane autograd, T={Tfin})",
        save_path=f"{out_dir}/Loss_{COLLOCATION_DISTRIBUTION}_NQUBITS{N_QUBITS}_SEED{SEED}_{FIXED_LAYER}_{INPUT_ROTATION_AXIS}.png",
    )

    plot_solution(
        x_plot, predictions, ref=u_ref,
        title=f"Quantum HPINN por estrutura vs FDM reference at T={Tfin}",
        save_path=f"{out_dir}/Solution_{COLLOCATION_DISTRIBUTION}_NQUBITS{N_QUBITS}_SEED{SEED}_{FIXED_LAYER}_{INPUT_ROTATION_AXIS}.png",
        show_cond=False,
    )

    return FIXED_LAYER, N_QUBITS, SEED, errors


if __name__ == "__main__":
    for N_QUBITS in [1,2,3,4,5]:

        # Launch all seeds for this (FIXED_LAYER, N_QUBITS) combo at once,
        # each in its own process, then wait for all of them to finish
        # before moving on to the next combo.
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_one, FIXED_LAYER, N_QUBITS, SEED): SEED
                for SEED in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    _, _, _, errors = future.result()
                except Exception:
                    print(f"--- T={Tfin} FIXED_LAYER={FIXED_LAYER} N_QUBITS={N_QUBITS} SEED={seed} FAILED ---")
                    raise
                print(f"--- T={Tfin} FIXED_LAYER={FIXED_LAYER} N_QUBITS={N_QUBITS} SEED={seed} ---")
                for label, err in errors.items():
                    print(f"Relative L2 error [{label}]: {err:.6e}")
