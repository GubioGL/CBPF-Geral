import math,os
import numpy as np

from src.functions import (
    plot_loss,
    plot_solution,
    predict,
    reference_at_time,
    sample_collocation_points,
    set_seed,
    solve_burgers_fd,
    relative_l2_error,
    train_autograd,)
from src.models_pennylane import QuantumPINNPennyLane
#from src.config import L, TFIN as Tfin, NU, HX, HT, N_COLLOCATION, SEED, EPOCHS, LR, PRE_NEURONS, PRE_DEPTH, POST_NEURONS, POST_DEPTH,SAVE_EXTRAS


# ---- physical parameters ---------------------------------------------------
L,  NU = 1.0 , 0.01
# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3
# ---- collocation sampling --------------------------------------------------
N_COLLOCATION = 300
# ---- training -------------------------------------------------------------
EPOCHS, LR = 10000, 1.0e-2
LBFGS_EPOCHS, LBFGS_LR = 5000, 1
# ---- architecture and training ---------------------------------------------
HIDDEN_NEURONS, HIDDEN_DEPTH = 20, 1
PRE_NEURONS, PRE_DEPTH = HIDDEN_NEURONS,HIDDEN_DEPTH
POST_NEURONS, POST_DEPTH =  HIDDEN_NEURONS,HIDDEN_DEPTH
SAVE_EXTRAS = True  # set True to also save the model checkpoint and loss plot

ANGLE_SCALE = math.pi
FIXED_LAYER = "cz_ring" # ring_cnot , 
FIXED_LAYER_REPEATS = 1
INPUT_ROTATION_AXIS = "rx"
N_QUBITS = 3  # estrutura=0 alimenta o quantum direto com o input bruto (sin,cos,tau) -> exige n_qubits=3

ESTRUTURAS = [0, 1, 2]

if __name__ == "__main__":
   
    for SEED in [0, 1,  2, 3]:
        for Tfin in [1, 0.1]:

            out_dir = f"data/quantum/T_{Tfin}"
            os.makedirs(out_dir, exist_ok=True)

            # mesmos pontos de colocacao para as 3 estruturas, comparacao justa
            x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)

            ############################### Numerico #############################
            x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
            x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
            u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

            loss_histories = {}
            predictions = {}
            errors = {}

            for estrutura in ESTRUTURAS:
                label = f"estrutura={estrutura}"

                set_seed(SEED)

                model = QuantumPINNPennyLane(
                    n_qubits=N_QUBITS, pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
                    post_neurons=POST_NEURONS, post_depth=POST_DEPTH, angle_scale=ANGLE_SCALE,
                    fixed_layer=FIXED_LAYER, fixed_layer_repeats=FIXED_LAYER_REPEATS,
                    input_rotation_axis=INPUT_ROTATION_AXIS, seed=SEED, L=L, Tfin=Tfin,
                    estrutura=estrutura,
                )

                model, hist = train_autograd(
                    model, x_f, t_f,
                    loss_kwargs=dict(L=L, Tfin=Tfin, nu=NU, hx=HX, ht=HT),
                    epochs=EPOCHS, lr=LR,
                    lbfgs_epochs=LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
                )
                loss_histories[label] = hist["loss"]

                u_pred = predict(model, x_plot, Tfin)
                predictions[label] = u_pred
                errors[label] = relative_l2_error(u_pred, u_ref)

                if SAVE_EXTRAS:
                    model.save(f"{out_dir}/model_NQUBITS{N_QUBITS}_estrutura{estrutura}.pt")

            ######################### PLOTS ###################################
            plot_loss(
                loss_histories,
                title=f"Quantum HPINN training loss por estrutura (PennyLane autograd, T={Tfin})",
                save_path=f"{out_dir}/Loss_NQUBITS{N_QUBITS}_comparativo_SEED{SEED}.png",
            )

            plot_solution(
                x_plot, predictions, ref=u_ref,
                title=f"Quantum HPINN por estrutura vs FDM reference at T={Tfin}",
                save_path=f"{out_dir}/Solution_NQUBITS{N_QUBITS}_comparativo_SEED{SEED}.png",
                show_cond=False,
            )

            print(f"--- T={Tfin} ---")
            for label, err in errors.items():
                print(f"Relative L2 error [{label}]: {err:.6e}")
