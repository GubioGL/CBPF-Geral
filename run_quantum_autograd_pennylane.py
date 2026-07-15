import math
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
from src.config import L, TFIN as Tfin, NU, HX, HT, N_COLLOCATION, SEED, EPOCHS, LR, PRE_NEURONS, PRE_DEPTH, POST_NEURONS, POST_DEPTH,SAVE_EXTRAS


if __name__ == "__main__":
    run_id =1

    # N_QUBITS = 1  # matches src/config.py's collocation/architecture sizing
    for N_QUBITS in range(10,15):  # 2, 3, 4 qubits

        ANGLE_SCALE = math.pi
        FIXED_LAYER = "cz_ring"
        FIXED_LAYER_REPEATS = 1
        INPUT_ROTATION_AXIS = "rx"

        set_seed(SEED)

        model = QuantumPINNPennyLane(
            n_qubits=N_QUBITS, pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
            post_neurons=POST_NEURONS, post_depth=POST_DEPTH, angle_scale=ANGLE_SCALE,
            fixed_layer=FIXED_LAYER, fixed_layer_repeats=FIXED_LAYER_REPEATS,
            input_rotation_axis=INPUT_ROTATION_AXIS, seed=SEED, L=L, Tfin=Tfin,
        )

        x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)

        model, hist = train_autograd(
            model, x_f, t_f,
            loss_kwargs=dict(L=L, Tfin=Tfin, nu=NU, hx=HX, ht=HT),
            epochs=EPOCHS, lr=LR,
        )

        if SAVE_EXTRAS:
            model.save(f"data/quantum/T_{Tfin}/model_NQUBITS{N_QUBITS}_p_{run_id}.pt")

            plot_loss({"Quantum HPINN (PennyLane autograd)": hist["loss"]},
                       title="Quantum HPINN training loss (PennyLane, autograd derivatives)",
                       save_path=f"data/quantum/T_{Tfin}/Loss_NQUBITS{N_QUBITS}_p_{run_id}.png")

############################### Numerico ###########################################
        x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
        x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

        u_pred = predict(model, x_plot, Tfin)
        u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

        ######################### PLOTS #####################################################
        plot_solution(
            x_plot, {"Quantum HPINN (PennyLane autograd)": u_pred}, ref=u_ref,
            title=f"Quantum HPINN (PennyLane autograd) vs FDM reference at T={Tfin}",
            save_path=f"data/quantum/T_{Tfin}/Solution_NQUBITS{N_QUBITS}_p_{run_id}.png",
            show_cond=False,
        )

        print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
