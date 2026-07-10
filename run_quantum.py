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
    train,)
from src.models import QuantumPINN
from src.config import L, TFIN as Tfin, NU, HX, HT, N_COLLOCATION, SEED, EPOCHS, LR, PRE_NEURONS, PRE_DEPTH, POST_NEURONS, POST_DEPTH


if __name__ == "__main__":
    # ---- quantum architecture -----------------------------------------------
    for N_QUBITS in range(5, 8):  # 2, 3, 4 qubits
        ANGLE_SCALE = math.pi
        FIXED_LAYER = "reverse_linear_cnot"          # one of FIXED_LAYERS, see models.py
        FIXED_LAYER_REPEATS = 1
        INPUT_ROTATION_AXIS = "rx"

        set_seed(SEED)

        model = QuantumPINN(
            n_qubits=N_QUBITS, pre_neurons=PRE_NEURONS, pre_depth=PRE_DEPTH,
            post_neurons=POST_NEURONS, post_depth=POST_DEPTH, angle_scale=ANGLE_SCALE,
            fixed_layer=FIXED_LAYER, fixed_layer_repeats=FIXED_LAYER_REPEATS,
            input_rotation_axis=INPUT_ROTATION_AXIS, seed=SEED, L=L, Tfin=Tfin,
        )
        
        # n_quantum_params = sum(p.numel() for p in model.quantum.parameters() if p.requires_grad)
        # print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
        # print("Trainable parameters inside the quantum circuit (should be 0):", n_quantum_params)

       # model.show_circuit().savefig(f"quantum_circuit_N_QUBITS{N_QUBITS}_noise.png", dpi=150)
        #print("Saved circuit diagram to quantum_circuit.png")

        x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)
        model, hist = train(
            model, x_f, t_f,
            loss_kwargs=dict(L=L, Tfin=Tfin, nu=NU, hx=HX, ht=HT),
            epochs=EPOCHS, lr=LR, print_every=50,
        )

        model.save(f"data/quantum/quantum_pinn_N_QUBITS{N_QUBITS}_noise.pt")
        #print("Saved model checkpoint to data/quantum/quantum_pinn.pt (config: data/quantum/quantum_pinn_config.json)")

        plot_loss({"Quantum HPINN": hist["loss"]}, title="Quantum HPINN training loss",
                save_path=f"data/quantum/quantum_loss_N_QUBITS{N_QUBITS}_noise.png")

        ############################### Numerico ###########################################
        x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
        x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

        u_pred = predict(model, x_plot, Tfin)
        u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
        
        ######################### PLOTS #####################################################
        plot_solution(
            x_plot, {"Quantum HPINN": u_pred}, ref=u_ref,
            title=f"Quantum HPINN vs FDM reference at T={Tfin}",
            save_path=f"data/quantum/quantum_solution_N_QUBITS{N_QUBITS}_noise.png",
            show_cond=False
        )

        # print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
