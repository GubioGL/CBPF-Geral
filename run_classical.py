import numpy as np

from src.functions import (
    plot_loss,
    plot_solution,
    predict,
    reference_at_time,
    relative_l2_error,
    sample_collocation_points,
    set_seed,
    solve_burgers_fd,
    train,)
from src.models import ClassicalPINN2
#from src.config import L, TFIN as Tfin, NU as nu, HX, HT, N_COLLOCATION, SEED, EPOCHS, LR, HIDDEN_NEURONS, HIDDEN_DEPTH,SAVE_EXTRAS


# ---- physical parameters ---------------------------------------------------
L,  NU = 1.0 , 0.01
# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3
# ---- collocation sampling --------------------------------------------------
N_COLLOCATION = 100
# ---- training -------------------------------------------------------------
EPOCHS, LR = 5000, 1.0e-2
# ---- architecture and training -------------------plot_solution-----------------------
HIDDEN_NEURONS, HIDDEN_DEPTH = 50, 1
PRE_NEURONS, PRE_DEPTH = HIDDEN_NEURONS,HIDDEN_DEPTH
POST_NEURONS, POST_DEPTH =  HIDDEN_NEURONS,HIDDEN_DEPTH

INTRA_DEPTH = 1
INTRA_NEUROS = HIDDEN_NEURONS
# INTRA_NEURO = 5

SAVE_EXTRAS = True  # set True to also save the model checkpoint and loss plot

if __name__ == "__main__":
    for INTRA_NEURO in [3,10]:
        for Tfin in [0.1,1]:
            for run_id in range(1, 6):       
                SEED = run_id
                set_seed(SEED)

                # Reference solution, used only for comparison/plots, never for training.
                x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
                x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

                model = ClassicalPINN2(
                    pre_neurons = PRE_NEURONS,
                    pre_depth = PRE_DEPTH,
                    intra_neuron = INTRA_NEURO,
                    intra_neurons = INTRA_NEUROS,
                    intra_depth = INTRA_DEPTH,
                    post_neurons = POST_NEURONS, 
                    post_depth = POST_DEPTH,
                    L = L,
                    Tfin = Tfin)
                x_f, t_f = sample_collocation_points(
                    N_COLLOCATION, L, Tfin, HT, 
                    seed = SEED)
                model, hist = train(
                    model, x_f, t_f,
                    loss_kwargs = dict(L=L, Tfin=Tfin, nu=NU, hx=HX, ht=HT),
                    epochs = EPOCHS, lr = LR)

                if SAVE_EXTRAS:
                    model.save(
                        f"data/classical/T_{Tfin}/Cmodel_NEURO{INTRA_DEPTH}{INTRA_NEUROS}{INTRA_NEURO}_{run_id}.pt",
                        training_params=dict(
                            NU = NU, HX = HX, HT = HT, N_COLLOCATION = N_COLLOCATION, SEED = SEED,
                            EPOCHS = EPOCHS, LR = LR, HIDDEN_NEURONS = HIDDEN_NEURONS, HIDDEN_DEPTH = HIDDEN_DEPTH,
                        ),
                    )

                plot_loss({"Classical PINN": hist["loss"]}, title="Classical PINN training loss",
                            save_path=f"data/classical/T_{Tfin}/loss_CNEURO{INTRA_DEPTH}{INTRA_NEUROS}{INTRA_NEURO}_{run_id}.png"
                            )

                u_pred = predict(model, x_plot, Tfin)
                u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
                plot_solution(
                    x_plot, {"Classical PINN": u_pred}, ref=u_ref,
                    title= f"Classical PINN vs FDM reference at T={Tfin}",
                    save_path = f"data/classical/T_{Tfin}/model_CNEURO{INTRA_DEPTH}{INTRA_NEUROS}{INTRA_NEURO}_{run_id}.png",
                    show_cond = False,
                    collocation_x = x_f.numpy(),
                )

                print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
