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
    train,
)
from src.models import ClassicalPINN
#from src.config import L, TFIN as Tfin, NU as nu, HX, HT, N_COLLOCATION, SEED, EPOCHS, LR, HIDDEN_NEURONS, HIDDEN_DEPTH,SAVE_EXTRAS


# ---- physical parameters ---------------------------------------------------
L,  NU = 1.0 , 0.01
# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3
# ---- collocation sampling --------------------------------------------------
N_COLLOCATION = 100
# ---- training -------------------------------------------------------------
EPOCHS, LR = 10000, 1.0e-2
# ---- architecture and training -------------------plot_solution-----------------------
HIDDEN_NEURONS, HIDDEN_DEPTH = 20, 1
PRE_NEURONS, PRE_DEPTH = HIDDEN_NEURONS,HIDDEN_DEPTH
POST_NEURONS, POST_DEPTH =  HIDDEN_NEURONS,HIDDEN_DEPTH
SAVE_EXTRAS = True  # set True to also save the model checkpoint and loss plot



if __name__ == "__main__":
    for Tfin in [0.1,1]:
        for run_id in range(1, 20):
        
        #run_id =42
        
            SEED = run_id
            set_seed(SEED)

            # Reference solution, used only for comparison/plots, never for training.
            x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
            x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

            model = ClassicalPINN(hidden_neurons=HIDDEN_NEURONS, hidden_depth=HIDDEN_DEPTH, L=L, Tfin=Tfin)

            x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)
            model, hist = train(
                model, x_f, t_f,
                loss_kwargs=dict(L=L, Tfin=Tfin, nu=NU, hx=HX, ht=HT),
                epochs=EPOCHS, lr=LR,
            )

            if SAVE_EXTRAS:
                model.save(
                    f"data/classical/T_{Tfin}/amostragem_{run_id}.pt",
                    training_params=dict(
                        NU=NU, HX=HX, HT=HT, N_COLLOCATION=N_COLLOCATION, SEED=SEED,
                        EPOCHS=EPOCHS, LR=LR, HIDDEN_NEURONS=HIDDEN_NEURONS, HIDDEN_DEPTH=HIDDEN_DEPTH,
                    ),
                )

            plot_loss({"Classical PINN": hist["loss"]}, title="Classical PINN training loss",
                        save_path=f"data/classical/T_{Tfin}/amostragem_{run_id}_loss.png"
                        )

            u_pred = predict(model, x_plot, Tfin)
            u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
            plot_solution(
                x_plot, {"Classical PINN": u_pred}, ref=u_ref,
                title=f"Classical PINN vs FDM reference at T={Tfin}",
                save_path=f"data/classical/T_{Tfin}/amostragem_{run_id}_solution.png",
                show_cond=False, collocation_x=x_f.numpy(),
            )

            print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
