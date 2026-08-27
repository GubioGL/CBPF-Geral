import numpy as np
from main.functions import (
    plot_loss,
    plot_solution,
    predict,
    reference_at_time,
    reference_pde_residual,
    relative_l2_error,
    sample_collocation_points,
    set_seed,
    solve_burgers_fd,
    train,)
from main.models import ClassicalPINN2
#from main.config import L, TFIN as Tfin, NU as nu, HX, HT, N_COLLOCATION, SEED, EPOCHS, LR, HIDDEN_NEURONS, HIDDEN_DEPTH,SAVE_EXTRAS

# ---- physical parameters ---------------------------------------------------
L,  NU = 1.0 , 0.01
# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3
# ---- collocation sampling --------------------------------------------------
#N_COLLOCATION = 100
COLLOCATION_DISTRIBUTION = "random"  # "uniform" or "random"
# ---- training -------------------------------------------------------------
EPOCHS, LR = 20000, 1.0e-3
LBFGS_EPOCHS, LBFGS_LR = 1000, .5
# ---- architecture and training -------------------plot_solution-----------------------
HIDDEN_NEURONS, HIDDEN_DEPTH = 10, 1
PRE_NEURONS, PRE_DEPTH = HIDDEN_NEURONS,HIDDEN_DEPTH
POST_NEURONS, POST_DEPTH =  HIDDEN_NEURONS,HIDDEN_DEPTH

INTRA_DEPTH = 1
INTRA_NEUROS = HIDDEN_NEURONS
INTRA_NEURO = 3

SAVE_EXTRAS = True  # set True to also save the model checkpoint and loss plot

if __name__ == "__main__":
    #for INTRA_NEURO in [1,2,3,4,5,6,7,8,9,10]:
    for N_COLLOCATION in [10,20,30,40, 50,60,70,80,90,100, 250,500]:
        for Tfin in [0.1,1]:
            for run_id in range(0,4):       
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
                    seed = SEED, distribution = COLLOCATION_DISTRIBUTION)

                resample_fn = None
                if COLLOCATION_DISTRIBUTION == "random":
                    resample_fn = lambda: sample_collocation_points(
                        N_COLLOCATION, L, Tfin, HT, distribution="random")

                # 1a condicao de parada: a loss do PINN so precisa ficar tao boa
                # quanto a propria referencia FD e' nesses mesmos pontos -- alem
                # disso o "ganho" e' apenas ruido do esquema de referencia.
                # 2a condicao (fallback): limite de epocas (EPOCHS + LBFGS_EPOCHS),
                # caso a meta acima nunca seja alcancada.
                target_loss = reference_pde_residual(
                    x_f, t_f, x_ref, t_ref, U_ref, L=L, nu=NU, hx=HX, ht=HT)

                model, hist = train(
                    model, x_f, t_f,
                    loss_kwargs = dict(L=L, Tfin=Tfin, nu=NU, hx=HX, ht=HT),
                    epochs = EPOCHS, lr = LR, resample_fn = resample_fn,
                    lbfgs_epochs = LBFGS_EPOCHS, lbfgs_lr = LBFGS_LR,
                    target_loss = target_loss)

                if SAVE_EXTRAS:
                    model.save(
                        f"data/classical/T_{Tfin}/Cmodel_{COLLOCATION_DISTRIBUTION}_{N_COLLOCATION}_NEURO{INTRA_DEPTH}{INTRA_NEUROS}{INTRA_NEURO}_{run_id}.pt",
                        training_params=dict(
                            NU = NU, HX = HX, HT = HT, N_COLLOCATION = N_COLLOCATION, SEED = SEED,
                            COLLOCATION_DISTRIBUTION = COLLOCATION_DISTRIBUTION,
                            EPOCHS = EPOCHS, LR = LR, LBFGS_EPOCHS = LBFGS_EPOCHS, LBFGS_LR = LBFGS_LR,
                            HIDDEN_NEURONS = HIDDEN_NEURONS, HIDDEN_DEPTH = HIDDEN_DEPTH,
                            TARGET_LOSS = target_loss, TARGET_REACHED = hist["target_reached"],
                            STOPPED_AT_STEP = len(hist["loss"]),
                        ),
                    )

                    # Historico completo da funcao custo (todos os passos, AdamW + L-BFGS),
                    # salvo a parte do checkpoint para nao inchar o json legivel do model.save.
                    np.savez(
                        f"data/classical/T_{Tfin}/losshist_{COLLOCATION_DISTRIBUTION}_{N_COLLOCATION}_CNEURO{INTRA_DEPTH}{INTRA_NEUROS}{INTRA_NEURO}_{run_id}.npz",
                        loss=np.array(hist["loss"]),
                        residual_rms=np.array(hist["residual_rms"]),
                        target_loss=target_loss,
                        target_reached=hist["target_reached"],
                    )

                plot_loss({"Classical PINN": hist["loss"]}, title="Classical PINN training loss",
                            save_path=f"data/classical/T_{Tfin}/loss_{COLLOCATION_DISTRIBUTION}_{N_COLLOCATION}_CNEURO{INTRA_DEPTH}{INTRA_NEUROS}{INTRA_NEURO}_{run_id}.png"
                            )

                u_pred = predict(model, x_plot, Tfin)
                u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
                plot_solution(
                    x_plot, {"Classical PINN": u_pred}, ref=u_ref,
                    title= f"Classical PINN vs FDM reference at T={Tfin}",
                    save_path = f"data/classical/T_{Tfin}/model_{COLLOCATION_DISTRIBUTION}_{N_COLLOCATION}_CNEURO{INTRA_DEPTH}{INTRA_NEUROS}{INTRA_NEURO}_{run_id}.png",
                    show_cond = False,
                    collocation_x = x_f.numpy(),
                )

                # print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
