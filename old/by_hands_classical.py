import math
import numpy as np
import torch as tc
import torch.nn as nn

from tqdm import tqdm
from src.functions import (
    sample_collocation_points,
    predict,reference_at_time,
    plot_solution,
    solve_burgers_fd
    )
from src.functions import set_seed

class ClassicalPINN(nn.Module):
    """Fully classical MLP ansatz for the 1D viscous Burgers equation."""

    def __init__(self, hidden_neurons=10, hidden_depth=2, L=1.0, Tfin=0.1):
        super().__init__()
        self.L, self.Tfin = float(L), float(Tfin)

        layers = [nn.Linear(3, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_depth - 1):
            layers += [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
        layers += [nn.Linear(hidden_neurons, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        # Modificando a entrada para usar funções trigonométricas, para impor periodicidade
        input,x = self.Input_angular(xt)
        
        # Rede neural para aproximar a solução da equação de Burgers
        output  = self.net(input)
        
        # Impor a condição inicial de forma dura, usando a saída da rede neural e a condição inicial
        return self.Hard_constrain(x,input[:, 2:3], output)
    
    def Input_angular(self, input_x):
        x, tempo = input_x[:, 0:1], input_x[:, 1:2]
        input = tc.cat(
            [tc.sin(2.0 * math.pi * x / self.L), 
             tc.cos(2.0 * math.pi * x / self.L), 
             tempo / self.Tfin],
            dim=1,
        )
        return input,x 
    
    def Hard_constrain(self, x, tempo, output_nn):
        cond_init = tc.sin(2.0 * math.pi * x / self.L)
        return cond_init + (tempo / self.Tfin) * output_nn


if __name__ == "__main__":
    # ---- physical parameters ----------------------------------------------
    L, Tfin, NU = 1.0, 0.1, 0.01
    HX, HT = L / 64.0, 1.0e-2

    # ---- architecture and training ------------------------------------------
    HIDDEN_NEURONS, HIDDEN_DEPTH = 10, 2
    N_COLLOCATION = 100
    EPOCHS, LR = 2000, 1.0e-2
    SEED = 42    
    
    
    model = ClassicalPINN(
        hidden_neurons = HIDDEN_NEURONS, 
        hidden_depth = HIDDEN_DEPTH, 
        L = L,
        Tfin = Tfin)
    
    # reproducibility
    set_seed(SEED)
    opt = tc.optim.Adam(model.parameters(), lr=LR)
    x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=SEED)
    x_p = tc.remainder(x_f + HX, L)
    x_m = tc.remainder(x_f - HX, L)
    t_p, t_m = t_f + HT, t_f - HT


    LOSS = []
    residual_rms = []

    for _ in tqdm(range(EPOCHS)):

        u_0 = model(tc.cat([x_f, t_f], dim=1))
        u_xp = model(tc.cat([x_p, t_f], dim=1))
        u_xm = model(tc.cat([x_m, t_f], dim=1))
        u_tp = model(tc.cat([x_f, t_p], dim=1))
        u_tm = model(tc.cat([x_f, t_m], dim=1))
        
        # Simplificação , colocar os pontos x_f, t_f, depois, logo em baixo 
        # x_p, t_f ....  até x_f, t_m. 
        # e rodar apenas uma vez, para calcular u_0, u_xp, u_xm, u_tp, u_tm.
        # O número total de pontos é 5 vezes N_COLLOCATION
        

        u_t = (u_tp - u_tm) / (2.0 * HT)
        u_x = (u_xp - u_xm) / (2.0 * HX)
        u_xx = (u_xp - 2.0 * u_0 + u_xm) / (HX**2)

        residual = u_t + u_0 * u_x - NU * u_xx
        loss = tc.mean(residual**2)
        
        opt.zero_grad()
        loss.backward()
        opt.step()


        LOSS.append(float(loss.detach()))
        residual_rms.append(float(tc.sqrt(tc.mean(residual.detach() ** 2))))

    # Reference solution, used only for comparison/plots, never for training.
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)        
    u_pred = predict(model, x_plot, Tfin)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    plot_solution(
        x_plot, {"Classical PINN": u_pred}, ref=u_ref,
        title=f"Classical PINN vs FDM reference at T={Tfin}",
        save_path="classical_solution.png",
    )

    # print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
