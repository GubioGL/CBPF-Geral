import numpy as np
from src.functions import (
    sample_collocation_points,
    set_seed,)
#from src.config import L, TFIN as Tfin, HT, N_COLLOCATION, SEED


if __name__ == "__main__":

    N_COLLOCATION =10
    HT = 1.0e-3
    L = 1.0
    Tfin = 1.0

    x_f, t_f = sample_collocation_points(N_COLLOCATION, L, Tfin, HT, seed=0)
    print(f"Collocation points shape: {x_f.shape}, {t_f.shape}")

    print(x_f)
    print(t_f)