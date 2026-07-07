"""Configurações gerais compartilhadas entre run_classical.py e run_quantum.py.

Mantém física, discretização, amostragem de colocação e seed idênticas nos
dois scripts para que a comparação entre os modelos clássico e quântico seja
justa. Hiperparâmetros de arquitetura e de treinamento (EPOCHS, LR, etc.)
continuam definidos em cada script, pois são específicos de cada modelo.
"""

# ---- physical parameters ---------------------------------------------------
L, TFIN, NU = 1.0, 0.1, 0.01

# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 64.0, 1.0e-2

# ---- collocation sampling --------------------------------------------------
N_COLLOCATION = 100

# ---- reproducibility --------------------------------------------------------
SEED = 42

# ---- training -------------------------------------------------------------
EPOCHS, LR = 600, 1.0e-3


# ---- architecture and training -------------------plot_solution-----------------------
HIDDEN_NEURONS, HIDDEN_DEPTH = 20, 2

PRE_NEURONS, PRE_DEPTH = HIDDEN_NEURONS,1
POST_NEURONS, POST_DEPTH =  HIDDEN_NEURONS,1