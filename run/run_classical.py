# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/Forward/..." are resolved from the repository root.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _ROOT / "src" / "quantum_hardware", _HERE):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch as tc
from tqdm import tqdm
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

SAVE_EXTRAS = True  # set True to also save the model checkpoint and loss plot

# ---- parallelization --------------------------------------------------------
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)


def run_one(N_COLLOCATION, intra_neuro, Tfin, run_id):
    tc.set_num_threads(1)

    SEED = run_id
    set_seed(SEED)

    # Reference solution, used only for comparison/plots, never for training.
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)

    model = ClassicalPINN2(
        pre_neurons = PRE_NEURONS,
        pre_depth = PRE_DEPTH,
        intra_neuron = intra_neuro,
        intra_neurons = INTRA_NEUROS,
        intra_depth = INTRA_DEPTH,
        post_neurons = POST_NEURONS,
        post_depth = POST_DEPTH,
        L = L,
        Tfin = Tfin,
        seno=True)
    x_f, t_f = sample_collocation_points(
        N_COLLOCATION, L, Tfin, HT,
        seed = SEED, distribution = COLLOCATION_DISTRIBUTION)

    resample_fn = None
    if COLLOCATION_DISTRIBUTION == "random":
        resample_fn = lambda: sample_collocation_points(
            N_COLLOCATION, L, Tfin, HT, distribution="random")

    STOPPING_DISTRIBUTION = "uniform_exact"
    x_val, t_val = sample_collocation_points(
        N_COLLOCATION, L, Tfin, HT, distribution=STOPPING_DISTRIBUTION)

    # 1a condicao de parada: a loss do PINN so precisa ficar tao boa
    # quanto a propria referencia FD e' nesses mesmos pontos -- alem
    # disso o "ganho" e' apenas ruido do esquema de referencia.
    # 2a condicao (fallback): limite de epocas (EPOCHS + LBFGS_EPOCHS),
    # caso a meta acima nunca seja alcancada.
    target_loss = reference_pde_residual(
        x_val, t_val, x_ref, t_ref, U_ref, L=L, nu=NU, hx=HX, ht=HT)

    model, hist = train(
        model, x_f, t_f,
        loss_kwargs = dict(L=L, Tfin=Tfin, nu=NU, hx=HX, ht=HT),
        epochs = EPOCHS, lr = LR, resample_fn = resample_fn,
        lbfgs_epochs = LBFGS_EPOCHS, lbfgs_lr = LBFGS_LR,
        target_loss = target_loss,
        validation_points = (x_val, t_val))

    if SAVE_EXTRAS:
        model.save(
            f"data/Forward/classical_seno/T_{Tfin}/Cmodel_{COLLOCATION_DISTRIBUTION}_{N_COLLOCATION}_NEURO{INTRA_DEPTH}{INTRA_NEUROS}{intra_neuro}_{run_id}.pt",
            training_params=dict(
                NU = NU, HX = HX, HT = HT, N_COLLOCATION = N_COLLOCATION, SEED = SEED,
                COLLOCATION_DISTRIBUTION = COLLOCATION_DISTRIBUTION,
                EPOCHS = EPOCHS, LR = LR, LBFGS_EPOCHS = LBFGS_EPOCHS, LBFGS_LR = LBFGS_LR,
                HIDDEN_NEURONS = HIDDEN_NEURONS, HIDDEN_DEPTH = HIDDEN_DEPTH,
                # Condicao de parada: sempre avaliada numa grade uniforme fixa
                # com N_COLLOCATION pontos (ver STOPPING_DISTRIBUTION), mesmo
                # quando o treino em si usa COLLOCATION_DISTRIBUTION="random".
                STOPPING_DISTRIBUTION = STOPPING_DISTRIBUTION,
                TARGET_LOSS = target_loss, TARGET_REACHED = hist["target_reached"],
                STOPPED_AT_STEP = len(hist["loss"]),
                STOPPING_LOSS_VALUE = hist["loss"][-1] if hist["loss"] else None,
            ),
        )

        # Historico completo da funcao custo (todos os passos, AdamW + L-BFGS),
        # salvo a parte do checkpoint para nao inchar o json legivel do model.save.
        np.savez(
            f"data/Forward/classical_seno/T_{Tfin}/losshist_{COLLOCATION_DISTRIBUTION}_{N_COLLOCATION}_CNEURO{INTRA_DEPTH}{INTRA_NEUROS}{intra_neuro}_{run_id}.npz",
            loss=np.array(hist["loss"]),
            residual_rms=np.array(hist["residual_rms"]),
            target_loss=target_loss,
            target_reached=hist["target_reached"],
        )

    plot_loss({"Classical PINN": hist["loss"]}, title="Classical PINN training loss",
                save_path=f"data/Forward/classical_seno/T_{Tfin}/loss_{COLLOCATION_DISTRIBUTION}_{N_COLLOCATION}_CNEURO{INTRA_DEPTH}{INTRA_NEUROS}{intra_neuro}_{run_id}.png"
                )

    u_pred = predict(model, x_plot, Tfin)
    u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)
    plot_solution(
        x_plot, {"Classical PINN": u_pred}, ref=u_ref,
        title= f"Classical PINN vs FDM reference at T={Tfin}",
        save_path = f"data/Forward/classical_seno/T_{Tfin}/model_{COLLOCATION_DISTRIBUTION}_{N_COLLOCATION}_CNEURO{INTRA_DEPTH}{INTRA_NEUROS}{intra_neuro}_{run_id}.png",
        show_cond = False,
        collocation_x = x_f.numpy(),
    )

    # print(f"Relative L2 error at T={Tfin}: {relative_l2_error(u_pred, u_ref):.6e}")
    return N_COLLOCATION, intra_neuro, Tfin, run_id


if __name__ == "__main__":
    ################# PARALELIZAÇAO ######################################
    SEEDS_NEW = range(0, 10)    # seeds completas para o bloco C (analise nova)

    # Bloco A: varredura de N_COLLOCATION com intra_neuro em [3, 5, 7, 10, 15]
    # (mesmos valores usados em plote_classico_vs_collocation.py, para que a
    # curva "Classical (seno)" tenha o sweep completo igual a "Classical").
    block_A = [
        (n_col, intra_neuro, Tfin, run_id)
        for n_col in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for intra_neuro in [3, 5, 7, 10, 15]
        for Tfin in [0.1, 1]
        for run_id in SEEDS_NEW
    ]

    # Bloco B: varredura de intra_neuro com N_COLLOCATION fixo em 10 e 100
    # (refaz/estende o que ja existia com seed 0-9).
    block_B = [
        (n_col, intra_neuro, Tfin, run_id)
        for n_col in [10, 100]
        for intra_neuro in [1, 2, 4, 5, 6, 7, 8, 9]
        for Tfin in [0.1, 1]
        for run_id in SEEDS_NEW
    ]

    jobs_by_key = {}
    for block in (block_A, block_B):
        for job in block:
            jobs_by_key[job] = job
    jobs = list(jobs_by_key.values())

    # Retomada apos interrupcao: pula jobs cujo checkpoint (.pt) ja existe em
    # disco, para nao re-treinar o que ja foi concluido antes do PC desligar.
    def _output_path(job):
        n_col, intra_neuro, Tfin, run_id = job
        return (
            f"data/Forward/classical_seno/T_{Tfin}/Cmodel_{COLLOCATION_DISTRIBUTION}_{n_col}"
            f"_NEURO{INTRA_DEPTH}{INTRA_NEUROS}{intra_neuro}_{run_id}.pt"
        )

    jobs_done = [job for job in jobs if os.path.exists(_output_path(job))]
    jobs = [job for job in jobs if not os.path.exists(_output_path(job))]
    print(f"Ja concluidos (pulados): {len(jobs_done)} runs")
    print(f"Faltando (serao rodados agora): {len(jobs)} runs")

    for _, _, Tfin, _ in jobs:
        os.makedirs(f"data/Forward/classical_seno/T_{Tfin}", exist_ok=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_one, *job): job for job in jobs}
        progress = tqdm(as_completed(futures), total=len(jobs), desc="runs", unit="run")
        n_failed = 0
        for future in progress:
            n_col, intra_neuro, Tfin, run_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                n_failed += 1
                progress.write(
                    f"FAILED: N_COLLOCATION={n_col} intra_neuro={intra_neuro} "
                    f"Tfin={Tfin} run_id={run_id} -> {exc!r}"
                )
            progress.set_postfix(failed=n_failed)
        if n_failed:
            print(f"{n_failed} run(s) failed -- see FAILED lines above.")

