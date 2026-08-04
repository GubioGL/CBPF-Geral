"""Plots the FDM (solver) reference solution against the quantum PINN
predictions (model_NQUBITS{N}_p_1.pt / p_2.pt) and a classical PINN baseline,
one figure per T_* directory in data/quantum_old.

Per-T settings (chosen to match what's interesting to compare at each
horizon): T_0.1 uses the N_QUBITS=3 quantum checkpoints against the
classical 20-20 (ClassicalPINN) baseline; T_1 uses the N_QUBITS=14 quantum
checkpoints against the classical 10-10-10 (ClassicalPINN2) baseline.
"""
import glob
import json
import os

import numpy as np
import torch as tc

from src.config import NU
from src.functions import plot_solution, predict, reference_at_time, solve_burgers_fd
from src.models import ClassicalPINN, ClassicalPINN2
from src.models_pennylane import QuantumPINNPennyLane

QUANTUM_DIR = "data/quantum_old"
CLASSICAL_DIR = "data/classical_old"
P_INDICES = [1, 2]

# T_* directory (basename) -> (N_QUBITS to plot, classical baseline to overlay)
T_SETTINGS = {
    "T_0.1": dict(n_qubits=3, classical="20-20"),
    "T_1": dict(n_qubits=12, classical="10-10-10"),
}
CLASSICAL_COLOR = "black"


def load_legacy_quantum(path):
    """Loads a QuantumPINNPennyLane checkpoint from `data/quantum_old`. Those
    checkpoints predate the `estrutura` config field and were all trained
    with the pre -> quantum -> post architecture, i.e. `estrutura=1` --
    `QuantumPINNPennyLane.load` would otherwise fall back to the current
    default `estrutura=0`, which requires n_qubits=3 and breaks on every
    other qubit count."""
    checkpoint = tc.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config.setdefault("estrutura", 1)
    model = QuantumPINNPennyLane(**config)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def classical_20_20_curve(classical_t_dir, x_plot, Tfin):
    """Mean prediction curve over every ClassicalPINN (20-20) `amostragem_*.pt`
    checkpoint in `classical_t_dir`, or None if none are found."""
    paths = sorted(glob.glob(os.path.join(classical_t_dir, "amostragem_*.pt")))
    if not paths:
        return None
    preds = [predict(ClassicalPINN.load(p), x_plot, Tfin) for p in paths]
    return np.mean(preds, axis=0)


def classical_mid_layer_curve(classical_t_dir, intra_neuron, x_plot, Tfin):
    """Mean prediction curve over every ClassicalPINN2 `Cmodel_NEURO110{n}_*.pt`
    checkpoint (pre=10, intra=n, post=10) in `classical_t_dir` with bottleneck
    width `intra_neuron`, or None if none are found."""
    paths = []
    for config_path in glob.glob(os.path.join(classical_t_dir, "Cmodel_NEURO*_config.json")):
        pt_path = config_path[: -len("_config.json")] + ".pt"
        if not os.path.exists(pt_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)["config"]
        if cfg.get("intra_neuron") == intra_neuron and cfg.get("intra_neurons") == 10 and cfg.get("intra_depth") == 1:
            paths.append(pt_path)
    if not paths:
        return None
    preds = [predict(ClassicalPINN2.load(p), x_plot, Tfin) for p in paths]
    return np.mean(preds, axis=0)


if __name__ == "__main__":
    for t_dir in sorted(glob.glob(os.path.join(QUANTUM_DIR, "T_*"))):
        t_name = os.path.basename(t_dir)
        settings = T_SETTINGS.get(t_name)
        if settings is None:
            print(f"  -> no T_SETTINGS entry for {t_name}, skipping")
            continue
        n_qubits, classical_arch = settings["n_qubits"], settings["classical"]

        paths = {
            p: os.path.join(t_dir, f"model_NQUBITS{n_qubits}_p_{p}.pt")
            for p in P_INDICES
        }
        paths = {p: path for p, path in paths.items() if os.path.exists(path)}
        if not paths:
            print(f"  -> no model_NQUBITS{n_qubits}_p_* checkpoints in {t_dir}, skipping")
            continue

        models = {p: load_legacy_quantum(path) for p, path in paths.items()}
        L, Tfin = next(iter(models.values())).L, next(iter(models.values())).Tfin

        ############################### Numerico (solver) ###################
        x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=NU)
        x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
        u_ref, _ = reference_at_time(Tfin, t_ref, x_ref, U_ref, x_plot, L=L)

        ############################### Quantum (PINN) #######################
        curves = {
            f"NQUBITS{n_qubits} p_{p}": predict(model, x_plot, Tfin)
            for p, model in models.items()
        }
        colors = {}

        ############################### Classical baseline ##################
        classical_t_dir = os.path.join(CLASSICAL_DIR, t_name)
        if classical_arch == "20-20":
            classical_curve = classical_20_20_curve(classical_t_dir, x_plot, Tfin)
        else:
            classical_curve = classical_mid_layer_curve(classical_t_dir, 10, x_plot, Tfin)

        if classical_curve is not None:
            classical_label = f"Classical {classical_arch}"
            curves[classical_label] = classical_curve
            colors[classical_label] = CLASSICAL_COLOR
        else:
            print(f"  -> no classical {classical_arch} checkpoints in {classical_t_dir}")

        ######################### PLOT ########################################
        t_label = t_name.replace("T_", "")
        save_path = f"Solution_NQUBITS{n_qubits}_comparativo_T_{t_label}.png"
        plot_solution(
            x_plot, curves, ref=u_ref,
            title=f"Quantum HPINN (NQUBITS={n_qubits}) vs FDM reference at T={Tfin}",
            save_path=save_path,
            show_cond=False,
            colors=colors,
        )
        print(f"T={Tfin}: saved {save_path}")
