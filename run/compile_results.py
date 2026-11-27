"""Compile the raw training outputs into the result tables of data/: one set of CSVs per problem, model and T.

    python run/compile_results.py

data/forward/classico/simulacao_classico_T_<T>.csv     one row per run (width and N_col studies, column `study`)
data/forward/quantico/simulador_quantico_T_<T>.csv     noiseless HQPINN, the same
data/forward/quantico/qpu_<backend>_T_<T>_*.csv        real-QPU run: resumo, perfil, curva, cadeias
data/inverse/<model>/<source>_T_<T>_*.csv              rodadas (one row per run), historico (nu and losses per epoch;
                                                       log-binned medians at 20 000 epochs), perfis (u(x,T) per run)
data/inverse/quantico/qpu_<backend>_T_<T>_*.csv        real-QPU run
A raw source that is missing is skipped. The raw files are not in the repository: they are in git up to commit e2be4d229.
"""
# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root.
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
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUT_F, OUT_I = DATA / "forward", DATA / "inverse"

# ---- raw sources (where the training scripts write) -------------------------------
FORWARD_T = (0.1,)          # T = 1 is being redone: add 1.0 when its runs exist
SWEEP = DATA / "Forward/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/qubit_sweep_runs.csv"
CLASSICAL_NCOL = DATA / "Forward/classical_same_protocol_ncol/qubit_sweep_runs.csv"
EXACT_NCOL = {3: "CBPF_quantum_exact_results_Brickwork3", 5: "CBPF_quantum_exact_results__Brickwork5",
              7: "CBPF_quantum_exact_results_fixed_BrickWork7", 10: "CBPF_quantum_exact_results_fixed_BrickWork10",
              15: "CBPF_quantum_exact_results_fixed_BrickWork15"}
MAX_N_COLLOCATION = 100
# forward real-QPU runs: run_/validation_ files, classical control, replayed learning curves (data/derived)
FORWARD_QPU = {0.1: dict(backend="ibm_berlin", raw=DATA / "quantum_forward_hardware_real/T_0.1",
                         tag="10_NQ10_0_ibm_berlin_none_HW110", classical=DATA / "Forward/classical_vs_qpu/T_0.1",
                         curves=DATA / "derived/learning_curves.npz", sims=DATA / "derived/same_budget_sims.npz")}

INV_CASE = {0.1: "T_0.1_nu_0.01", 1.0: "T_1.0_nu_0.01"}
INV_BUDGETS = {200: dict(root=DATA / "Inverso_200", quantum="quantico", sizes=(10,)),          # HQPINN on FakeAachen
               20000: dict(root=DATA / "Inverso", quantum="quantico_ideal", sizes=(3, 5, 7, 10, 12, 15))}
INV_MODELS = {"classical": ("classico", "NE", "neurons"), "hqpinn": ("quantico", "NQ", "n_qubits")}
INVERSE_QPU = {0.1: dict(backend="ibm_aachen", folder=DATA / "Inverso_real/ibm_aachen/quantico/T_0.1_nu_0.01",
                         tag="10_NQ10_1_E200")}


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{path}  ({len(df)} rows)")


def col(a):
    return np.ravel(a)


# ---- forward --------------------------------------------------------------------------
def forward_simulations(T):
    drop = ["model_checkpoint", "checkpoint_saved_at_target"]
    sweep = pd.read_csv(SWEEP).query("Tfin == @T").assign(study="width")
    extra = pd.read_csv(CLASSICAL_NCOL).query("Tfin == @T and protocol_version == 'fixed_validation_v2'")
    classical = pd.concat([sweep[sweep.model_family == "classical_mlp"], extra.assign(study="points")])
    exact = [pd.read_csv(DATA / "Forward" / d / "quantum_exact_runs.csv").query(
        "Tfin == @T and n_collocation <= @MAX_N_COLLOCATION").assign(n_qubits=n, model_family="quantum_exact",
                                                                    study="points") for n, d in EXACT_NCOL.items()]
    quantum = pd.concat([sweep[sweep.model_family == "quantum_exact"], *exact])
    for df, name in ((classical, f"classico/simulacao_classico_T_{T:g}"), (quantum, f"quantico/simulador_quantico_T_{T:g}")):
        write(df.drop(columns=drop, errors="ignore"), OUT_F / f"{name}.csv")


def forward_qpu(T, c):
    run_file = c["raw"] / f"run_{c['tag']}.json"
    if not run_file.exists():
        return
    run, v = json.loads(run_file.read_text()), np.load(c["raw"] / f"validation_{c['tag']}.npz")
    base = OUT_F / "quantico" / f"qpu_{c['backend']}_T_{T:g}"
    keys = ("backend", "seed", "n_qubits", "n_collocation", "Tfin", "mitigation", "shots", "mse", "relative_l2")
    write(pd.DataFrame([{k: run[k] for k in keys} | dict(epochs=run["training_config"]["epochs"])]),
          Path(f"{base}_resumo.csv"))
    prof = dict(x=col(v["x"]), u_fdm=col(v["u_reference"]), u_initial=col(v["u_initial_surrogate"]), u_qpu=col(v["u_qpu"]),
                u_qpu_weights_ideal_circuit=col(v["u_ideal"]))
    cls = c["classical"] / f"validation_{run['n_collocation']}_NEURO110{run['n_qubits']}_tanh_{run['seed']}_E110.npz"
    if cls.exists():
        prof[f"u_classical_seed{run['seed']}"] = col(np.load(cls)["u"])
    write(pd.DataFrame(prof), Path(f"{base}_perfil.csv"))
    write(pd.DataFrame([dict(chain=k + 1, qubit=q) for k, ch in enumerate(run["chains"]) for q in ch]),
          Path(f"{base}_cadeias.csv"))
    if c["curves"].exists() and c["sims"].exists():     # MSE per epoch, long format: one curve per (series, width, seed)
        lc, sims = np.load(c["curves"]), np.load(c["sims"])
        ep = lc["epochs"]
        rows = [pd.DataFrame(dict(series="qpu", width=run["n_qubits"], seed=run["seed"], epoch=ep, mse=lc["real_mse_surrogate"]))]
        for i, w in enumerate(lc["classical_widths"]):
            for j, s in enumerate(lc["classical_seeds"]):
                rows.append(pd.DataFrame(dict(series="classical", width=w, seed=s, epoch=ep, mse=lc["classical_mse"][i, j])))
        for j, s in enumerate(sims["seeds"]):
            rows.append(pd.DataFrame(dict(series="noiseless_hqpinn", width=run["n_qubits"], seed=s, epoch=ep,
                                          mse=sims["ideal_mse"][j])))
        write(pd.concat(rows), Path(f"{base}_curva.csv"))


# ---- inverse --------------------------------------------------------------------------
def log_bins(values, per_decade=60):
    """Medians over logarithmic bins of epochs (single epochs at the start, wide bins at the end)."""
    n = len(values)
    edges = np.unique(np.round(np.logspace(0, np.log10(n + 1), int(per_decade * np.log10(n + 1)) + 1)).astype(int))
    return np.array([(np.sqrt(a * (b - 1)), np.median(values[a - 1:b - 1])) for a, b in zip(edges[:-1], edges[1:])]).T


def trainable_parameters(model, size):
    """Parameters trained by AdamW (nu, updated in closed form, is not counted)."""
    if model == "classical":
        from Inverser_classical_pinn_paralelo import build_model
    else:
        from Inverser_quantum_exact_pinn_paralelo import build_model
    net = build_model(size, 1.0, 0.5)
    return sum(p.numel() for name, p in net.named_parameters() if p.requires_grad and "learnable_parameter" not in name)


def inverse_runs(T, model):
    folder, prefix, size_key = INV_MODELS[model]
    runs, hist, prof = [], [], []
    for epochs, b in INV_BUDGETS.items():
        path = b["root"] / (b["quantum"] if model == "hqpinn" else folder) / INV_CASE[T]
        emulated = model == "hqpinn" and epochs == 200
        for f in sorted(path.glob(f"run_*_E{epochs}.json")):
            run = json.loads(f.read_text())
            size, seed = run[size_key], run["seed"]
            if size not in b["sizes"]:
                continue
            tag = f"10_{prefix}{size}_{seed}_E{epochs}"
            v, h = np.load(path / f"validation_{tag}.npz"), np.load(path / f"losshist_{tag}.npz")
            u = col(v["u_qpu"] if emulated else v["u_pred"])
            row = dict(epochs=epochs, size=size, seed=seed, nu=run["nu"], mse_u=float(np.mean((u - col(v["u_exact"])) ** 2)),
                       time_s=run["time_s"], params=trainable_parameters(model, size))
            if emulated:
                row |= dict(backend=run["backend"], l2=run["rel_l2_T_qpu"], l2_surrogate=run["rel_l2_T_measured_model"],
                            l2_ideal=run["rel_l2_T_ideal_circuit"], n_jobs=run["n_jobs"], chains=run["chains"],
                            fit_rms=run["fit_rms_calibration"])
            else:
                row |= dict(backend="ideal" if model == "hqpinn" else "", l2=run["rel_l2_T"])
            runs.append(row)
            if len(h["nu"]) > 1000:
                epoch, nu = log_bins(h["nu"])
                lr, ld = log_bins(h["loss_r"])[1], log_bins(h["loss_d"])[1]
            else:
                epoch, nu, lr, ld = np.arange(1, len(h["nu"]) + 1), h["nu"], h["loss_r"], h["loss_d"]
            hist.append(pd.DataFrame(dict(epochs=epochs, size=size, seed=seed, epoch=epoch, nu=nu, loss_residual=lr,
                                          loss_data=ld)))
            p = dict(epochs=epochs, size=size, seed=seed, x=col(v["x"]), u_exact=col(v["u_exact"]), u=u)
            if emulated:
                p["u_ideal"] = col(v["u_ideal_circuit"])
            prof.append(pd.DataFrame(p))
    if not runs:
        return
    source = "simulador_quantico" if model == "hqpinn" else "simulacao_classico"
    base = OUT_I / folder / f"{source}_T_{T:g}"
    write(pd.DataFrame(runs), Path(f"{base}_rodadas.csv"))
    write(pd.concat(hist), Path(f"{base}_historico.csv"))
    write(pd.concat(prof), Path(f"{base}_perfis.csv"))


def inverse_qpu(T, c):
    folder, tag = c["folder"], c["tag"]
    if not (folder / f"run_{tag}.json").exists():
        return
    run = json.loads((folder / f"run_{tag}.json").read_text())
    h, v = np.load(folder / f"losshist_{tag}.npz"), np.load(folder / f"validation_{tag}.npz")
    base = OUT_I / "quantico" / f"qpu_{c['backend']}_T_{T:g}"
    write(pd.DataFrame([dict(backend=c["backend"], n_qubits=run["n_qubits"], seed=run["seed"], epochs=run["epochs_done"],
                             nu=run["nu"], l2=run["rel_l2_T_qpu"], l2_surrogate=run["rel_l2_T_measured_model"],
                             l2_ideal=run["rel_l2_T_ideal_circuit"], chains=run["chains"], n_jobs=run["n_jobs"],
                             fit_rms=run["fit_rms_calibration"], time_s=run["time_s"])]), Path(f"{base}_resumo.csv"))
    write(pd.DataFrame(dict(epoch=np.arange(1, len(h["nu"]) + 1), nu=h["nu"], loss_residual=h["loss_r"],
                            loss_data=h["loss_d"])), Path(f"{base}_historico.csv"))
    write(pd.DataFrame(dict(x=col(v["x"]), u_qpu=col(v["u_qpu"]), u_exact=col(v["u_exact"]))), Path(f"{base}_perfil.csv"))
    write(pd.DataFrame([dict(chain=k, qubit=q) for k, ch in enumerate(run["chain_list"]) for q in ch]),
          Path(f"{base}_cadeias.csv"))
    write(coupling_map(c["backend"], run["chain_list"]), OUT_I / "quantico" / f"acoplamento_{c['backend']}.csv")


def coupling_map(backend, chains):
    """Couplers of the device (its fake snapshot in qiskit-ibm-runtime); every link of every chain must be one."""
    import importlib
    fake = getattr(importlib.import_module("qiskit_ibm_runtime.fake_provider"), "Fake" + backend.removeprefix("ibm_").title())()
    edges = sorted({tuple(sorted(e)) for e in fake.coupling_map.get_edges()})
    bad = [(a, b) for ch in chains for a, b in zip(ch, ch[1:]) if tuple(sorted((a, b))) not in set(edges)]
    if bad:
        raise ValueError(f"chain links not in the coupling map of {backend}: {bad}")
    return pd.DataFrame(edges, columns=["q0", "q1"])


if __name__ == "__main__":
    for T in FORWARD_T:
        if SWEEP.exists():
            forward_simulations(T)
        if T in FORWARD_QPU:
            forward_qpu(T, FORWARD_QPU[T])
    for T in INV_CASE:
        for model in INV_MODELS:
            inverse_runs(T, model)
        if T in INVERSE_QPU:
            inverse_qpu(T, INVERSE_QPU[T])
