"""Build manuscript/data/: the small tables the manuscript figures are drawn from.

    python manuscript/scripts/build_paper_data.py

data/ holds the raw results and never goes to Overleaf. manuscript/ does, so this script condenses the raw
results into a few CSV files there, and the figure scripts read only those files. This file writes the forward
problem (manuscript/data/forward/); build_inverse_data.py writes the inverse problem (manuscript/data/inverse/).

The simulation tables use same-protocol runs only (same stopping target per seed for both models).
The old data/Forward/classical* is left out on purpose: it stops on another monitor and, for n != 10, has
another middle block (see docs/manuscript_notes_pt.md).

A QPU case listed in QPU_CASES with raw data missing is written as status "pending"; the figure then
draws an empty, labelled panel. Nothing is ever filled in for a run that did not happen.
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

import numpy as np
import pandas as pd

import build_inverse_data
from paper_common import (CLASSICAL_NCOL_CSV, COLS, PAPER_DATA, QPU_CASES, load_exact_ncol, load_json, load_npz,
                          load_sweep)

MODEL = {"classical_mlp": "classical", "quantum_exact": "hqpinn"}
MAX_N_COLLOCATION = 100        # range of the analysis versus N_col


def run_stats(df, keys):
    """Median and quartiles of the MSE, mean and standard deviation of the steps, over all runs of each group.

    A run that misses the target stops at the step budget, so its step count is a lower bound of the steps it
    would need: steps_mean counts those runs at the budget and is a lower bound too.
    """
    g = df.groupby(keys)
    return pd.DataFrame(dict(
        params=g.trainable_parameters.first(), mse_median=g.mse.median(), mse_q25=g.mse.quantile(0.25),
        mse_q75=g.mse.quantile(0.75), steps_mean=g.stopped_at_step.mean(), steps_std=g.stopped_at_step.std(ddof=0),
        reached_fraction=g.target_reached.mean(), n_runs=g.size())).reset_index()


def check_same_setup(classical, hqpinn, keys):
    """Both models must stop on the same monitor, with the same target for the same seed. Fails loudly otherwise."""
    for name, df in (("classical", classical), ("hqpinn", hqpinn)):
        monitors = set(df.stopping_monitor)
        if monitors != {"fixed_validation"}:
            raise ValueError(f"{name}: other stopping monitor {monitors}")
    both = classical.merge(hqpinn, on=keys, suffixes=("_c", "_q"))
    if both.empty or not np.allclose(both.target_loss_c, both.target_loss_q, rtol=1e-9, atol=0):
        raise ValueError("classical and HQPINN runs do not share the stopping target")


def build_simulation(out):
    sweep = load_sweep().rename(columns={"n_qubits": "n"})
    sweep["model"] = sweep.model_family.map(MODEL)
    check_same_setup(sweep[sweep.model == "classical"], sweep[sweep.model == "hqpinn"], ["Tfin", "n_collocation", "n", "seed"])
    run_stats(sweep, ["model", "Tfin", "n_collocation", "n"]).to_csv(out / "width_sweep.csv", index=False,
                                                                     float_format="%.6g")

    exact = load_exact_ncol().rename(columns={"n_qubits": "n"})
    exact["model"] = "hqpinn"
    classical = sweep[(sweep.model == "classical") & sweep.n.isin(exact.n.unique())]
    if CLASSICAL_NCOL_CSV.exists():
        extra = pd.read_csv(CLASSICAL_NCOL_CSV).rename(columns={"n_qubits": "n"})
        extra = extra[(extra.model_family == "classical_mlp") & extra.n.isin(exact.n.unique())
                      & (extra.protocol_version == "fixed_validation_v2")].assign(model="classical")
        classical = pd.concat([classical, extra], ignore_index=True).drop_duplicates(
            ["Tfin", "n", "n_collocation", "seed"])
    check_same_setup(classical, exact, ["Tfin", "n_collocation", "n", "seed"])
    runs = pd.concat([exact[exact.n_collocation <= MAX_N_COLLOCATION], classical], ignore_index=True)
    run_stats(runs, ["model", "Tfin", "n", "n_collocation"]).to_csv(out / "points_sweep.csv", index=False,
                                                                    float_format="%.6g")


def fdm_profile(T, n_points=200):
    from main.functions import reference_at_time, solve_burgers_fd
    x_ref, t_ref, U_ref = solve_burgers_fd(L=1.0, Tfin=T, nu=0.01)
    x = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
    u, _ = reference_at_time(T, t_ref, x_ref, U_ref, x, L=1.0)
    return x.ravel(), np.ravel(u)


def build_qpu_case(case, out):
    """Tables of one QPU case. Returns its row of cases.csv."""
    T, tag = case["Tfin"], case["tag"]
    raw = case["raw_dir"]
    row = {k: case[k] for k in ("Tfin", "backend", "n", "n_collocation", "seed", "epochs", "shots", "mitigation")}
    row.update(trainable_parameters=21 * case["n"] + 61 if case["n"] else "", status="pending", mse="", relative_l2="")
    folder = out / f"T_{T:g}"
    folder.mkdir(parents=True, exist_ok=True)
    run_file = raw / f"run_{tag}.json"
    if not run_file.exists():
        x, u = fdm_profile(T)
        pd.DataFrame(dict(x=x, u_fdm=u)).to_csv(folder / "profile.csv", index=False, float_format="%.6g")
        return row

    run, val = load_json(run_file), load_npz(raw / f"validation_{tag}.npz")
    profile = dict(x=val["x"].ravel(), u_fdm=val["u_reference"].ravel(), u_initial=val["u_initial_surrogate"].ravel(),
                   u_qpu=val["u_qpu"].ravel(), u_qpu_weights_ideal_circuit=val["u_ideal"].ravel())
    cls = case["classical_dir"] / f"validation_{case['n_collocation']}_NEURO110{case['n']}_tanh_{case['seed']}_E{case['epochs']}.npz"
    if cls.exists():
        profile[f"u_classical_seed{case['seed']}"] = load_npz(cls)["u"].ravel()
    pd.DataFrame(profile).to_csv(folder / "profile.csv", index=False, float_format="%.6g")

    if case["curves"].exists() and case["sims"].exists():      # one replay per horizon: never reuse another T's
        lc, sims = load_npz(case["curves"]), load_npz(case["sims"])
        c = lc["classical_mse"].reshape(-1, len(lc["epochs"]))
        pd.DataFrame(dict(
            epoch=lc["epochs"], mse_qpu=lc["real_mse_surrogate"],
            mse_classical_median=np.median(c, 0), mse_classical_q25=np.quantile(c, 0.25, 0),
            mse_classical_q75=np.quantile(c, 0.75, 0), mse_noiseless_hqpinn_median=np.median(sims["ideal_mse"], 0),
        )).to_csv(folder / "learning_curve.csv", index=False, float_format="%.6g")

    pd.DataFrame([dict(chain=k + 1, qubit=q, row=q // COLS, column=q % COLS)
                  for k, chain in enumerate(run["chains"]) for q in chain]).to_csv(folder / "chains.csv", index=False)
    row.update(status="done", mse=run["mse"], relative_l2=run["relative_l2"])
    return row


def main():
    forward = PAPER_DATA / "forward"
    (forward / "simulation").mkdir(parents=True, exist_ok=True)
    (forward / "qpu").mkdir(parents=True, exist_ok=True)
    build_simulation(forward / "simulation")
    rows = [build_qpu_case(case, forward / "qpu") for case in QPU_CASES]
    pd.DataFrame(rows).to_csv(forward / "qpu" / "cases.csv", index=False)
    build_inverse_data.main()
    files = [p for p in PAPER_DATA.rglob("*") if p.is_file()]
    print(f"manuscript/data: {len(files)} files, {sum(p.stat().st_size for p in files) / 1024:.0f} kB")
    for p in sorted(files):
        print(f"  {p.relative_to(PAPER_DATA)}  ({p.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
