"""Build manuscript/data/inverse/: the tables behind the inverse-problem figures and tables.

    python manuscript/scripts/build_inverse_data.py

Reads the runs of data/Inverso_110 (FakeBerlin) and data/Inverso_500 (FakeAachen), and writes:

  runs.csv            one row per run (264 rows)
  summary.csv         per budget, T and model: error of nu and of u(x,T) over the 33 runs
  per_size.csv        the same per size n = k = 3, 7, 10
  paired.csv          per budget and T: paired wins and two-sided Wilcoxon p, HQPINN vs classical
  budget_effect.csv   per T and model: 110 vs 500 epochs, same size and seed (HQPINN: confounded with the backend)
  nu_curves.csv       per budget, T, model and epoch: nu and its relative error over the 33 runs
  field.csv           u(x,T) of the paired run n = k = 10, seed 0, at each budget
  devices.csv         chains and calibration misfit per backend and n
  long_budget.csv     older long-budget sweeps (another protocol, never compared with the runs above)
  long_budget_ncol.csv  the same sweeps: mean squared error of nu per model, T, width and N_col (Appendix C figure)
  best_profile.csv, best_history.csv  best run of each model and T at 110 epochs: u(x,T), and nu and loss per epoch
  qpu_cases.csv       the real-QPU cases and their status; qpu/T_<T>/ holds the data of a case that exists

and the two LaTeX tables the manuscript \\input-s, so that they cannot drift from the data:

  manuscript/tables/inverse_summary.tex, manuscript/tables/inverse_per_size.tex

A real-QPU case with raw data missing is written as status "pending" and its panels stay empty. Nothing is ever
filled in for a run that did not happen.
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
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from paper_common import (INV_CASES, INV_N, INV_NU, INV_SIZES, INVERSE_BUDGETS, INVERSE_LONG, INVERSE_QPU_CASES,
                          INVERSE_REAL_ROOT, PAPER_DATA, ROOT)

TABLES = ROOT / "manuscript" / "tables"

MODELS = {"classical": ("classico", "NE", "neurons"), "hqpinn": ("quantico", "NQ", "n_qubits")}


def tag(prefix, size, seed, epochs):
    return f"{INV_N}_{prefix}{size}_{seed}_E{epochs}"


def load_runs():
    """One row per run, with the per-epoch nu of each run kept apart (it does not go in the table)."""
    rows, nu = [], {}
    for epochs, budget in INVERSE_BUDGETS.items():
        for T, case in INV_CASES.items():
            for model, (folder, prefix, size_key) in MODELS.items():
                path = budget["root"] / folder / case
                for run in json.loads((path / "summary.json").read_text())["runs"]:
                    size, seed = run[size_key], run["seed"]
                    if size not in INV_SIZES or seed not in range(11):
                        raise ValueError(f"unexpected run {model} size {size} seed {seed} in {path}")
                    row = dict(epochs=epochs, Tfin=T, model=model, size=size, seed=seed, nu=run["nu"],
                               eps_nu=abs(run["nu"] - INV_NU) / INV_NU, time_s=run["time_s"])
                    if model == "classical":
                        row.update(backend="", l2=run["rel_l2_T"])
                    else:
                        row.update(backend=run["backend"], l2=run["rel_l2_T_qpu"],
                                   l2_surrogate=run["rel_l2_T_measured_model"], l2_ideal=run["rel_l2_T_ideal_circuit"],
                                   n_jobs=run["n_jobs"], chains=run["chains"], fit_rms=run["fit_rms_calibration"])
                    rows.append(row)
                    nu[epochs, T, model, size, seed] = np.load(path / f"losshist_{tag(prefix, size, seed, epochs)}.npz")["nu"]
    df = pd.DataFrame(rows)
    counts = df.groupby(["epochs", "Tfin", "model"]).size()
    if not (counts == 33).all():
        raise ValueError(f"expected 33 runs per model and case:\n{counts}")
    return df, nu


def summarize(g):
    out = dict(runs=len(g), eps_median=g.eps_nu.median(), eps_mean=g.eps_nu.mean(), eps_p90=g.eps_nu.quantile(0.9),
               eps_max=g.eps_nu.max(), eps_q25=g.eps_nu.quantile(0.25), eps_q75=g.eps_nu.quantile(0.75),
               overestimates=(g.nu > INV_NU).mean(), l2_median=g.l2.median(), l2_mean=g.l2.mean(), l2_max=g.l2.max(),
               time_median_s=g.time_s.median())
    if g.l2_ideal.notna().any():
        out.update(l2_ideal_median=g.l2_ideal.median(), l2_surrogate_median=g.l2_surrogate.median(),
                   ideal_worse=(g.l2_ideal > g.l2).mean(), ideal_over_device_median=(g.l2_ideal / g.l2).median())
    return out


def summary_table(df, keys):
    """One row per group, in wide format (the HQPINN-only columns are empty for the classical model)."""
    return pd.DataFrame([dict(zip(keys, k), **summarize(g)) for k, g in df.groupby(keys)])


def compare(a, b, column):
    """a and b aligned on (size, seed): how many times a < b, and the two-sided paired Wilcoxon p."""
    wins = int((a[column].values < b[column].values).sum())
    return wins, int(len(a) - wins), float(wilcoxon(a[column].values, b[column].values).pvalue)


def paired(df):
    rows = []
    for (epochs, T), g in df.groupby(["epochs", "Tfin"]):
        q = g[g.model == "hqpinn"].sort_values(["size", "seed"])
        c = g[g.model == "classical"].sort_values(["size", "seed"])
        if not (q[["size", "seed"]].values == c[["size", "seed"]].values).all():
            raise ValueError("the runs are not paired")
        wq, wc, p = compare(q, c, "eps_nu")
        wql, wcl, pl = compare(q, c, "l2")
        rows.append(dict(epochs=epochs, Tfin=T, nu_wins_hqpinn=wq, nu_wins_classical=wc, nu_p=p, l2_wins_hqpinn=wql,
                         l2_wins_classical=wcl, l2_p=pl))
    return pd.DataFrame(rows)


def budget_effect(df):
    rows = []
    for (T, model), g in df.groupby(["Tfin", "model"]):
        a = g[g.epochs == 500].sort_values(["size", "seed"])
        b = g[g.epochs == 110].sort_values(["size", "seed"])
        wn, _, pn = compare(a, b, "eps_nu")
        wl, _, pl = compare(a, b, "l2")
        rows.append(dict(Tfin=T, model=model, eps_median_110=b.eps_nu.median(), eps_median_500=a.eps_nu.median(),
                         nu_better_at_500=wn, nu_p=pn, l2_median_110=b.l2.median(), l2_median_500=a.l2.median(),
                         l2_better_at_500=wl, l2_p=pl))
    return pd.DataFrame(rows)


def nu_curves(nu):
    rows = []
    for epochs in INVERSE_BUDGETS:
        for T in INV_CASES:
            for model in MODELS:
                H = np.array([h for k, h in sorted(nu.items()) if k[:3] == (epochs, T, model)])
                E = np.abs(H - INV_NU) / INV_NU
                rows.append(pd.DataFrame(dict(
                    epochs=epochs, Tfin=T, model=model, epoch=np.arange(1, H.shape[1] + 1), nu_median=np.median(H, 0),
                    nu_p10=np.percentile(H, 10, 0), nu_p90=np.percentile(H, 90, 0), eps_median=np.median(E, 0),
                    eps_q25=np.percentile(E, 25, 0), eps_q75=np.percentile(E, 75, 0))))
    return pd.concat(rows, ignore_index=True)


def field(size=10, seed=0):
    rows = []
    for epochs, budget in INVERSE_BUDGETS.items():
        for T, case in INV_CASES.items():
            q = np.load(budget["root"] / "quantico" / case / f"validation_{tag('NQ', size, seed, epochs)}.npz")
            c = np.load(budget["root"] / "classico" / case / f"validation_{tag('NE', size, seed, epochs)}.npz")
            rows.append(pd.DataFrame(dict(epochs=epochs, Tfin=T, x=q["x"].ravel(), u_exact=q["u_exact"].ravel(),
                                          u_classical=c["u_pred"].ravel(), u_device=q["u_qpu"].ravel(),
                                          u_ideal=q["u_ideal_circuit"].ravel())))
    return pd.concat(rows, ignore_index=True)


def best_110(df):
    """The run of each model and T with the smallest error of nu at 110 epochs: u(x,T) and its training history."""
    best = df[df.epochs == 110]
    best = best.loc[best.groupby(["Tfin", "model"]).eps_nu.idxmin()]
    prof, hist = [], []
    for r in best.itertuples():
        folder, prefix, _ = MODELS[r.model]
        path = INVERSE_BUDGETS[110]["root"] / folder / INV_CASES[r.Tfin]
        t = tag(prefix, r.size, r.seed, 110)
        v, h = np.load(path / f"validation_{t}.npz"), np.load(path / f"losshist_{t}.npz")
        u = v["u_qpu"] if r.model == "hqpinn" else v["u_pred"]
        prof.append(pd.DataFrame(dict(Tfin=r.Tfin, model=r.model, size=r.size, seed=r.seed, nu=r.nu, eps_nu=r.eps_nu, l2=r.l2,
                                      x=v["x"].ravel(), u_exact=v["u_exact"].ravel(), u=u.ravel())))
        hist.append(pd.DataFrame(dict(Tfin=r.Tfin, model=r.model, epoch=np.arange(1, len(h["nu"]) + 1), nu=h["nu"],
                                      loss_residual=h["loss_r"], loss_data=h["loss_d"])))
    return pd.concat(prof, ignore_index=True), pd.concat(hist, ignore_index=True)


def devices(df):
    g = df[df.model == "hqpinn"]
    return g.groupby(["epochs", "backend", "size"]).agg(chains=("chains", "first"), fit_rms=("fit_rms", "first"),
                                                         jobs=("n_jobs", "first"), time_median_s=("time_s", "median"),
                                                         time_max_s=("time_s", "max")).reset_index()


def long_budget():
    """data/Inverso: data only at t = T, 20 000 AdamW + L-BFGS. The two folders differ in circuit, stopping rule and
    number of seeds, so they are tabulated side by side but never tested against each other."""
    rows = []
    for model, folder, pattern in (("classical", "classical_inverse", r"T_([\d.]+)/losshist_(\d+)_INTRA(\d+)_(\d+)"),
                                   ("hqpinn_older", "quantum_inverse", r"T_([\d.]+)/losshist_(\d+)_NQ(\d+)_(\d+)")):
        h = pd.read_csv(INVERSE_LONG / folder / "histories_summary.csv")
        keys = pd.DataFrame([re.match(pattern, f).groups() for f in h.file], columns=["Tfin", "n_collocation", "size", "seed"])
        h = pd.concat([h, keys.astype(dict(Tfin=float, n_collocation=int, size=int, seed=int))], axis=1)
        h["eps_nu"] = (h.nu_final - INV_NU).abs() / INV_NU
        s = h.groupby(["Tfin", "n_collocation", "size"]).agg(runs=("eps_nu", "size"), eps_median=("eps_nu", "median")).reset_index()
        rows.append(s.assign(model=model))
    out = pd.concat(rows, ignore_index=True)
    return out[out["size"].isin(INV_SIZES) & out.n_collocation.isin([10, 100])]


def long_budget_ncol():
    """Long budget: mean squared error of nu, mean over the seeds of (nu - nu_true)^2, per model, T, width and N_col."""
    rows = []
    for model, folder, pattern in (("classical", "classical_inverse", r"T_([\d.]+)/losshist_(\d+)_INTRA(\d+)_(\d+)"),
                                   ("hqpinn_older", "quantum_inverse", r"T_([\d.]+)/losshist_(\d+)_NQ(\d+)_(\d+)")):
        h = pd.read_csv(INVERSE_LONG / folder / "histories_summary.csv")
        keys = pd.DataFrame([re.match(pattern, f).groups() for f in h.file], columns=["Tfin", "n_collocation", "size", "seed"])
        h = pd.concat([h, keys.astype(dict(Tfin=float, n_collocation=int, size=int, seed=int))], axis=1)
        h["se"] = (h.nu_final - INV_NU) ** 2
        g = h.groupby(["Tfin", "size", "n_collocation"])
        rows.append(pd.DataFrame(dict(mse_nu=g.se.mean(), runs=g.size())).reset_index().assign(model=model))
    return pd.concat(rows, ignore_index=True)


def qpu_cases(out):
    rows = []
    for case in INVERSE_QPU_CASES:
        T, n, seed, epochs = case["Tfin"], case["n"], case["seed"], case["epochs"]
        folder = INVERSE_REAL_ROOT / "quantico" / INV_CASES[T]
        run_file = folder / f"run_{tag('NQ', n, seed, epochs)}.json"
        row = dict(case, status="pending", nu="", eps_nu="", l2="", l2_ideal="")
        if run_file.exists():
            run = json.loads(run_file.read_text())
            row.update(status="done", nu=run["nu"], eps_nu=abs(run["nu"] - INV_NU) / INV_NU, l2=run["rel_l2_T_qpu"],
                       l2_ideal=run["rel_l2_T_ideal_circuit"])
            target = out / "qpu" / f"T_{T:g}"
            target.mkdir(parents=True, exist_ok=True)
            hist = np.load(folder / f"losshist_{tag('NQ', n, seed, epochs)}.npz")["nu"]
            val = np.load(folder / f"validation_{tag('NQ', n, seed, epochs)}.npz")
            pd.DataFrame(dict(epoch=np.arange(1, len(hist) + 1), nu=hist)).to_csv(target / "nu.csv", index=False, float_format="%.6g")
            pd.DataFrame(dict(x=val["x"].ravel(), u_qpu=val["u_qpu"].ravel(), u_exact=val["u_exact"].ravel())).to_csv(
                target / "profile.csv", index=False, float_format="%.6g")
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_p(p):
    if p < 1e-3:
        exponent = int(np.floor(np.log10(p)))
        return rf"${p / 10 ** exponent:.0f}\times10^{{{exponent}}}$"
    return f"{p:.2g}"


def triple(row, prefix, scale, digits):
    """median / mean / max of one row."""
    return " / ".join(f"{scale * row[f'{prefix}_{k}']:.{digits}f}" for k in ("median", "mean", "max"))


def write_summary_table(summary, paired_table):
    lines = [r"\begin{tabular}{lcc}", r" & HQPINN$^{b}$ & classical\\", r"\colrule"]
    for epochs, T in ((500, 0.1), (500, 1.0), (110, 0.1), (110, 1.0)):
        s = summary[(summary.epochs == epochs) & (summary.Tfin == T)].set_index("model")
        p = paired_table[(paired_table.epochs == epochs) & (paired_table.Tfin == T)].iloc[0]
        q, c = s.loc["hqpinn"], s.loc["classical"]
        lines += [rf"\multicolumn{{3}}{{c}}{{$T={T:g}$, $K={epochs}$ ({INVERSE_BUDGETS[epochs]['backend']})}}\\", r"\colrule",
                  rf"$\epsilon_\nu$ (\%)$^{{a}}$ & {triple(q, 'eps', 100, 1)} & {triple(c, 'eps', 100, 1)}\\",
                  rf"$\epsilon_{{L_2}}$$^{{a}}$ & {triple(q, 'l2', 1, 3)} & {triple(c, 'l2', 1, 3)}\\",
                  rf"wins ($\epsilon_\nu$; $\epsilon_{{L_2}}$) & {p.nu_wins_hqpinn:.0f}; {p.l2_wins_hqpinn:.0f} & {p.nu_wins_classical:.0f}; {p.l2_wins_classical:.0f}\\",
                  rf"$p$ ($\epsilon_\nu$; $\epsilon_{{L_2}}$) & \multicolumn{{2}}{{c}}{{{fmt_p(p.nu_p)}; {fmt_p(p.l2_p)}}}\\",
                  r"\colrule"]
    lines = lines[:-1] + [r"\end{tabular}"]
    (TABLES / "inverse_summary.tex").write_text("% written by manuscript/scripts/build_inverse_data.py: do not edit\n" + "\n".join(lines) + "\n")


def write_per_size_table(per_size):
    lines = [r"\begin{tabular}{lcccc}", r" & \multicolumn{2}{c}{$T=0.1$} & \multicolumn{2}{c}{$T=1$}\\",
             r"\cline{2-3}\cline{4-5}", r" & $\epsilon_\nu$ (\%) & $\epsilon_{L_2}$ & $\epsilon_\nu$ (\%) & $\epsilon_{L_2}$\\"]
    for epochs in (500, 110):
        lines += [r"\colrule", rf"\multicolumn{{5}}{{c}}{{$K={epochs}$ ({INVERSE_BUDGETS[epochs]['backend']})}}\\", r"\colrule"]
        for model, label in (("hqpinn", "HQPINN"), ("classical", "classical")):
            for size in INV_SIZES:
                cells = []
                for T in (0.1, 1.0):
                    r = per_size[(per_size.epochs == epochs) & (per_size.Tfin == T) & (per_size.model == model)
                                 & (per_size["size"] == size)].iloc[0]
                    cells += [triple(r, "eps", 100, 1), f"{r.l2_median:.3f}"]
                lines.append(rf"{label}, $n={size}$ & " + " & ".join(cells) + r"\\")
    lines.append(r"\end{tabular}")
    (TABLES / "inverse_per_size.tex").write_text("% written by manuscript/scripts/build_inverse_data.py: do not edit\n" + "\n".join(lines) + "\n")


def main():
    out = PAPER_DATA / "inverse"
    out.mkdir(parents=True, exist_ok=True)
    df, nu = load_runs()
    tables = dict(
        runs=df, summary=summary_table(df, ["epochs", "Tfin", "model"]),
        per_size=summary_table(df, ["epochs", "Tfin", "model", "size"]),
        paired=paired(df), budget_effect=budget_effect(df), nu_curves=nu_curves(nu), field=field(), devices=devices(df),
        long_budget=long_budget(), long_budget_ncol=long_budget_ncol(), qpu_cases=qpu_cases(out))
    tables["best_profile"], tables["best_history"] = best_110(df)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
    TABLES.mkdir(parents=True, exist_ok=True)
    write_summary_table(tables["summary"], tables["paired"])
    write_per_size_table(tables["per_size"])


if __name__ == "__main__":
    main()
