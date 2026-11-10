"""Build manuscript/data/inverse/: the tables behind the inverse-problem figures and tables.

    python manuscript/scripts/build_inverse_data.py

Reads the runs of data/Inverso_200 (200 epochs, n = 10, seeds 0-9, HQPINN on FakeAachen) and data/Inverso (20 000 epochs,
n = 3, 7, 10, seeds 0-10, HQPINN with the ideal circuit), and writes:

  runs.csv            one row per run (40 + 132 rows), with the squared error of nu and the MSE of u(x,T)
  summary.csv         per budget, T and model: error of nu and of u(x,T) over the runs
  per_size.csv        the same per size n
  paired.csv          per budget and T: paired wins and two-sided Wilcoxon p, HQPINN vs classical
  nu_curves.csv       per emulator budget, T, model and epoch: nu and its relative error over the runs
  field.csv           u(x,T) of the paired run n = 10, seed 0, at each emulator budget
  devices.csv         chains and calibration misfit per backend and n
  size_errors.csv     per budget, T, model and size: MSE of nu (mean over seeds of (nu - nu_true)^2) and median and
                      interquartile range of the MSE of u(x,T) (Appendix C figure, as Figs. 2(a,d))
  best_profile.csv, best_history.csv  best run of each model and T at each budget: u(x,T), and nu and the loss terms
                      per epoch (at 20 000 epochs, medians over logarithmic bins of epochs)
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

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from paper_common import (INV_CASES, INV_N, INV_NU, INVERSE_BUDGETS, INVERSE_IDEAL, INVERSE_QPU_CASES, INVERSE_REAL_ROOT,
                          PAPER_DATA, ROOT)

TABLES = ROOT / "manuscript" / "tables"

MODELS = {"classical": ("classico", "NE", "neurons"), "hqpinn": ("quantico", "NQ", "n_qubits")}
IDEAL = INVERSE_IDEAL["epochs"]
# every budget: its folder, emulator (or "noiseless"), sizes, seeds and the folder of its HQPINN runs
BUDGETS = {**{k: dict(v, quantum="quantico") for k, v in INVERSE_BUDGETS.items()},
           IDEAL: dict(INVERSE_IDEAL, quantum="quantico_ideal")}
EMULATED = list(INVERSE_BUDGETS)


def tag(prefix, size, seed, epochs):
    return f"{INV_N}_{prefix}{size}_{seed}_E{epochs}"


def run_dir(epochs, model, T):
    b = BUDGETS[epochs]
    return b["root"] / (b["quantum"] if model == "hqpinn" else MODELS[model][0]) / INV_CASES[T]


def field_of(model, v):
    """u(x,T) of a run: the HQPINN on an emulator is evaluated with the circuit run on the emulator."""
    return v["u_qpu"] if (model == "hqpinn" and "u_qpu" in v) else v["u_pred"]


def load_runs():
    """One row per run, with the per-epoch nu of each emulator run kept apart (it does not go in the table)."""
    rows, nu = [], {}
    for epochs, budget in BUDGETS.items():
        for T in INV_CASES:
            for model, (_, prefix, size_key) in MODELS.items():
                path = run_dir(epochs, model, T)
                for f in sorted(path.glob(f"run_*_E{epochs}.json")):
                    run = json.loads(f.read_text())
                    size, seed = run[size_key], run["seed"]
                    if size not in budget["sizes"] or seed not in budget["seeds"]:
                        raise ValueError(f"unexpected run {model} size {size} seed {seed} in {path}")
                    v = np.load(path / f"validation_{tag(prefix, size, seed, epochs)}.npz")
                    u = field_of(model, v).ravel()
                    row = dict(epochs=epochs, Tfin=T, model=model, size=size, seed=seed, nu=run["nu"],
                               eps_nu=abs(run["nu"] - INV_NU) / INV_NU, se_nu=(run["nu"] - INV_NU) ** 2,
                               mse_u=float(np.mean((u - v["u_exact"].ravel()) ** 2)), time_s=run["time_s"])
                    if model == "hqpinn" and epochs in EMULATED:
                        row.update(backend=run["backend"], l2=run["rel_l2_T_qpu"],
                                   l2_surrogate=run["rel_l2_T_measured_model"], l2_ideal=run["rel_l2_T_ideal_circuit"],
                                   n_jobs=run["n_jobs"], chains=run["chains"], fit_rms=run["fit_rms_calibration"])
                    else:
                        row.update(backend="ideal" if model == "hqpinn" else "", l2=run["rel_l2_T"])
                    rows.append(row)
                    if epochs in EMULATED:
                        nu[epochs, T, model, size, seed] = np.load(path / f"losshist_{tag(prefix, size, seed, epochs)}.npz")["nu"]
    df = pd.DataFrame(rows)
    for epochs, budget in BUDGETS.items():
        expected = len(budget["sizes"]) * len(budget["seeds"])
        counts = df[df.epochs == epochs].groupby(["Tfin", "model"]).size()
        if len(counts) != 4 or not (counts == expected).all():
            raise ValueError(f"expected {expected} runs per model and T at {epochs} epochs:\n{counts}")
    return df, nu


def summarize(g):
    out = dict(runs=len(g), eps_median=g.eps_nu.median(), eps_mean=g.eps_nu.mean(), eps_p90=g.eps_nu.quantile(0.9),
               eps_max=g.eps_nu.max(), eps_q25=g.eps_nu.quantile(0.25), eps_q75=g.eps_nu.quantile(0.75),
               mse_nu=g.se_nu.mean(), overestimates=(g.nu > INV_NU).mean(), l2_median=g.l2.median(), l2_mean=g.l2.mean(),
               l2_max=g.l2.max(), time_median_s=g.time_s.median())
    if "l2_ideal" in g and g.l2_ideal.notna().any():
        out.update(l2_ideal_median=g.l2_ideal.median(), l2_surrogate_median=g.l2_surrogate.median(),
                   ideal_worse=(g.l2_ideal > g.l2).mean(), ideal_over_device_median=(g.l2_ideal / g.l2).median())
    return out


def summary_table(df, keys):
    """One row per group, in wide format (the emulator-only columns are empty elsewhere)."""
    return pd.DataFrame([dict(zip(keys, k), **summarize(g)) for k, g in df.groupby(keys)])


def compare(a, b, column):
    """a and b aligned on (size, seed): how many times a < b, and the two-sided paired Wilcoxon p."""
    wins = int((a[column].values < b[column].values).sum())
    return wins, int(len(a) - wins), float(wilcoxon(a[column].values, b[column].values).pvalue)


def paired(df):
    rows = []
    for (epochs, T), g in df.groupby(["epochs", "Tfin"]):   # same size, seed, nu_0 and collocation stream
        q = g[g.model == "hqpinn"].sort_values(["size", "seed"])
        c = g[g.model == "classical"].sort_values(["size", "seed"])
        if not (q[["size", "seed"]].values == c[["size", "seed"]].values).all():
            raise ValueError("the runs are not paired")
        wq, wc, p = compare(q, c, "eps_nu")
        wql, wcl, pl = compare(q, c, "l2")
        rows.append(dict(epochs=epochs, Tfin=T, pairs=len(q), nu_wins_hqpinn=wq, nu_wins_classical=wc, nu_p=p,
                         l2_wins_hqpinn=wql, l2_wins_classical=wcl, l2_p=pl))
    return pd.DataFrame(rows)


def nu_curves(nu):
    rows = []
    for epochs in EMULATED:
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
    for epochs in EMULATED:
        for T in INV_CASES:
            q = np.load(run_dir(epochs, "hqpinn", T) / f"validation_{tag('NQ', size, seed, epochs)}.npz")
            c = np.load(run_dir(epochs, "classical", T) / f"validation_{tag('NE', size, seed, epochs)}.npz")
            rows.append(pd.DataFrame(dict(epochs=epochs, Tfin=T, x=q["x"].ravel(), u_exact=q["u_exact"].ravel(),
                                          u_classical=c["u_pred"].ravel(), u_device=q["u_qpu"].ravel(),
                                          u_ideal=q["u_ideal_circuit"].ravel())))
    return pd.concat(rows, ignore_index=True)


def log_bins(values, per_decade=60):
    """Medians of a per-epoch series over logarithmic bins of epochs: single epochs at the start, wide bins at the end
    (the per-epoch loss fluctuates by tens of percent from one batch to the next)."""
    n = len(values)
    edges = np.unique(np.round(np.logspace(0, np.log10(n + 1), int(per_decade * np.log10(n + 1)) + 1)).astype(int))
    rows = [(np.sqrt(a * (b - 1)), np.median(values[a - 1:b - 1])) for a, b in zip(edges[:-1], edges[1:])]
    return np.array(rows).T


def best_runs(df):
    """The run of each budget, model and T with the smallest error of nu: u(x,T) and its training history (per epoch
    for the short budget, medians over logarithmic bins of epochs for the long one)."""
    best = df.loc[df.groupby(["epochs", "Tfin", "model"]).eps_nu.idxmin()]
    prof, hist = [], []
    for r in best.itertuples():
        path = run_dir(r.epochs, r.model, r.Tfin)
        t = tag(MODELS[r.model][1], r.size, r.seed, r.epochs)
        v, h = np.load(path / f"validation_{t}.npz"), np.load(path / f"losshist_{t}.npz")
        prof.append(pd.DataFrame(dict(epochs=r.epochs, Tfin=r.Tfin, model=r.model, size=r.size, seed=r.seed, nu=r.nu,
                                      eps_nu=r.eps_nu, l2=r.l2, x=v["x"].ravel(), u_exact=v["u_exact"].ravel(),
                                      u=field_of(r.model, v).ravel())))
        if len(h["nu"]) > 1000:
            epoch, nu = log_bins(h["nu"])
            lr, ld = log_bins(h["loss_r"])[1], log_bins(h["loss_d"])[1]
        else:
            epoch, nu, lr, ld = np.arange(1, len(h["nu"]) + 1), h["nu"], h["loss_r"], h["loss_d"]
        hist.append(pd.DataFrame(dict(epochs=r.epochs, Tfin=r.Tfin, model=r.model, epoch=epoch, nu=nu, loss_residual=lr,
                                      loss_data=ld)))
    return pd.concat(prof, ignore_index=True), pd.concat(hist, ignore_index=True)


def size_errors(df):
    """Per budget, T, model and size: the MSE of nu, mean over the seeds of (nu - nu_true)^2, and the median and quartiles
    of the MSE of u(x,T) over the seeds (as the median MSE of Figs. 2(a,d))."""
    g = df.groupby(["epochs", "Tfin", "model", "size"])
    return pd.DataFrame(dict(mse_nu=g.se_nu.mean(), eps_median=g.eps_nu.median(), mse_u_median=g.mse_u.median(),
                             mse_u_q25=g.mse_u.quantile(0.25), mse_u_q75=g.mse_u.quantile(0.75), runs=g.size())).reset_index()


def devices(df):
    g = df[(df.model == "hqpinn") & df.epochs.isin(EMULATED)]
    return g.groupby(["epochs", "backend", "size"]).agg(chains=("chains", "first"), fit_rms=("fit_rms", "first"),
                                                         jobs=("n_jobs", "first"), time_median_s=("time_s", "median"),
                                                         time_max_s=("time_s", "max")).reset_index()


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


def budget_label(epochs):
    k = f"{epochs:,}".replace(",", r"\,")
    return rf"$K={k}$ ({BUDGETS[epochs]['backend']})"


def write_summary_table(summary, paired_table):
    lines = [r"\begin{tabular}{lcc}", r" & HQPINN$^{b}$ & classical\\", r"\colrule"]
    for epochs in BUDGETS:
        for T in INV_CASES:
            s = summary[(summary.epochs == epochs) & (summary.Tfin == T)].set_index("model")
            p = paired_table[(paired_table.epochs == epochs) & (paired_table.Tfin == T)].iloc[0]
            q, c = s.loc["hqpinn"], s.loc["classical"]
            sizes = ", ".join(map(str, BUDGETS[epochs]["sizes"]))
            lines += [rf"\multicolumn{{3}}{{c}}{{$T={T:g}$, {budget_label(epochs)}; $n={sizes}$, {p.pairs:.0f} pairs}}\\",
                      r"\colrule",
                      rf"$\epsilon_\nu$ (\%)$^{{a}}$ & {triple(q, 'eps', 100, 1)} & {triple(c, 'eps', 100, 1)}\\",
                      rf"$\epsilon_{{L_2}}$$^{{a}}$ & {triple(q, 'l2', 1, 3)} & {triple(c, 'l2', 1, 3)}\\",
                      rf"wins ($\epsilon_\nu$; $\epsilon_{{L_2}}$) & {p.nu_wins_hqpinn:.0f}; {p.l2_wins_hqpinn:.0f} & {p.nu_wins_classical:.0f}; {p.l2_wins_classical:.0f}\\",
                      rf"$p$ ($\epsilon_\nu$; $\epsilon_{{L_2}}$) & \multicolumn{{2}}{{c}}{{{fmt_p(p.nu_p)}; {fmt_p(p.l2_p)}}}\\",
                      r"\colrule"]
    lines = lines[:-1] + [r"\end{tabular}"]
    (TABLES / "inverse_summary.tex").write_text("% written by manuscript/scripts/build_inverse_data.py: do not edit\n" + "\n".join(lines) + "\n")


def write_per_size_table(per_size):
    """Only the long budget has several sizes."""
    lines = [r"\begin{tabular}{lcccc}", r" & \multicolumn{2}{c}{$T=0.1$} & \multicolumn{2}{c}{$T=1$}\\",
             r"\cline{2-3}\cline{4-5}", r" & $\epsilon_\nu$ (\%) & $\epsilon_{L_2}$ & $\epsilon_\nu$ (\%) & $\epsilon_{L_2}$\\"]
    for epochs in (e for e, b in BUDGETS.items() if len(b["sizes"]) > 1):
        lines += [r"\colrule", rf"\multicolumn{{5}}{{c}}{{{budget_label(epochs)}}}\\", r"\colrule"]
        for model, label in (("hqpinn", "HQPINN"), ("classical", "classical")):
            for size in BUDGETS[epochs]["sizes"]:
                cells = []
                for T in INV_CASES:
                    r = per_size[(per_size.epochs == epochs) & (per_size.Tfin == T) & (per_size.model == model)
                                 & (per_size["size"] == size)].iloc[0]
                    cells += [triple(r, "eps", 100, 1), f"{r.l2_median:.3f}"]
                lines.append(rf"{label}, $n={size}$ & " + " & ".join(cells) + r"\\")
    lines.append(r"\end{tabular}")
    (TABLES / "inverse_per_size.tex").write_text("% written by manuscript/scripts/build_inverse_data.py: do not edit\n" + "\n".join(lines) + "\n")


def main():
    out = PAPER_DATA / "inverse"
    out.mkdir(parents=True, exist_ok=True)
    for old in ("budget_effect.csv", "mse_size.csv"):          # tables of earlier versions
        (out / old).unlink(missing_ok=True)
    df, nu = load_runs()
    tables = dict(
        runs=df, summary=summary_table(df, ["epochs", "Tfin", "model"]),
        per_size=summary_table(df, ["epochs", "Tfin", "model", "size"]),
        paired=paired(df), nu_curves=nu_curves(nu), field=field(), devices=devices(df), size_errors=size_errors(df),
        qpu_cases=qpu_cases(out))
    tables["best_profile"], tables["best_history"] = best_runs(df)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
    TABLES.mkdir(parents=True, exist_ok=True)
    write_summary_table(tables["summary"], tables["paired"])
    write_per_size_table(tables["per_size"])


if __name__ == "__main__":
    main()
