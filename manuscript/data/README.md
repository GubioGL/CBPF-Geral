# Tables behind the figures

The only data that goes to Overleaf: small tables, one set per figure. Written by `manuscript/scripts/build_paper_data.py`
and `build_inverse_data.py` from the raw results in `data/`; do not edit them by hand.
Regenerate tables and figures with `python manuscript/scripts/make_all.py`.

## Forward problem

| File | Figure | Content |
|---|---|---|
| `forward/simulation/width_sweep.csv` | Fig. 2 | classical PINN and noiseless HQPINN, n = 2..15, N_col = 10 and 100, T = 0.1 and 1 |
| `forward/simulation/points_sweep.csv` | Fig. 3 | n = 3, 5, 7, 10, 15 versus N_col (both models, N_col = 10..100) |
| `forward/qpu/cases.csv` | Fig. 4 | the single QPU case of each horizon, with its status (`done` or `pending`) |
| `forward/qpu/T_0.1/profile.csv` | Fig. 4(a) | u(x,T) of the reference, initial network, QPU run and classical control |
| `forward/qpu/T_0.1/learning_curve.csv` | Fig. 4(b) | MSE per epoch: QPU run, classical control, noiseless HQPINN |
| `forward/qpu/T_0.1/chains.csv` | Fig. 8 (chip map) | the 8 chains of 10 qubits on `ibm_berlin` |
| `forward/qpu/T_1/profile.csv` | Fig. 4(c) | reference only, until the QPU run at T = 1 exists |

Simulation columns: median and quartiles of the MSE, mean and standard deviation of the steps, over the 50 seeds of each
configuration. A run that misses the target stops at the step budget, so `steps_mean` is a lower bound of the steps needed;
`reached_fraction` says how many runs reached it.

## Inverse problem

| File | Used by | Content |
|---|---|---|
| `inverse/runs.csv` | Figs. 5, 6, 9 | one row per run: budget, T, model, size, seed, nu, errors, time (264 rows) |
| `inverse/nu_curves.csv` | Figs. 5, 7, 9 | per budget, T, model and epoch: nu and its relative error over the 33 runs |
| `inverse/field.csv` | Figs. 6, 7 | u(x,T) of the paired run n = 10, seed 0, at each budget |
| `inverse/summary.csv`, `paired.csv` | Table II | errors over the 33 runs; paired wins and Wilcoxon p, HQPINN vs classical |
| `inverse/per_size.csv` | Table X | the same errors per size |
| `inverse/budget_effect.csv` | text, Sec. IV A | 110 vs 500 epochs, same size and seed. For the HQPINN the backend changes too |
| `inverse/devices.csv` | Table IX | chains, calibration misfit, jobs and time per backend and size |
| `inverse/long_budget.csv` | Appendix C | older long-budget sweeps, another protocol: never compared with the runs above |
| `inverse/qpu_cases.csv` | Fig. 7 | the real-QPU cases and their status. All `pending`: no real run exists yet |

`manuscript/tables/inverse_summary.tex` and `inverse_per_size.tex` are written from the same data and `\input` by `main.tex`.
