# Raw results

Nothing in this folder goes to Overleaf: the manuscript reads `manuscript/data/`, a few small tables written from here by
`manuscript/scripts/build_paper_data.py` and `build_inverse_data.py`.

## Used by the paper

| Folder | Content | Used in |
|---|---|---|
| `Forward/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/` | same-protocol sweep, both models, n = 2..15, N_col = 10 and 100 | Figs. 2, 3 |
| `Forward/classical_same_protocol_ncol/` | classical PINN, n = 3, 5, 7, 10, 15 at N_col = 20..90, same setup as the HQPINN runs, from `run/run_classical_ncol.sh`. `raw_runs/` is what lets the script resume | Fig. 3 |
| `Forward/CBPF_quantum_exact_results_Brickwork3/`, `__Brickwork5/`, `_fixed_BrickWork7/`, `_fixed_BrickWork10/`, `_fixed_BrickWork15/` | noiseless HQPINN versus N_col | Fig. 3 |
| `quantum_forward_hardware_real/T_0.1/` | the training run on `ibm_berlin`, complete: it is the record of the experiment. The run at T = 1 will go to `T_1.0/` | Fig. 4, chip map, device table |
| `Forward/classical_vs_qpu/T_0.1/` | classical control with the QPU protocol | Fig. 4 |
| `derived/` | replayed learning curves, same-budget simulations, hardware-budget controls. **Record of the author's Windows machine: do not regenerate on another platform** (`manuscript/README.md`) | Fig. 4, Appendix B |
| `quantum_ibm_faker/T_0.1/` | log of the FakeBerlin run with seed 0 and of its prediction on `ibm_berlin` (restored from the git history, commit `3298f9a`). The raw history of that run and the seeds 1–4 are not available | Appendix B |
| `Inverso_500/` | **inverse problem, 500 epochs, FakeAachen**, 33 runs per model and T (see its README) | Figs. 5, 6, Tables II, X |
| `Inverso_110/` | inverse problem, 110 epochs, FakeBerlin, 33 runs per model and T (see its README) | Figs. 6, 7, 9, Tables II, X |
| `Inverso/` | older long-budget sweeps, another protocol: `classical_inverse` (50 seeds), `quantum_inverse` (10 seeds, linear CNOT circuit). `histories_summary.csv` has the final values | Appendix C only |

A real-QPU run of the inverse problem will go to `Inverso_real/<backend>/` (it does not exist yet).

## Not used by the paper (archive)

- `Forward/classical/`, `Forward/classical2/`, `Forward/quantum/`, `Forward/CBPF_quantum_exact_results_3_Brickwork/`, `_10_Brickwork/`: older sweeps.
  **Reminder:** `Forward/classical*` is not comparable with the HQPINN runs (other stopping monitor and, for n != 10, another middle block). See `docs/manuscript_notes_pt.md`.
- `Inverso_110/{classico,quantico}/T_0.1/` and `T_1.0/`: the earlier case with ν = 0.1 (the hyperparameters were selected on it).
- `quantum_ibm_faker/T_1.0/`: previews of tests with n = 3 on FakeBerlin at T = 1.
- `Dados1Qubit.csv`: one-qubit data from the first tests.
