# Raw results

Nothing in this folder goes to Overleaf: the manuscript reads `manuscript/data/`, a few small tables written from here by
`manuscript/scripts/build_paper_data.py` and `build_inverse_data.py`.

## Used by the paper

| Folder | Content | Used in |
|---|---|---|
| `Forward/CBPF_qubit_sweep_quantum_classical_fixed_validation_results/` | same-protocol sweep, both models, n = 2..15, N_col = 10 and 100 | Figs. 2, 3 |
| `Forward/classical_same_protocol_ncol/` | classical PINN, n = 3, 5, 7, 10, 15 at N_col = 20..90, same setup as the HQPINN runs, from `run/run_classical_ncol.sh`. `raw_runs/` is what lets the script resume | Fig. 3 |
| `Forward/CBPF_quantum_exact_results_Brickwork3/`, `__Brickwork5/`, `_fixed_BrickWork7/`, `_fixed_BrickWork10/`, `_fixed_BrickWork15/` | noiseless HQPINN versus N_col | Fig. 3 |
| `quantum_forward_hardware_real/T_0.1/` | the training run on `ibm_berlin` (raw files removed in commit `1c2412dd5`; recover them from its parent, `git checkout 1c2412dd5~1 -- data/quantum_forward_hardware_real/T_0.1`). Compiled in `forward/quantico/qpu_ibm_berlin_T_0.1_*.csv` | Fig. 4, chip map, device table |
| `quantum_forward_hardware_real/T_1/` | real-QPU run on `ibm_aachen`, T = 1: HQPINN, 8 replicas of 10 qubits, seed 2, N_col = 128, soft IC, AdamW + L-BFGS, 500 epochs, from `run/run_quantumIBMTrain_vol2_parallel8.ipynb` (with its matched classical PINN in `classical_comparison/`). Compiled in `forward/quantico/qpu_ibm_aachen_T_1_*.csv` (resumo, perfil, cadeias, historico). Raw files removed after compiling; recover them with `git checkout e2be4d229 -- data/quantum_forward_hardware_real/T_1` | -- |
| `quantum_forward_hardware/T_1.0/` | **simulations only** (FakeAachen emulator, `run/run_quantum_hardware.py`), not a QPU run | -- |
| `Forward/classical_vs_qpu/T_0.1/` | classical control with the QPU protocol | Fig. 4 |
| `forward/quantico/qpu_ibm_berlin_T_0.1_derivados/` (formerly `derived/`) | of the T = 0.1 `ibm_berlin` run: replayed learning curves (`learning_curves.npz`), same-budget simulations (`same_budget_sims.npz`), hardware-budget controls (`controls_hardware_budget.npz`). Input of `qpu_ibm_berlin_T_0.1_curva.csv`; the rest (fake curves, per-epoch profiles, controls) is in no CSV. **Record of the author's Windows machine: do not regenerate on another platform** | Fig. 4, Appendix B |
| `quantum_ibm_faker/T_0.1/` | log of the FakeBerlin run with seed 0 and of its prediction on `ibm_berlin` (restored from the git history, commit `daebd09`). The raw history of that run and the seeds 1–4 are not available | Appendix B |
| `Inverso_200/` | **inverse problem, 200 epochs, HQPINN on FakeAachen**: n = 10, seeds 0-9, 10 runs per model and T (see its README) | Fig. 5, Tables II, IX |
| `Inverso/` | **inverse problem, 20 000 epochs, ideal circuit**: same training, n = 3, 7, 10, seeds 0-10, 33 runs per model and T (`classico/`, `quantico_ideal/`; see its README) | Figs. 8, 9, Tables II, X |
| ~~`Inverso_110/`~~, ~~`Inverso_500/`~~ | inverse problem, 110 epochs on FakeBerlin and 500 on FakeAachen. **Deleted by the author on 09/11/2026**; they are in the git history (commit `a36240c96`) and are no longer in the paper | -- |

A real-QPU run of the inverse problem will go to `Inverso_real/<backend>/` (it does not exist yet).

The older long-budget sweeps of the inverse problem (another protocol, formerly in `Inverso/`) were deleted on 09/11/2026 at the author's request; they are in the git history (commit `a36240c96`). The old scripts that wrote them now point to `Inverso_antigo/`, so that they cannot mix with the new runs.

## Not used by the paper (archive)

- `Forward/classical/`, `Forward/classical2/`, `Forward/quantum/`, `Forward/CBPF_quantum_exact_results_3_Brickwork/`, `_10_Brickwork/`: older sweeps.
  **Reminder:** `Forward/classical*` is not comparable with the HQPINN runs (other stopping monitor and, for n != 10, another middle block). See `docs/manuscript_notes_pt.md`.
- `quantum_ibm_faker/T_1.0/`: previews of tests with n = 3 on FakeBerlin at T = 1.
- `Dados1Qubit.csv`: one-qubit data from the first tests.
