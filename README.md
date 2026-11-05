# HQPINN: hybrid quantum physics-informed neural networks for the viscous Burgers equation

Code, data and manuscript for a **hybrid quantum PINN (HQPINN)**, a PINN with a fixed (non-variational) quantum layer, applied to
u_t + u u_x = ν u_xx, with ν = 0.01, on the unit periodic interval. Both problems are here, compared against a classical PINN of the
same form:

- **forward problem:** ν is known and u is sought;
- **inverse problem:** ν is unknown and is learned together with u.

This repository merges `forward-problem-hqpinn` and `inverse-problem-hqpinn` (see *History*).

## Main results

| | Simulation | Real hardware |
|---|---|---|
| **Forward** | Same protocol, 50 seeds. T = 0.1: same error, about half of the optimizer steps, n = 10 reaches the target in 100 % of the runs against 64 % (N_col = 100). T = 1, n = 10: MSE 4.8×10⁻⁵ against 3.0×10⁻⁴ | `ibm_berlin` in the loop, T = 0.1, 110 epochs, no mitigation: relative L2 error 5.7 %; classical median 24.7 %. **T = 1: pending** |
| **Inverse** | Noisy emulators, 33 paired runs. Median error of ν, HQPINN vs classical, in 500 epochs: T = 0.1, 2.8 % vs 2.3 % (tie, p = 0.76); T = 1, 30.5 % vs 20.9 % (p = 0.37). The classical PINN reconstructs u(x,T) better in all cases | **No run yet** |

The noiseless quantum layer is classically computable, so the simulations measure an architecture, not a quantum advantage.

## Layout

```text
CBPF-Geral/
├── src/                     importable library
│   ├── main/                PINN models, quantum circuits, physics, training loops
│   ├── Inverser_quantum_hardware_pinn.py       hardware machinery (qubit chains, calibration surrogate)
│   ├── Inverser_quantum_exact_pinn_paralelo.py exact quantum layer and finite-difference residual
│   ├── quantum_hardware/    local helpers (git-ignored): find_chains, IBM instance
│   └── paths.py             repository paths + save_figure()
├── run/                     entry points
│   ├── run_*.py, search_optimizer_ibm_faker.py, run_classical_ncol.sh    forward problem
│   ├── Inverser_*.py, diagnostico_checkpoints.py                         inverse problem
│   └── plots/               exploratory plots (plots/legacy: older classical data, see the reminder below)
├── data/                    raw results (data/README.md says what each folder is and what the paper uses)
├── manuscript/              the paper: main.tex, references.bib, figures/, tables/, data/, scripts/ (manuscript/README.md)
├── docs/                    working notes in Portuguese (manuscript notes, older results report)
├── archive/git_bundles/     git history of the two merged repositories (local, git-ignored)
└── requirements.txt, requirements-lock.txt
```

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt       # requirements-lock.txt: the exact versions on the author's Linux machine

python manuscript/scripts/make_all.py           # tables and figures of the paper
bash manuscript/build.sh                        # main.pdf (needs revtex4-2 and tikz; see manuscript/README.md)
```

Scripts can be started from any folder: each one puts `src/` and `run/` on the import path and resolves `data/...` from the root.

**The figures read `data/derived/`, which is a record of the author's Windows machine, not a cache.** The training is chaotic at the level
of rounding errors, so those results change on another platform. `make_all.py` does not touch them unless you pass `--refresh-derived`.

## Running the inverse problem

Each script trains 3, 7 and 10 neurons (classical) or qubits (quantum) × seeds 0–10, with 10 random collocation points per epoch, and
writes to `<out-dir>/<classico|quantico>/T_<T>_nu_0.01/` (one `run_*.json`, `losshist_*.npz`, `validation_*.npz` and `model_*.pt` per run, plus `summary.json`).

**Classical** (seconds per run):

```bash
python run/Inverser_classical_pinn_E110.py --tfin 0.1 --nu 0.01                                    # 110 epochs -> data/Inverso_110
python run/Inverser_classical_pinn_E110.py --tfin 1 --nu 0.01
python run/Inverser_classical_pinn_E110.py --tfin 0.1 --nu 0.01 --epochs 500 --out-dir data/Inverso_500   # 500 epochs
```

**Quantum on a noisy emulator** (minutes per run; the runs go in parallel). In a pip `.venv` on Linux, `qiskit_aer` and `torch` need their `libgomp`
preloaded (otherwise: *cannot allocate memory in static TLS block*), which `run/env.sh` does:

```bash
source run/env.sh
python run/Inverser_quantum_hardware_pinn_E110.py --tfin 0.1 --nu 0.01 --workers 12                 # FakeBerlin, 110 epochs
python run/Inverser_quantum_hardware_pinn_E110.py --tfin 0.1 --nu 0.01 --epochs 500 --backend ibm_aachen --out-dir data/Inverso_500 --workers 12
```

`--workers 12` fits a 16-core, 16 GB machine. The classical runs reproduce bit by bit on the same machine; the emulator does not fix its seed.

**What changes at T = 1.** With ν = 0.01 a shock forms before T = 1. For this case only (`ADJUSTED_CASES` in `run/Inverser_classical_pinn_E110.py`),
both scripts apply the same three adjustments, with no extra flag: the initial condition as a soft loss, 10 warm-up epochs on the data only with ν frozen,
and a plateau learning rate. `--treino-antigo` runs T = 1 without them. Why these: `data/Inverso_110/README.md`.

**Real QPU** (not run yet): `--real` writes to `data/Inverso_real/<backend>/` and never over the emulator runs; `manuscript/scripts/build_inverse_data.py` picks up
the cases of `INVERSE_QPU_CASES` by themselves. Before running, see the open choices in Appendix D of the paper: the default mitigation of the script is `suppress`, whereas the forward
run used none.

## Running the forward problem

```bash
python run/run_qubit_sweep.py --fixed-validation --help      # same-protocol sweep
bash run/run_classical_ncol.sh                               # classical PINN at N_col = 20..90 (long, parallel, resumable)
source run/env.sh                                            # before the scripts that use qiskit_aer
python run/run_quantum_hardware_real.py --plan --seed 0      # what a real-QPU run would do, without spending a job
```

**Reminder asked by the author:** `data/Forward/classical*` stops on another monitor and, for n != 10, has another middle block. Do not plot it next to the HQPINN
runs (`docs/manuscript_notes_pt.md`). Only `run/plots/legacy/` uses it.

## Paper and Overleaf

To use Overleaf, upload the zip, not the repository:

```bash
python manuscript/scripts/make_overleaf_zip.py    # manuscript/manuscript_overleaf.zip: main.tex at the root, about 30 files, under 1 MB
```

In Overleaf: *New project -> Upload project*, compiler pdfLaTeX. **Never upload the whole repository:** `data/` alone has more than 12 000 files and 4 GB, and the manuscript
does not read it (it reads `manuscript/data/` and `manuscript/tables/`). There is no automatic sync: after changing `manuscript/` here, make the zip again, and edits made in Overleaf have to be brought back by hand.

## History

The forward and inverse problems were split from this repository at commit `5776ab2`, developed in
[GubioGL/forward-problem-hqpinn](https://github.com/GubioGL/forward-problem-hqpinn) and [GubioGL/inverse-problem-hqpinn](https://github.com/GubioGL/inverse-problem-hqpinn), and merged back here.
Their full history is on GitHub and in `archive/git_bundles/` (`git clone archive/git_bundles/<name>.bundle`). Commit hashes quoted in the notes, such as `552d791`, belong to those repositories.
