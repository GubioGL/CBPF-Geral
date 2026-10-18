"""Regenerate every manuscript figure (no QPU or Aer job is run).

    C:/Users/Gubio/anaconda3/envs/qiskit/python.exe manuscript/scripts/make_all.py [--skip-cache]

1. replay_training.py      bit-exact offline replay of the recorded hardware/emulator runs and of the
                           classical control -> cache/learning_curves.npz           (~20 s)
2. simulate_same_budget.py HQPINN ideal / ideal + shot noise at the hardware budget
                           -> cache/same_budget_sims.npz                           (~10 s)
3. controls_hardware_budget.py  classical LR sweep and sine middle block at the hardware budget
                           -> cache/controls_hardware_budget.npz  (~40 s; reused if present)
4. the figure scripts      -> manuscript/figures/*.pdf and *.png
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = ["replay_training.py", "simulate_same_budget.py", "controls_hardware_budget.py"]
FIGURES = ["fig2_simulations.py", "fig3_hardware.py", "fig4_noise_ladder.py", "fig5_device_model.py",
           "figS1_exact_ncol.py"]

if __name__ == "__main__":
    scripts = FIGURES if "--skip-cache" in sys.argv else CACHE + FIGURES
    for name in scripts:
        print(f"--- {name}", flush=True)
        subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=HERE)
