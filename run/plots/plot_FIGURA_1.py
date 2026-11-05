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

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no window: figures are only saved
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  -- registers the 'science' styles

ROOT = _ROOT
TFIN, SEED, WIDTH, EPOCHS = 0.1, 0, 10, 110
CLASSICAL = ROOT / "data" / "Forward" / "classical_vs_qpu" / f"T_{TFIN}" / f"validation_10_NEURO110{WIDTH}_tanh_{SEED}_E{EPOCHS}.npz"
FAKE_TRAINED = ROOT / "data" / "quantum_ibm_faker" / f"T_{TFIN}" / f"prediction_real_10_NQ10_{SEED}_fake_berlin_HW{EPOCHS}.npz"
REAL_TRAINED = ROOT / "data" / "quantum_forward_hardware_real" / f"T_{TFIN}" / f"validation_10_NQ10_{SEED}_ibm_berlin_none_HW{EPOCHS}.npz"
# per-epoch loss of each training run (the batch the optimizer saw: classical net / fake QPU / real QPU)
CLASSICAL_HIST = CLASSICAL.with_name(f"history_10_NEURO110{WIDTH}_tanh_{SEED}_E{EPOCHS}.npz")
FAKE_HIST = FAKE_TRAINED.with_name(f"history_10_NQ10_{SEED}_fake_berlin_HW{EPOCHS}.npz")
REAL_HIST = REAL_TRAINED.with_name(f"history_10_NQ10_{SEED}_ibm_berlin_none_HW{EPOCHS}.npz")
OUT = ROOT / "paper" / "figures" / f"FIGURA_1_{SEED}_E{EPOCHS}"


def mse(u, ref):
    return float(np.mean((np.ravel(u) - np.ravel(ref)) ** 2))


def main():
    classical, fake_trained, real_trained = np.load(CLASSICAL), np.load(FAKE_TRAINED), np.load(REAL_TRAINED)
    x, u_ref = fake_trained["x"].ravel(), fake_trained["u_reference"].ravel()
    for name, d in (("clássica", classical), ("treinada no QPU real", real_trained)):
        if not (np.allclose(d["x"].ravel(), x) and np.allclose(d["u_reference"].ravel(), u_ref)):
            raise SystemExit(f"Rede {name}: grade x ou referência FDM diferente da predição no QPU.")
    u_classical, u_fake_trained, u_real_trained = (classical["u"].ravel(), fake_trained["u_real"].ravel(),
                                                   real_trained["u_qpu"].ravel())
    loss_classical, loss_fake, loss_real = (np.load(CLASSICAL_HIST)["loss_train"], np.load(FAKE_HIST)["loss_qpu"],
                                            np.load(REAL_HIST)["loss_qpu"])

    # LaTeX only if it is installed; otherwise the same style with mathtext
    plt.style.use(["science"] if shutil.which("latex") else ["science", "no-latex"])
    fig, (ax, ax_loss) = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw=dict(width_ratios=(1.0, 0.8)))

    ax.plot(x, u_ref, "-", color="black", lw=1.6, label="FDM (reference)", zorder=1)
    ax.plot(x, u_classical, "--", color="tab:blue", lw=1.4,
            label=f" PINN ({WIDTH} neurons)", zorder=2)
    ax.plot(x, u_fake_trained, "o", color="tab:green", ms=2.0, mew=0, markevery=2,
            label=f"Quantum HPINN, trained on FakeBerlin, prediction on ibm\\_berlin, "
                  if shutil.which("latex") else f"Quantum HPINN, trained on FakeBerlin, prediction on ibm_berlin, ", zorder=3)
    ax.plot(x, u_real_trained, "*", color="tab:red", ms=5.0, mew=0, markevery=(1, 8),
            label=(r"Quantum HPINN, fully trained on ibm\_berlin" if shutil.which("latex") else
                   "Quantum HPINN, fully trained on ibm_berlin") ,
            zorder=4)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"$u(x, t={TFIN})$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.15, 1.15)
    # outside the axes, above them: inside it would cover the minimum of the solution
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=7, frameon=False, ncol=1)

    # (b) training loss per epoch, same color/marker per model as in (a)
    epochs = np.arange(1, EPOCHS + 1)
    berlin = r"ibm\_berlin" if shutil.which("latex") else "ibm_berlin"
    ax_loss.plot(epochs, loss_classical, "--", color="tab:blue", lw=1.2, label=f"PINN ({WIDTH} neurons)")
    ax_loss.plot(epochs, loss_fake, "-", color="tab:green", lw=1.0, marker="o", ms=3.0, mew=0, markevery=10,
                 label="Quantum HPINN, FakeBerlin")
    ax_loss.plot(epochs, loss_real, "-", color="tab:red", lw=1.0, marker="*", ms=5.0, mew=0, markevery=(5, 10),
                 label=f"Quantum HPINN, {berlin}")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss")
    ax_loss.set_xlim(1, EPOCHS)
    ax_loss.legend(loc="upper right", fontsize=7, frameon=False)
    for a, tag in ((ax, "(a)"), (ax_loss, "(b)")):
        a.text(-0.02, 1.02, tag, transform=a.transAxes, ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=600)
    print(f"MSE vs FDM: clássica {mse(u_classical, u_ref):.3e} | treinada no fake -> QPU real "
          f"{mse(u_fake_trained, u_ref):.3e} | treinada no QPU real {mse(u_real_trained, u_ref):.3e}")
    print(f"loss na última época: clássica {loss_classical[-1]:.3e} | FakeBerlin {loss_fake[-1]:.3e} | "
          f"ibm_berlin {loss_real[-1]:.3e}")
    print(f"salvo em {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
