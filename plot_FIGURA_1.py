import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no window: figures are only saved
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  -- registers the 'science' styles

ROOT = Path(__file__).resolve().parent
TFIN, SEED, WIDTH, EPOCHS = 0.1, 0, 10, 110
CLASSICAL = ROOT / "data" / "Forward" / "classical_vs_qpu" / f"T_{TFIN}" / f"validation_10_NEURO110{WIDTH}_tanh_{SEED}_E{EPOCHS}.npz"
FAKE_TRAINED = ROOT / "data" / "quantum_ibm_faker" / f"T_{TFIN}" / f"prediction_real_10_NQ10_{SEED}_fake_berlin_HW{EPOCHS}.npz"
REAL_TRAINED = ROOT / "data" / "quantum_forward_hardware_real" / f"T_{TFIN}" / f"validation_10_NQ10_{SEED}_ibm_berlin_none_HW{EPOCHS}.npz"
OUT = ROOT / f"FIGURA_1_{SEED}_E{EPOCHS}"


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

    # LaTeX only if it is installed; otherwise the same style with mathtext
    plt.style.use(["science"] if shutil.which("latex") else ["science", "no-latex"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    
    ax.plot(x, u_ref, "-", color="black", lw=1.6, label="FDM (reference)", zorder=1)
    ax.plot(x, u_classical, "--", color="tab:blue", lw=1.4,
            label=f" PINN ({WIDTH} neurons)", zorder=2)
    ax.plot(x, u_fake_trained, "o", color="tab:green", ms=2.0, mew=0, markevery=2,
            label=f"Quantum HPINN, trained on FakeBerlin, prediction on ibm\\_berlin, "
                  if shutil.which("latex") else f"Quantum HPINN, trained on FakeBerlin, prediction on ibm_berlin, ", zorder=3)
    ax.plot(x, u_real_trained, "*", color="tab:red", ms=5.0, mew=0, markevery=(1, 8),
            label=(r"Quantum PINN, fully trained on ibm\_berlin" if shutil.which("latex") else
                   "Quantum PINN, fully trained on ibm_berlin") ,
            zorder=4)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"$u(x, t={TFIN})$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.15, 1.15)
    # outside the axes, above them: inside it would cover the minimum of the solution
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=7, frameon=False, ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=600)
    print(f"MSE vs FDM: clássica {mse(u_classical, u_ref):.3e} | treinada no fake -> QPU real "
          f"{mse(u_fake_trained, u_ref):.3e} | treinada no QPU real {mse(u_real_trained, u_ref):.3e}")
    print(f"salvo em {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
