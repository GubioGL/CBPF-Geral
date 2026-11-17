"""Where the chains of the inverse runs sit on the chip: ibm_aachen (Heron r3, 156 qubits), from the error table of its
fake (FakeAachen). (a) find_chains' greedy choice, the 10 chains of the emulator runs (data/Inverso_200); (b) the most
chains that fit (pack_chains, 12 chains of 10 qubits), the choice of the real-QPU run (REAL_PACKING = "max"). Every chain
is a path of n qubits linked by CZ; the circuit runs on all chains at once (one packed circuit).
On the real QPU the chains are chosen from the calibration of the day, so they can differ from these.

    python run/plot_qpu_chains.py            # -> docs/qpu_chains_ibm_aachen.pdf (and .png)
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

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estimate_qpu_inverse import H  # noqa: E402  (H = Inverser_quantum_hardware_pinn_E110, with libgomp preloaded)
from Inverser_quantum_hardware_pinn import find_chains  # noqa: E402

ROW, GAP = 16, 4      # Heron r3 156: 8 rows of 16 qubits (20 r .. 20 r + 15), 4 bridge qubits between two rows


def coordinates(backend):
    edges = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
    xy = {}
    for q in range(backend.num_qubits):
        r, k = divmod(q, ROW + GAP)
        if k < ROW:
            xy[q] = (k, -r)
    for q in range(backend.num_qubits):
        r, k = divmod(q, ROW + GAP)
        if k >= ROW:
            above = next(a for a, b in edges if b == q and a // (ROW + GAP) == r and a % (ROW + GAP) < ROW)
            xy[q] = (xy[above][0], -r - 0.5)
    return xy, edges


def draw(ax, backend, chains, scores, title):
    xy, edges = coordinates(backend)
    on = {q: i for i, c in enumerate(chains) for q in c}
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9]]
    for a, b in edges:
        ax.plot(*zip(xy[a], xy[b]), color="#d6d6d6", lw=1, zorder=1)
    for i, c in enumerate(chains):
        pts = np.array([xy[q] for q in c])
        ax.plot(pts[:, 0], pts[:, 1], color=colors[i], lw=4, solid_capstyle="round", zorder=2)
        mid = pts[len(pts) // 2]
        ax.text(mid[0], mid[1] + 0.28, f"{i + 1}", ha="center", va="bottom", fontsize=7, color=colors[i], weight="bold")
    for q, (x, y) in xy.items():
        ax.scatter(x, y, s=55, color=colors[on[q]] if q in on else "white", edgecolor="#555" if q not in on else "none",
                   lw=0.6, zorder=3)
        ax.text(x, y, str(q), ha="center", va="center", fontsize=4, color="white" if q in on else "#555", zorder=4)
    used = sum(map(len, chains))
    ax.set_title(f"{title}\n{len(chains)} chains x {len(chains[0])} qubits = {used} of {backend.num_qubits} qubits; "
                 f"error score {min(scores):.3f}-{max(scores):.3f}", fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--qubits", type=int, default=10)
    args = p.parse_args()
    backend = H.get_backend(False, "ibm_aachen")
    n, limit = args.qubits, H.MAX_CHAIN_SCORE_PER_QUBIT * args.qubits
    greedy, gs, _ = find_chains(backend, n)
    greedy, gs = zip(*[(c, s) for c, s in zip(greedy, gs) if s <= limit])
    packed, ps = H.pack_chains(backend, n, limit)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    draw(axes[0], backend, list(greedy), gs, "(a) greedy choice (emulator runs, data/Inverso_200)")
    draw(axes[1], backend, packed, ps, "(b) most chains that fit (real-QPU run)")
    fig.suptitle(f"ibm_aachen (Heron r3, from FakeAachen): chains of {n} qubits for the packed circuit", fontsize=10)
    os.makedirs("docs", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join("docs", f"qpu_chains_ibm_aachen.{ext}"), dpi=160)
    print("-> docs/qpu_chains_ibm_aachen.pdf / .png")


if __name__ == "__main__":
    main()
