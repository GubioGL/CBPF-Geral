"""Chip map (appendix) -- the 8 disjoint 10-qubit chains used on the 120-qubit lattice of ibm_berlin.

Data: manuscript/data/forward/qpu/T_0.1/chains.csv.
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

import numpy as np

from paper_common import COL1, COLS, C_QPU, GRID, INK, SURFACE, panel_label, read_table, save, set_style

import matplotlib.pyplot as plt

ROWS = 12
C_QPU_TINT = "#f3a582"       # lighter step of the orange hue: alternate chains in the map


def pos(q):
    return q % COLS, -(q // COLS)


def main():
    set_style()
    chains = read_table("forward/qpu/T_0.1/chains.csv")
    fig, ax = plt.subplots(figsize=(COL1 * 0.72, COL1 * 0.82), constrained_layout=True)
    for q in range(ROWS * COLS):           # couplers of the full lattice
        r, c = divmod(q, COLS)
        if c + 1 < COLS:
            ax.plot(*zip(pos(q), pos(q + 1)), color=GRID, lw=0.8, zorder=1)
        if r + 1 < ROWS:
            ax.plot(*zip(pos(q), pos(q + COLS)), color=GRID, lw=0.8, zorder=1)
    used = set(chains.qubit)
    idle = np.array([pos(q) for q in range(ROWS * COLS) if q not in used])
    ax.scatter(idle[:, 0], idle[:, 1], s=7, color="#d4d3cf", zorder=3, lw=0)
    for k, g in chains.groupby("chain"):
        xy = np.array([pos(q) for q in g.qubit])
        color = C_QPU if k % 2 == 1 else C_QPU_TINT
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.6, zorder=2, solid_capstyle="round")
        ax.scatter(xy[:, 0], xy[:, 1], s=11, color=color, zorder=4, edgecolor=SURFACE, lw=0.5)
        mid = xy[len(xy) // 2 - 1:len(xy) // 2 + 1].mean(0)
        ax.text(*mid, str(k), fontsize=5.2, color=INK, ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.12", fc=SURFACE, ec="none"))
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, COLS - 0.4)
    ax.set_ylim(-ROWS + 0.4, 0.6)
    ax.axis("off")
    save(fig, "figB1_chip")


if __name__ == "__main__":
    main()
