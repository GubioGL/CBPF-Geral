"""
Pure numerical baseline: finite-difference solution of the 1D viscous
Burgers equation. No neural network is involved. This is the ground truth
that run_classical.py, run_quantum.py, and run_compare.py train against.

Run with:
    python run_numerical.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from functions import reference_at_time, solve_burgers_fd, plot_solution

# ---- physical parameters ----------------------------------------------
L, Tfin, nu = 1.0, 0.1, 0.01
Nx, dt, scheme = 512, 1.0e-4, "upwind"

if __name__ == "__main__":
    x_ref, t_ref, U_ref = solve_burgers_fd(L=L, Tfin=Tfin, nu=nu, Nx=Nx, dt=dt, scheme=scheme)

    x_plot = np.linspace(0.0, L, 200).reshape(-1, 1)
    times_to_plot = [0.0, 0.25 * Tfin, 0.5 * Tfin, 0.75 * Tfin, Tfin]

    curves = {}
    for tv in times_to_plot:
        u_tv, t_used = reference_at_time(tv, t_ref, x_ref, U_ref, x_plot, L=L)
        curves[fr"$t={t_used:.3f}$"] = u_tv

    plot_solution(
        x_plot,
        curves,
        title="Finite-difference reference solution of the viscous Burgers equation",
        save_path="numerical_solution.png",
    )
    print("Saved plot to numerical_solution.png")
