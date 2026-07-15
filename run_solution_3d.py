import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch as tc

from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.config import L, TFIN, NU
from src.functions import solve_burgers_fd
from src.models import load_model
from matplotlib.patches import Patch

CAMINHO_DOS_DADOS = f"data/classical/T_0.1/amostragem_1.pt"


def build_surface(model, x_grid, t_grid):
    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid)
    xt = np.column_stack([x_mesh.ravel(), t_mesh.ravel()]).astype(np.float32)
    model.eval()
    

    with tc.no_grad():
        xt_tensor = tc.tensor(xt, dtype=tc.get_default_dtype())
        u = model(xt_tensor).cpu().numpy().reshape(t_mesh.shape)
    return x_mesh, t_mesh, u


def reference_surface(x_grid, t_grid, x_ref, t_ref, u_ref_hist, domain_length):
    u_surface = np.zeros((t_grid.size, x_grid.size), dtype=np.float64)
    xp = np.concatenate([x_ref, [domain_length]])

    for i, t_val in enumerate(t_grid):
        idx = int(np.argmin(np.abs(t_ref - t_val)))
        fp = np.concatenate([u_ref_hist[idx], [u_ref_hist[idx][0]]])
        u_surface[i] = np.interp(x_grid, xp, fp)

    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid)
    return x_mesh, t_mesh, u_surface


def plot_3d_panels(x_mesh, t_mesh, fdm_u, model_u, error_u, output_path, model_label):
    fig = plt.figure(figsize=(18, 5.5))
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]

    plot_specs = [
        (fdm_u, "FDM reference", "a)", cm.coolwarm, Normalize(vmin=np.min(fdm_u), vmax=np.max(fdm_u))),
        (model_u, model_label, "b)", cm.coolwarm, Normalize(vmin=np.min(model_u), vmax=np.max(model_u))),
        (error_u, r"|erro|", "c)", cm.coolwarm, Normalize(vmin=0.0, vmax=np.max(error_u) if np.max(error_u) > 0.0 else 1.0)),
    ]

    legend_colors = []
    for ax, (surface_u, legend_label, panel_label, cmap, norm) in zip(axes, plot_specs):
        facecolors = cmap(norm(surface_u))
        ax.plot_surface(
            x_mesh,
            t_mesh,
            surface_u,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("t", fontsize=12)
        ax.set_zlabel("u(x,t)", fontsize=12)
        ax.view_init(elev=28, azim=-135)
        ax.text2D(0.03, 0.94, panel_label, transform=ax.transAxes, fontsize=14, fontweight="bold")
        ax.grid(False)

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(surface_u)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08)
        cbar.ax.tick_params(labelsize=9)
        legend_colors.append(({"FDM reference": "#1f77b4", "Classical PINN": "#ff7f0e", r"|erro|": "#d62728"}[legend_label], legend_label))


    legend_handles = [Patch(facecolor=color, edgecolor="none", label=label) for color, label in legend_colors]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Gera um plot 3D da evolução u(x,t) e do erro.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(CAMINHO_DOS_DADOS),
        help="Checkpoint do modelo treinado.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/classical/classical_solution_3d_N100.png"),
        help="Arquivo PNG de saída.",
    )
    parser.add_argument("--nx", type=int, default=120, help="Número de pontos em x para a malha 3D.")
    parser.add_argument("--nt", type=int, default=80, help="Número de pontos em t para a malha 3D.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {args.checkpoint}")

    model = load_model(str(args.checkpoint), map_location="cpu")

    x_ref, t_ref, u_ref_hist = solve_burgers_fd(L=L, Tfin=TFIN, nu=NU)
    x_grid = np.linspace(0.0, L, args.nx)
    t_grid = np.linspace(0.0, TFIN, args.nt)

    x_mesh, t_mesh, fdm_u = reference_surface(x_grid, t_grid, x_ref, t_ref, u_ref_hist, L)
    _, _, model_u = build_surface(model, x_grid, t_grid)
    error_u = np.abs(model_u - fdm_u)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_3d_panels(
        x_mesh=x_mesh,
        t_mesh=t_mesh,
        fdm_u=fdm_u,
        model_u=model_u,
        error_u=error_u,
        output_path=args.output,
        model_label="Classical PINN",
    )

    print(f"Figura salva em: {args.output}")
    print(f"Erro L2 relativo no grid 3D: {np.linalg.norm(model_u - fdm_u) / np.linalg.norm(fdm_u):.6e}")


if __name__ == "__main__":
    main()