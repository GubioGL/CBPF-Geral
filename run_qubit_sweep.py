"""Quantum-versus-classical internal-width sweep using Experiment 1.

Default configuration
---------------------
* collocation counts: N = 10 and 100;
* quantum qubits / matched classical interface widths: 2, 3, ..., 25;
* CNOT network: brickwork;
* CNOT network repetitions: 1;
* classical replacement: one hidden tanh layer with width matched to qubits;
* random collocation points, resampled at every AdamW epoch;
* stopping monitored on the current training batch, without fixed validation;
* time horizons T = 0.1 and 1.0 and seeds 0, ..., 4;
* MSE plots show mean +/- population variance; stopping plots show the mean
  and population variance on separate vertical axes.

This file does not modify ``run_quantumExact.ipynb``. Results are written by
default to ``Experiment1/CBPF_qubit_sweep_quantum_classical_results``. Each run
is saved to JSON first, so an interrupted sweep can resume without repeating
completed cases.

Examples
--------
Inspect the configuration without training::

    python run_qubit_sweep.py --dry-run

Run the default configuration::

    python run_qubit_sweep.py

Run only two new cases as a smoke test::

    python run_qubit_sweep.py --max-new-runs 2

Change the main variables::

    python run_qubit_sweep.py --network ring_cnot --layers 3 \
        --n-collocation 20 50 --qubits 4 8 12 --tfin 0.1 --seeds 0 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Default values. Every value can also be changed from the command line.
# ---------------------------------------------------------------------------
SUPPORTED_NETWORKS = (
    "linear_cnot",
    "reverse_linear_cnot",
    "ring_cnot",
    "brickwork_cnot",
    "symmetric_cnot_ladder",
)
DEFAULT_N_COLLOCATION = (10, 100)
DEFAULT_MIN_QUBITS = 2
DEFAULT_MAX_QUBITS = 25
DEFAULT_NETWORK = "brickwork_cnot"
DEFAULT_CNOT_LAYERS = 1
DEFAULT_COLLOCATION_DISTRIBUTION = "random"
DEFAULT_TFIN = (0.1, 1.0)
DEFAULT_SEEDS = tuple(range(5))
SUPPORTED_MODEL_FAMILIES = ("quantum_exact", "classical_mlp")
DEFAULT_MODEL_FAMILIES = SUPPORTED_MODEL_FAMILIES

DEFAULT_PRE_NEURONS = 10
DEFAULT_PRE_DEPTH = 1
DEFAULT_POST_NEURONS = 10
DEFAULT_POST_DEPTH = 1
DEFAULT_INPUT_ROTATION_AXIS = "rx"
DEFAULT_ANGLE_SCALE = None
DEFAULT_CLASSICAL_INTRA_DEPTH = 1
DEFAULT_CLASSICAL_INTRA_NEURONS = None

DEFAULT_L = 1.0
DEFAULT_NU = 0.01
DEFAULT_REFERENCE_NX = 512
DEFAULT_REFERENCE_DT = 1.0e-4
DEFAULT_HX = DEFAULT_L / 512
DEFAULT_HT = 1.0e-3
DEFAULT_EVALUATION_POINTS = 200

DEFAULT_ADAM_EPOCHS = 20_000
DEFAULT_ADAM_LR = 1.0e-3
DEFAULT_WEIGHT_DECAY = 1.0e-6
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_LBFGS_EPOCHS = 1_000
DEFAULT_LBFGS_LR = 0.5
DEFAULT_LBFGS_PATIENCE = 50
DEFAULT_LBFGS_TOL = 1.0e-9

PROTOCOL_VERSION = "experiment1_training_batch_v1"

# Scientific dependencies are loaded only after processing ``--dry-run``.
# This keeps configuration inspection available outside the notebook's Python
# environment.
np = None
tc = None
predict = None
reference_at_time = None
reference_pde_residual = None
relative_l2_error = None
sample_collocation_points = None
set_seed = None
solve_burgers_fd = None
train = None
QuantumExactPINN = None
ClassicalPINN2 = None

RUN_FIELDS = (
    "protocol_version",
    "model_family",
    "network",
    "n_cnot_layers",
    "n_qubits",
    "internal_width",
    "Tfin",
    "n_collocation",
    "seed",
    "mse",
    "mae",
    "relative_l2",
    "target_loss",
    "target_reached",
    "stopping_monitor",
    "stopped_at_step",
    "adam_steps",
    "lbfgs_steps",
    "elapsed_seconds",
    "trainable_parameters",
)

SUMMARY_FIELDS = (
    "protocol_version",
    "model_family",
    "network",
    "n_cnot_layers",
    "n_qubits",
    "internal_width",
    "Tfin",
    "n_collocation",
    "mse_mean",
    "mse_min",
    "mse_max",
    "mse_std",
    "mse_variance",
    "steps_mean",
    "steps_std",
    "steps_variance",
    "steps_min",
    "steps_max",
    "target_reached_fraction",
    "trainable_parameters",
    "n_samples",
)


def _load_analysis_dependencies() -> None:
    global np
    if np is not None:
        return
    try:
        import numpy as numpy_module
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "NumPy is unavailable. Run this file in the same Python "
            "environment used by run_quantumExact.ipynb."
        ) from error
    np = numpy_module


def _load_training_dependencies() -> None:
    global tc
    global predict, reference_at_time, reference_pde_residual
    global relative_l2_error, sample_collocation_points, set_seed
    global solve_burgers_fd, train, QuantumExactPINN, ClassicalPINN2

    _load_analysis_dependencies()
    if tc is not None:
        return
    try:
        import torch as torch_module
        from main.functions import (
            predict as predict_function,
            reference_at_time as reference_at_time_function,
            reference_pde_residual as reference_pde_residual_function,
            relative_l2_error as relative_l2_error_function,
            sample_collocation_points as sample_collocation_points_function,
            set_seed as set_seed_function,
            solve_burgers_fd as solve_burgers_fd_function,
            train as train_function,
        )
        from main.models import (
            ClassicalPINN2 as ClassicalPINN2Class,
            QuantumExactPINN as QuantumExactPINNClass,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch or the project modules are unavailable. Run this file "
            "from 'CBPF geral' in the same Python environment used by "
            "run_quantumExact.ipynb."
        ) from error

    tc = torch_module
    predict = predict_function
    reference_at_time = reference_at_time_function
    reference_pde_residual = reference_pde_residual_function
    relative_l2_error = relative_l2_error_function
    sample_collocation_points = sample_collocation_points_function
    set_seed = set_seed_function
    solve_burgers_fd = solve_burgers_fd_function
    train = train_function
    QuantumExactPINN = QuantumExactPINNClass
    ClassicalPINN2 = ClassicalPINN2Class


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("the value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("the value must be a non-negative integer")
    return parsed


def _optional_positive_int(value: str) -> int:
    return _positive_int(value)


def resolve_project_directory() -> Path:
    """Locate ``CBPF geral`` both as a script and inside a Jupyter cell."""
    source_file = globals().get("__file__")
    if source_file:
        return Path(source_file).resolve().parent

    # Jupyter does not define __file__. Usually its working directory is the
    # project itself; walking through the parents also covers kernels started
    # from a subdirectory of CBPF geral.
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "main" / "functions.py").is_file()
            and (candidate / "main" / "models.py").is_file()
        ):
            return candidate
    return current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the exact HPINN qubit count and a matched classical "
            "internal width using the random Experiment 1 protocol."
        )
    )
    parser.add_argument(
        "--n-collocation",
        nargs="+",
        type=_positive_int,
        default=list(DEFAULT_N_COLLOCATION),
        help="fixed collocation counts (default: 10 100)",
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=_positive_int,
        default=None,
        help="explicit widths; overrides --min-qubits and --max-qubits",
    )
    parser.add_argument("--min-qubits", type=_positive_int, default=DEFAULT_MIN_QUBITS)
    parser.add_argument("--max-qubits", type=_positive_int, default=DEFAULT_MAX_QUBITS)
    parser.add_argument(
        "--network",
        choices=SUPPORTED_NETWORKS,
        default=DEFAULT_NETWORK,
        help="fixed CNOT topology for the quantum model",
    )
    parser.add_argument(
        "--layers",
        type=_nonnegative_int,
        default=DEFAULT_CNOT_LAYERS,
        help="number of CNOT topology repetitions (default: 1)",
    )
    parser.add_argument(
        "--distribution",
        choices=("random", "uniform"),
        default=DEFAULT_COLLOCATION_DISTRIBUTION,
    )
    parser.add_argument("--tfin", nargs="+", type=float, default=list(DEFAULT_TFIN))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODEL_FAMILIES,
        default=list(DEFAULT_MODEL_FAMILIES),
        help="model families to run (default: quantum_exact classical_mlp)",
    )

    parser.add_argument("--pre-neurons", type=_positive_int, default=DEFAULT_PRE_NEURONS)
    parser.add_argument("--pre-depth", type=_positive_int, default=DEFAULT_PRE_DEPTH)
    parser.add_argument("--post-neurons", type=_positive_int, default=DEFAULT_POST_NEURONS)
    parser.add_argument("--post-depth", type=_positive_int, default=DEFAULT_POST_DEPTH)
    parser.add_argument(
        "--rotation-axis",
        choices=("rx", "ry"),
        default=DEFAULT_INPUT_ROTATION_AXIS,
    )
    parser.add_argument(
        "--angle-scale",
        type=float,
        default=DEFAULT_ANGLE_SCALE,
        help="None keeps the unbounded-angle convention from Experiment 1",
    )
    parser.add_argument(
        "--classical-intra-depth",
        type=_positive_int,
        default=DEFAULT_CLASSICAL_INTRA_DEPTH,
        help="number of hidden tanh layers in the classical replacement",
    )
    parser.add_argument(
        "--classical-intra-neurons",
        type=_positive_int,
        default=DEFAULT_CLASSICAL_INTRA_NEURONS,
        help=(
            "hidden neurons in the classical replacement; omission matches "
            "the current qubit count/internal interface width"
        ),
    )

    parser.add_argument("--L", type=float, default=DEFAULT_L)
    parser.add_argument("--nu", type=float, default=DEFAULT_NU)
    parser.add_argument("--reference-nx", type=_positive_int, default=DEFAULT_REFERENCE_NX)
    parser.add_argument("--reference-dt", type=float, default=DEFAULT_REFERENCE_DT)
    parser.add_argument("--hx", type=float, default=DEFAULT_HX)
    parser.add_argument("--ht", type=float, default=DEFAULT_HT)
    parser.add_argument(
        "--evaluation-points",
        type=_positive_int,
        default=DEFAULT_EVALUATION_POINTS,
    )

    parser.add_argument("--adam-epochs", type=_positive_int, default=DEFAULT_ADAM_EPOCHS)
    parser.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument(
        "--lbfgs-epochs",
        type=_nonnegative_int,
        default=DEFAULT_LBFGS_EPOCHS,
    )
    parser.add_argument("--lbfgs-lr", type=float, default=DEFAULT_LBFGS_LR)
    parser.add_argument(
        "--lbfgs-patience",
        type=_positive_int,
        default=DEFAULT_LBFGS_PATIENCE,
    )
    parser.add_argument("--lbfgs-tol", type=float, default=DEFAULT_LBFGS_TOL)

    default_output = resolve_project_directory().parent / (
        "CBPF_qubit_sweep_quantum_classical_results"
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--max-new-runs",
        type=_optional_positive_int,
        default=None,
        help="limit only new runs; useful for smoke tests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun cases that already have a JSON result",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="save one .pt checkpoint for each case",
    )
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="only rebuild CSV files and plots from existing JSON results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the configuration without writing or training",
    )
    return parser


def resolve_qubit_values(args: argparse.Namespace) -> tuple[int, ...]:
    if args.qubits is not None:
        values = tuple(sorted(set(args.qubits)))
    else:
        if args.min_qubits > args.max_qubits:
            raise ValueError("--min-qubits cannot be greater than --max-qubits")
        values = tuple(range(args.min_qubits, args.max_qubits + 1))
    if not values:
        raise ValueError("at least one internal width must be provided")
    return values


def validate_args(args: argparse.Namespace, qubits: tuple[int, ...]) -> None:
    del qubits
    if args.L <= 0.0 or args.nu < 0.0:
        raise ValueError("L must be positive and nu must be non-negative")
    if args.hx <= 0.0 or args.ht <= 0.0 or args.reference_dt <= 0.0:
        raise ValueError("hx, ht, and reference-dt must be positive")
    if any(value <= 2.0 * args.ht for value in args.tfin):
        raise ValueError("each Tfin value must satisfy Tfin > 2*ht")
    if args.adam_lr <= 0.0 or args.lbfgs_lr <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.weight_decay < 0.0 or args.lbfgs_tol < 0.0:
        raise ValueError("weight-decay and lbfgs-tol must be non-negative")
    if args.angle_scale is not None and (
        not math.isfinite(args.angle_scale) or args.angle_scale <= 0.0
    ):
        raise ValueError("angle-scale must be positive and finite")


def _float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_key(row: dict) -> tuple:
    return (
        str(row.get("model_family", "quantum_exact")),
        str(row["network"]),
        int(row["n_cnot_layers"]),
        int(row.get("internal_width", row["n_qubits"])),
        float(row["Tfin"]),
        int(row["n_collocation"]),
        int(row["seed"]),
    )


def raw_result_path(
    raw_dir: Path,
    model_family: str,
    network: str,
    layers: int,
    internal_width: int,
    tfin: float,
    n_collocation: int,
    seed: int,
) -> Path:
    return (
        raw_dir
        / model_family
        / network
        / f"layers_{layers}"
        / f"width_{internal_width}"
        / f"T_{_float_tag(tfin)}"
        / f"N_{n_collocation}"
        / f"seed_{seed}.json"
    )


def model_label(model_family: str) -> str:
    labels = {
        "quantum_exact": "Exact CNOT-cosine HPINN",
        "classical_mlp": "Classical tanh PINN",
    }
    return labels.get(model_family, model_family)


def build_model(
    args: argparse.Namespace,
    model_family: str,
    internal_width: int,
    tfin: float,
):
    """Build quantum and classical models with the same outer architecture."""
    common = {
        "pre_neurons": args.pre_neurons,
        "pre_depth": args.pre_depth,
        "post_neurons": args.post_neurons,
        "post_depth": args.post_depth,
        "L": args.L,
        "Tfin": tfin,
    }
    if model_family == "quantum_exact":
        return QuantumExactPINN(
            n_qubits=internal_width,
            fixed_layer=args.network,
            fixed_layer_repeats=args.layers,
            input_rotation_axis=args.rotation_axis,
            angle_scale=args.angle_scale,
            **common,
        )
    if model_family == "classical_mlp":
        hidden_width = (
            internal_width
            if args.classical_intra_neurons is None
            else args.classical_intra_neurons
        )
        return ClassicalPINN2(
            intra_neuron=internal_width,
            intra_neurons=hidden_width,
            intra_depth=args.classical_intra_depth,
            **common,
        )
    raise ValueError(f"Unsupported model family: {model_family}")


def _atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def _atomic_csv_dump(rows: Iterable[dict], path: Path, fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def load_raw_rows(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not raw_dir.exists():
        return rows
    for path in sorted(raw_dir.rglob("seed_*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
            row.setdefault("model_family", "quantum_exact")
            row.setdefault("internal_width", int(row["n_qubits"]))
            run_key(row)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            print(f"Warning: ignored result at {path}: {error}")
            continue
        rows.append(row)
    return rows


def summarize(rows: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            str(row.get("model_family", "quantum_exact")),
            str(row["network"]),
            int(row["n_cnot_layers"]),
            int(row.get("internal_width", row["n_qubits"])),
            float(row["Tfin"]),
            int(row["n_collocation"]),
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict] = []
    for (
        model_family,
        network,
        layers,
        internal_width,
        tfin,
        n_collocation,
    ), group in sorted(
        grouped.items()
    ):
        mse = np.asarray([float(row["mse"]) for row in group], dtype=np.float64)
        steps = np.asarray(
            [float(row["stopped_at_step"]) for row in group], dtype=np.float64
        )
        targets = np.asarray(
            [bool(row["target_reached"]) for row in group], dtype=np.float64
        )
        parameters = np.asarray(
            [int(row["trainable_parameters"]) for row in group], dtype=np.int64
        )
        summary.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "model_family": model_family,
                "network": network,
                "n_cnot_layers": layers,
                "n_qubits": internal_width,
                "internal_width": internal_width,
                "Tfin": tfin,
                "n_collocation": n_collocation,
                "mse_mean": float(np.mean(mse)),
                "mse_min": float(np.min(mse)),
                "mse_max": float(np.max(mse)),
                "mse_std": float(np.std(mse)),
                "mse_variance": float(np.var(mse)),
                "steps_mean": float(np.mean(steps)),
                "steps_std": float(np.std(steps)),
                "steps_variance": float(np.var(steps)),
                "steps_min": int(np.min(steps)),
                "steps_max": int(np.max(steps)),
                "target_reached_fraction": float(np.mean(targets)),
                "trainable_parameters": int(parameters[0]),
                "n_samples": int(len(group)),
            }
        )
    return summary


def save_tables(rows: Iterable[dict], output_dir: Path) -> list[dict]:
    ordered_rows = sorted(rows, key=run_key)
    summary = summarize(ordered_rows)
    _atomic_csv_dump(ordered_rows, output_dir / "qubit_sweep_runs.csv", RUN_FIELDS)
    _atomic_csv_dump(
        summary,
        output_dir / "qubit_sweep_summary.csv",
        SUMMARY_FIELDS,
    )
    return summary


def _save_figure_pair(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def _plot_mse_panels(
    plt,
    rows: list[dict],
    families: list[str],
    title: str,
    stem: Path,
) -> None:
    times = sorted({float(row["Tfin"]) for row in rows})
    if not times:
        return
    colors = {"quantum_exact": "#2878C7", "classical_mlp": "#F28E2B"}
    markers = {"quantum_exact": "o", "classical_mlp": "s"}
    all_widths = sorted({int(row["internal_width"]) for row in rows})

    fig, axes = plt.subplots(
        1,
        len(times),
        figsize=(9.0 * len(times), 6.2),
        squeeze=False,
    )
    for axis, tfin in zip(axes[0], times):
        for family in families:
            selected = sorted(
                [
                    row
                    for row in rows
                    if float(row["Tfin"]) == tfin
                    and str(row["model_family"]) == family
                ],
                key=lambda row: int(row["internal_width"]),
            )
            if not selected:
                continue
            widths = np.asarray(
                [int(row["internal_width"]) for row in selected], dtype=np.int64
            )
            means = np.asarray(
                [float(row["mse_mean"]) for row in selected], dtype=np.float64
            )
            variances = np.asarray(
                [float(row["mse_variance"]) for row in selected], dtype=np.float64
            )
            label = (
                "MSE (mean +/- variance)"
                if len(families) == 1
                else model_label(family)
            )
            axis.errorbar(
                widths,
                means,
                yerr=variances,
                fmt=f"-{markers[family]}",
                color=colors[family],
                capsize=4,
                linewidth=2,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_xlabel("INTERNAL WIDTH / N_QUBITS")
        axis.set_ylabel("MSE")
        axis.set_title(f"T = {tfin:g}")
        axis.set_xticks(all_widths)
        axis.set_xticklabels([str(value) for value in all_widths])
        axis.grid(True, which="both", alpha=0.35, linestyle="--")
        axis.legend(loc="best")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    _save_figure_pair(fig, stem)
    plt.close(fig)


def _plot_step_panels(
    plt,
    rows: list[dict],
    families: list[str],
    title: str,
    stem: Path,
) -> None:
    times = sorted({float(row["Tfin"]) for row in rows})
    if not times:
        return
    colors = {"quantum_exact": "#2878C7", "classical_mlp": "#F28E2B"}
    markers = {"quantum_exact": "o", "classical_mlp": "s"}
    all_widths = sorted({int(row["internal_width"]) for row in rows})

    fig, axes = plt.subplots(
        1,
        len(times),
        figsize=(9.0 * len(times), 6.2),
        squeeze=False,
    )
    for axis, tfin in zip(axes[0], times):
        variance_axis = axis.twinx()
        handles = []
        labels = []
        for family in families:
            selected = sorted(
                [
                    row
                    for row in rows
                    if float(row["Tfin"]) == tfin
                    and str(row["model_family"]) == family
                ],
                key=lambda row: int(row["internal_width"]),
            )
            if not selected:
                continue
            widths = np.asarray(
                [int(row["internal_width"]) for row in selected], dtype=np.int64
            )
            means = np.asarray(
                [float(row["steps_mean"]) for row in selected], dtype=np.float64
            )
            variances = np.asarray(
                [float(row["steps_variance"]) for row in selected], dtype=np.float64
            )
            mean_label = (
                "Mean stopping steps"
                if len(families) == 1
                else f"{model_label(family)} mean"
            )
            variance_label = (
                "Stopping-step variance"
                if len(families) == 1
                else f"{model_label(family)} variance"
            )
            mean_line = axis.plot(
                widths,
                means,
                marker=markers[family],
                color=colors[family],
                linewidth=2,
                label=mean_label,
            )[0]
            variance_line = variance_axis.plot(
                widths,
                variances,
                marker=markers[family],
                linestyle="--",
                color=colors[family],
                alpha=0.75,
                linewidth=2,
                label=variance_label,
            )[0]
            handles.extend([mean_line, variance_line])
            labels.extend([mean_label, variance_label])

        axis.set_xlabel("INTERNAL WIDTH / N_QUBITS")
        axis.set_ylabel("Mean stopping steps")
        variance_axis.set_ylabel("Stopping-step variance")
        axis.set_title(f"T = {tfin:g}")
        axis.set_xticks(all_widths)
        axis.set_xticklabels([str(value) for value in all_widths])
        axis.grid(True, alpha=0.35, linestyle="--")
        axis.legend(handles, labels, loc="best", fontsize=8)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    _save_figure_pair(fig, stem)
    plt.close(fig)


def save_plots(summary: list[dict], output_dir: Path) -> None:
    if not summary:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "Warning: matplotlib is unavailable. CSV files were saved, but "
            "plots were not generated."
        )
        return

    plot_dir = output_dir / "plots"
    individual_dir = plot_dir / "individual"
    comparison_dir = plot_dir / "comparison"
    individual_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    experiment_groups = sorted(
        {
            (
                str(row["network"]),
                int(row["n_cnot_layers"]),
                int(row["n_collocation"]),
            )
            for row in summary
        }
    )
    for network, layers, n_collocation in experiment_groups:
        selected_experiment = [
            row
            for row in summary
            if str(row["network"]) == network
            and int(row["n_cnot_layers"]) == layers
            and int(row["n_collocation"]) == n_collocation
        ]
        families = sorted(
            {str(row["model_family"]) for row in selected_experiment}
        )
        for family in families:
            family_rows = [
                row
                for row in selected_experiment
                if str(row["model_family"]) == family
            ]
            family_title = (
                f"{model_label(family)}: {network}, {layers} layer(s), "
                f"N_COLLOCATION={n_collocation}"
            )
            family_stem = (
                f"{family}_{network}_layers_{layers}_"
                f"N_COLLOCATION_{n_collocation}"
            )
            _plot_mse_panels(
                plt,
                family_rows,
                [family],
                family_title,
                individual_dir / f"MSE_vs_INTERNAL_WIDTH_{family_stem}",
            )
            _plot_step_panels(
                plt,
                family_rows,
                [family],
                family_title,
                individual_dir / f"Steps_vs_INTERNAL_WIDTH_{family_stem}",
            )

        if {"quantum_exact", "classical_mlp"}.issubset(families):
            comparison_title = (
                "Quantum versus classical: "
                f"{network}, {layers} layer(s), N_COLLOCATION={n_collocation}"
            )
            comparison_stem = (
                f"{network}_layers_{layers}_N_COLLOCATION_{n_collocation}"
            )
            ordered_families = ["quantum_exact", "classical_mlp"]
            _plot_mse_panels(
                plt,
                selected_experiment,
                ordered_families,
                comparison_title,
                comparison_dir / f"Comparison_MSE_{comparison_stem}",
            )
            _plot_step_panels(
                plt,
                selected_experiment,
                ordered_families,
                comparison_title,
                comparison_dir / f"Comparison_Steps_{comparison_stem}",
            )


def print_plan(args: argparse.Namespace, qubits: tuple[int, ...]) -> None:
    total = (
        len(args.models)
        * len(qubits)
        * len(args.n_collocation)
        * len(args.tfin)
        * len(args.seeds)
    )
    print("Sweep configuration")
    print(f"  protocol: {PROTOCOL_VERSION}")
    print(f"  model families: {tuple(args.models)}")
    print(
        f"  internal widths: {qubits[0]}..{qubits[-1]} "
        f"({len(qubits)} values)"
    )
    print(f"  collocation counts: {tuple(args.n_collocation)}")
    print(f"  Tfin: {tuple(args.tfin)}")
    print(f"  seeds: {tuple(args.seeds)}")
    print(f"  CNOT network: {args.network}, {args.layers} layer(s)")
    if "classical_mlp" in args.models:
        hidden_width = (
            "matched to each internal width"
            if args.classical_intra_neurons is None
            else str(args.classical_intra_neurons)
        )
        print(
            "  classical replacement: "
            f"{args.classical_intra_depth} hidden tanh layer(s), "
            f"hidden width {hidden_width}"
        )
    print(f"  collocation distribution: {args.distribution}")
    print("  stopping monitor: current training batch")
    print(f"  planned total: {total} runs")
    print(f"  results: {args.output_dir.resolve()}")


def run_sweep(args: argparse.Namespace) -> None:
    qubits = resolve_qubit_values(args)
    validate_args(args, qubits)
    selected_models = set(args.models)
    args.models = [
        model_family
        for model_family in SUPPORTED_MODEL_FAMILIES
        if model_family in selected_models
    ]
    args.n_collocation = sorted(set(args.n_collocation))
    args.tfin = sorted(set(float(value) for value in args.tfin))
    args.seeds = sorted(set(args.seeds))
    args.output_dir = args.output_dir.expanduser().resolve()
    print_plan(args, qubits)

    if args.dry_run:
        return

    if args.rebuild_only:
        _load_analysis_dependencies()
    else:
        _load_training_dependencies()

    output_dir = args.output_dir
    raw_dir = output_dir / "raw_runs"
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config_payload.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "qubit_values": list(qubits),
            "internal_width_values": list(qubits),
            "stopping_monitor": "training_batch",
            "fixed_validation": False,
        }
    )
    _atomic_json_dump(config_payload, output_dir / "last_run_config.json")

    rows_by_key = {run_key(row): row for row in load_raw_rows(raw_dir)}
    if args.rebuild_only:
        summary = save_tables(rows_by_key.values(), output_dir)
        save_plots(summary, output_dir)
        print(f"Rebuilt {len(summary)} aggregate points.")
        return

    reference_cache: dict[float, tuple] = {}
    for tfin in args.tfin:
        x_ref, t_ref, u_ref = solve_burgers_fd(
            L=args.L,
            Tfin=tfin,
            nu=args.nu,
            Nx=args.reference_nx,
            dt=args.reference_dt,
        )
        x_evaluation = np.linspace(
            0.0, args.L, args.evaluation_points, endpoint=False
        ).reshape(-1, 1)
        u_ref_evaluation, resolved_time = reference_at_time(
            tfin,
            t_ref,
            x_ref,
            u_ref,
            x_evaluation,
            L=args.L,
        )
        reference_cache[tfin] = (
            x_ref,
            t_ref,
            u_ref,
            x_evaluation,
            u_ref_evaluation,
            resolved_time,
        )

    new_runs = 0
    stop_requested = False
    try:
        for tfin in args.tfin:
            (
                x_ref,
                t_ref,
                u_ref,
                x_evaluation,
                u_ref_evaluation,
                resolved_time,
            ) = reference_cache[tfin]
            for n_collocation in args.n_collocation:
                for internal_width in qubits:
                    for seed in args.seeds:
                        for model_family in args.models:
                            key = (
                                model_family,
                                args.network,
                                args.layers,
                                internal_width,
                                tfin,
                                n_collocation,
                                seed,
                            )
                            result_path = raw_result_path(
                                raw_dir,
                                model_family,
                                args.network,
                                args.layers,
                                internal_width,
                                tfin,
                                n_collocation,
                                seed,
                            )
                            if key in rows_by_key and not args.overwrite:
                                print(
                                    "Skipping completed case: "
                                    f"model={model_family}, T={tfin:g}, "
                                    f"N={n_collocation}, width={internal_width}, "
                                    f"seed={seed}"
                                )
                                continue
                            if (
                                args.max_new_runs is not None
                                and new_runs >= args.max_new_runs
                            ):
                                stop_requested = True
                                break

                            print(
                                "Training: "
                                f"model={model_family}, T={tfin:g}, "
                                f"N={n_collocation}, width={internal_width}, "
                                f"seed={seed}"
                            )
                            set_seed(seed)
                            model = build_model(
                                args,
                                model_family,
                                internal_width,
                                tfin,
                            )

                            x_f, t_f = sample_collocation_points(
                                n_collocation,
                                args.L,
                                tfin,
                                args.ht,
                                seed=seed,
                                distribution=args.distribution,
                            )

                            # Original Experiment 1 training protocol:
                            # - compute the loss target on the initial point set;
                            # - give AdamW a new random point set every epoch;
                            # - monitor the current batch, without validation_points;
                            # - keep the final AdamW batch fixed throughout L-BFGS.
                            if args.distribution == "random":
                                resample_fn = (
                                    lambda n=n_collocation, tf=tfin: sample_collocation_points(
                                        n,
                                        args.L,
                                        tf,
                                        args.ht,
                                        seed=None,
                                        distribution="random",
                                    )
                                )
                            else:
                                resample_fn = None

                            target_loss = reference_pde_residual(
                                x_f,
                                t_f,
                                x_ref,
                                t_ref,
                                u_ref,
                                L=args.L,
                                nu=args.nu,
                                hx=args.hx,
                                ht=args.ht,
                            )

                            start_time = time.time()
                            model, history = train(
                                model,
                                x_f,
                                t_f,
                                loss_kwargs={
                                    "L": args.L,
                                    "Tfin": tfin,
                                    "nu": args.nu,
                                    "hx": args.hx,
                                    "ht": args.ht,
                                },
                                epochs=args.adam_epochs,
                                lr=args.adam_lr,
                                grad_clip=args.grad_clip,
                                weight_decay=args.weight_decay,
                                resample_fn=resample_fn,
                                lbfgs_epochs=args.lbfgs_epochs,
                                lbfgs_lr=args.lbfgs_lr,
                                lbfgs_resample_fn=None,
                                lbfgs_patience=args.lbfgs_patience,
                                lbfgs_tol=args.lbfgs_tol,
                                target_loss=target_loss,
                            )
                            elapsed_seconds = time.time() - start_time

                            prediction = predict(model, x_evaluation, tfin)
                            mse = float(
                                np.mean((prediction - u_ref_evaluation) ** 2)
                            )
                            mae = float(
                                np.mean(np.abs(prediction - u_ref_evaluation))
                            )
                            stopped_at_step = int(len(history["loss"]))
                            adam_steps = min(stopped_at_step, args.adam_epochs)
                            lbfgs_steps = max(
                                0, stopped_at_step - args.adam_epochs
                            )

                            payload = {
                                "protocol_version": PROTOCOL_VERSION,
                                "model_family": model_family,
                                "network": args.network,
                                "n_cnot_layers": int(args.layers),
                                "n_qubits": int(internal_width),
                                "internal_width": int(internal_width),
                                "Tfin": float(tfin),
                                "reference_time": float(resolved_time),
                                "n_collocation": int(n_collocation),
                                "seed": int(seed),
                                "mse": mse,
                                "mae": mae,
                                "relative_l2": float(
                                    relative_l2_error(
                                        prediction, u_ref_evaluation
                                    )
                                ),
                                "target_loss": float(target_loss),
                                "target_reached": bool(
                                    history["target_reached"]
                                ),
                                "stopping_monitor": str(
                                    history["monitor_source"]
                                ),
                                "stopped_at_step": stopped_at_step,
                                "adam_steps": int(adam_steps),
                                "lbfgs_steps": int(lbfgs_steps),
                                "elapsed_seconds": float(elapsed_seconds),
                                "trainable_parameters": int(
                                    sum(
                                        parameter.numel()
                                        for parameter in model.parameters()
                                        if parameter.requires_grad
                                    )
                                ),
                                "model_config": dict(model.config),
                                "training_config": {
                                    "adam_epochs": args.adam_epochs,
                                    "adam_lr": args.adam_lr,
                                    "weight_decay": args.weight_decay,
                                    "grad_clip": args.grad_clip,
                                    "lbfgs_epochs": args.lbfgs_epochs,
                                    "lbfgs_lr": args.lbfgs_lr,
                                    "lbfgs_patience": args.lbfgs_patience,
                                    "lbfgs_tol": args.lbfgs_tol,
                                    "collocation_distribution": (
                                        args.distribution
                                    ),
                                    "adam_resamples_each_epoch": bool(
                                        args.distribution == "random"
                                    ),
                                    "lbfgs_resamples": False,
                                    "fixed_validation": False,
                                },
                            }
                            _atomic_json_dump(payload, result_path)
                            rows_by_key[key] = payload

                            if args.save_models:
                                checkpoint = (
                                    model_dir
                                    / model_family
                                    / args.network
                                    / f"layers_{args.layers}"
                                    / f"width_{internal_width}"
                                    / f"T_{_float_tag(tfin)}"
                                    / f"N_{n_collocation}"
                                    / f"seed_{seed}.pt"
                                )
                                checkpoint.parent.mkdir(
                                    parents=True, exist_ok=True
                                )
                                model.save(
                                    checkpoint,
                                    training_params=payload["training_config"],
                                )

                            new_runs += 1
                            summary = save_tables(
                                rows_by_key.values(), output_dir
                            )
                            print(
                                f"Saved: MSE={mse:.6e}, "
                                f"steps={stopped_at_step}, "
                                f"target={history['target_reached']}, "
                                f"time={elapsed_seconds:.1f}s"
                            )
                        if stop_requested:
                            break
                    if stop_requested:
                        break
                if stop_requested:
                    break
            if stop_requested:
                break
    finally:
        summary = save_tables(rows_by_key.values(), output_dir)
        save_plots(summary, output_dir)

    print(f"New runs completed: {new_runs}")
    print(f"Results available at: {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    # A Jupyter kernel adds its own command-line arguments (for example
    # ``-f kernel-....json``). When this code is pasted into a cell, ignore
    # those arguments and use the defaults. To customize from a cell, call
    # main(["--qubits", "2", "3", ...]).
    if argv is None and globals().get("__file__") is None:
        argv = []
    args = parser.parse_args(argv)
    run_sweep(args)


if __name__ == "__main__":
    main()
