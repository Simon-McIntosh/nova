"""Compare the frozen-plasma turning-point tangent with the self-consistent one.

The shape inverse places its current step against
``turning_point_response_matrix``, which differences the boundary read of the
seed flux plus a circuit's *vacuum* response with the plasma current frozen. On
a near-marginal vertical channel the plasma's own re-equilibration dominates
that vacuum motion, so the frozen tangent understates the achieved motion and
the regularised least squares has nothing predictive to invert.

This driver measures the response the plasma actually delivers. At the plus and
the minus of a stated current step on each free circuit it re-solves the same
forward problem to convergence from the converged seed state and differences
the achieved turning points. The result is tabled per circuit and per extremum
beside the frozen tangent, with the whole-plasma (common-mode) displacement
separated from the differential (shape-changing) motion.

The measurement runs on the accelerated lane and persists each circuit as it
lands. ``--render`` draws the measurement on the CPU lane from the persisted
receipt alone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from apps.playable.production import ForwardMachine, ProductionSolver
from nova.equilibrium.shape_inverse import (
    TURNING_POINT_TANGENT_STEP_A,
    achieved_target,
    turning_point_response_matrix,
)
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import ink, poloidal

ROOT = Path(__file__).resolve().parent.parent
TARGET = (22086, 43)
DIRECTORY = ROOT / "docs/figures/playable-forward-solve/shape-inverse"
RECEIPT_NAME = "response-comparison.json"
STATES_NAME = "response-comparison-states.npz"
FIGURE_NAME = "response-comparison.png"

#: Symmetric circuit-current steps whose central differences are compared [A].
STEPS_A = (500.0, 2000.0)
EXTREMA = ("outer", "upper", "inner", "lower")


def _source_revision() -> str:
    """Return the revision this measurement runs from."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _points(target: Any) -> np.ndarray:
    """Return one target's principal turning points as a host array."""
    return np.asarray(target.flux_points, dtype=float)[:4]


def _circuit_names(policy: dict[str, Any]) -> dict[int, str]:
    """Return active-family names keyed by zero-based response column."""
    return {
        int(item["stored_circuit"]) - 1: str(item["family"])
        for item in policy["active_mapping"]
    }


def _circuit_label(index: int, names: dict[int, str]) -> str:
    """Return a circuit label, naming its active family when one is known."""
    family = names.get(index)
    return f"circuit_{index:02d}" if family is None else f"circuit_{index:02d}_{family}"


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Persist one payload so a partial run leaves a readable receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _topology_summary(profile: Any, flux: Any) -> dict[str, Any]:
    """Return the achieved topology class and null locations of one state."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    axis = np.asarray(topology.axis, dtype=float)
    x_point = np.asarray(topology.x_point, dtype=float)
    diverted = bool(np.asarray(topology.diverted))
    return {
        "class": "diverted" if diverted else "limited",
        "diverted": diverted,
        "axis_m": [float(axis[0]), float(axis[1])],
        "x_point_m": [float(x_point[0]), float(x_point[1])],
        "boundary_flux_wb": float(np.asarray(topology.boundary_flux)),
    }


def _vector(value: np.ndarray) -> dict[str, float]:
    """Return one planar displacement with its magnitude."""
    return {
        "dR": float(value[0]),
        "dZ": float(value[1]),
        "norm": float(np.linalg.norm(value)),
    }


def _solve_at(
    solver: ProductionSolver,
    profile: Any,
    warm: Any,
    base_current: np.ndarray,
    circuit: int,
    step: float,
    sign: float,
) -> dict[str, Any]:
    """Re-solve the forward problem at one perturbed circuit current."""
    currents = np.array(base_current, dtype=float)
    currents[circuit] += sign * step
    started = perf_counter()
    equilibrium, trips, _program = solver._forward(profile, warm, currents)
    wall_s = perf_counter() - started
    points = _points(achieved_target(profile, equilibrium.flux))
    return {
        "current_a": float(currents[circuit]),
        "delta_a": float(sign * step),
        "converged": bool(np.asarray(equilibrium.fixed_point.converged)),
        "terminal_residual": float(np.asarray(equilibrium.fixed_point.residual)),
        "trips": int(trips),
        "wall_s": float(wall_s),
        "turning_points_m": points.tolist(),
        "topology": _topology_summary(profile, equilibrium.flux),
        "flux": np.asarray(equilibrium.flux, dtype=float),
    }


def _measured_flux(record: dict[str, Any]) -> np.ndarray:
    """Return one solve's flux state without giving the flux to the receipt."""
    return np.asarray(record["flux"], dtype=float)


def _circuit_entry(
    circuit: int,
    names: dict[int, str],
    frozen: np.ndarray,
    measured: dict[float, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Assemble the per-extremum comparison for one circuit."""
    central = {}
    for step in STEPS_A:
        plus = np.asarray(measured[step]["plus"]["turning_points_m"], dtype=float)
        minus = np.asarray(measured[step]["minus"]["turning_points_m"], dtype=float)
        central[step] = 0.5 * (plus - minus) / step
    common = {step: central[step].mean(axis=0) for step in STEPS_A}
    frozen_points = np.reshape(np.asarray(frozen, dtype=float), (len(EXTREMA), 2))
    rows = []
    for index, name in enumerate(EXTREMA):
        entry = {"extremum": name}
        entry["frozen_tangent_m_per_a"] = _vector(frozen_points[index])
        for step in STEPS_A:
            value = central[step][index]
            entry[f"self_consistent_{int(step)}_m_per_a"] = _vector(value)
            entry[f"differential_{int(step)}_m_per_a"] = _vector(value - common[step])
            denominator = float(np.linalg.norm(frozen_points[index]))
            entry[f"ratio_{int(step)}_self_consistent_over_frozen"] = (
                float(np.linalg.norm(value) / denominator)
                if denominator > 0.0
                else None
            )
        rows.append(entry)
    return {
        "circuit": int(circuit),
        "family": names.get(int(circuit)),
        "label": _circuit_label(int(circuit), names),
        "central_difference_table": rows,
        "whole_plasma_motion_m_per_a": {
            str(int(step)): _vector(common[step]) for step in STEPS_A
        },
        "solves": {
            str(int(step)): {
                key: _public_solve(measured[step][key]) for key in ("plus", "minus")
            }
            for step in STEPS_A
        },
    }


def _public_solve(record: dict[str, Any]) -> dict[str, Any]:
    """Return one solve record without its device flux state."""
    return {key: value for key, value in record.items() if key != "flux"}


def _manifest_table(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the largest and smallest ratio over every circuit and extremum."""
    pairs = []
    for entry in entries:
        for row in entry["central_difference_table"]:
            ratio = row["ratio_500_self_consistent_over_frozen"]
            if ratio is None or not np.isfinite(float(ratio)):
                continue
            pairs.append(
                {
                    "circuit": entry["label"],
                    "family": entry["family"],
                    "extremum": row["extremum"],
                    "ratio_500_self_consistent_over_frozen": float(ratio),
                    "frozen_tangent_norm_m_per_a": row["frozen_tangent_m_per_a"][
                        "norm"
                    ],
                    "self_consistent_500_norm_m_per_a": row[
                        "self_consistent_500_m_per_a"
                    ]["norm"],
                }
            )
    if not pairs:
        return {
            "compared_pairs": 0,
            "largest_ratio": None,
            "smallest_ratio": None,
            "reading": "no finite ratio was measured",
        }
    return {
        "compared_pairs": len(pairs),
        "largest_ratio": max(
            pairs, key=lambda item: item["ratio_500_self_consistent_over_frozen"]
        ),
        "smallest_ratio": min(
            pairs, key=lambda item: item["ratio_500_self_consistent_over_frozen"]
        ),
    }


def _environment() -> dict[str, Any]:
    """Return the runtime identity of one measurement."""
    return {
        "source_commit": _source_revision(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "scheduler": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        },
    }


def measure(directory: Path = DIRECTORY) -> dict[str, Any]:
    """Re-solve at symmetric current steps on every free circuit and compare."""
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "extended precision was not enabled before array construction"
        )
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    selected_row, qualification = selected[TARGET]
    case, context = _mast_case_from_selection(SHOT_STORE, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    names = _circuit_names(policy)
    machine = ForwardMachine(
        profile=profile,
        seed=jnp.asarray(passive_case["state"]),
        wall=np.asarray(profile.operator.wall.coordinate),
        identity="mast-22086/43",
        drivable_circuits=tuple(range(int(policy["active_circuit_count"]))),
    )
    solver = ProductionSolver(machine)
    free = [int(circuit) for circuit in machine.drivable_circuits]
    base_current = np.array(solver.prescribed_current, dtype=float)

    started = perf_counter()
    prime, _trips, _program = solver._forward(profile, machine.seed, base_current)
    prime_flux = np.asarray(prime.flux, dtype=float)
    prime_points = _points(achieved_target(profile, prime.flux))
    frozen = np.asarray(
        turning_point_response_matrix(profile, prime.flux, free), dtype=float
    )
    if frozen.shape != (2 * len(EXTREMA), len(free)):
        raise RuntimeError(f"unexpected frozen tangent shape {frozen.shape}")

    lattice = profile.lattice
    states: dict[str, np.ndarray] = {"prime": prime_flux}
    payload: dict[str, Any] = {
        "machine": machine.identity,
        "environment": _environment(),
        "carrier": carrier_evidence,
        "route": str(getattr(solver, "route", "")),
        "steps_a": [float(step) for step in STEPS_A],
        "frozen_tangent_current_step_a": float(TURNING_POINT_TANGENT_STEP_A),
        "extrema": list(EXTREMA),
        "free_circuits": free,
        "seed": {
            "turning_points_m": prime_points.tolist(),
            "converged": bool(np.asarray(prime.fixed_point.converged)),
            "terminal_residual": float(np.asarray(prime.fixed_point.residual)),
            "topology": _topology_summary(profile, prime.flux),
            "current_a": {
                str(circuit): float(base_current[circuit]) for circuit in free
            },
        },
        "circuits": [],
        "manifest_table": _manifest_table([]),
        "convergence": {"perturbed_solves": 0, "non_converged": []},
    }

    entries: list[dict[str, Any]] = []
    for position, circuit in enumerate(free):
        measured: dict[float, dict[str, dict[str, Any]]] = {}
        for step in STEPS_A:
            measured[step] = {}
            for sign, key in ((1.0, "plus"), (-1.0, "minus")):
                record = _solve_at(
                    solver, profile, prime.flux, base_current, circuit, step, sign
                )
                states[f"circuit_{circuit:02d}_{key}_{int(step)}"] = _measured_flux(
                    record
                )
                measured[step][key] = record
                payload["convergence"]["perturbed_solves"] += 1
                if not record["converged"]:
                    payload["convergence"]["non_converged"].append(
                        {
                            "circuit": circuit,
                            "delta_a": record["delta_a"],
                            "terminal_residual": record["terminal_residual"],
                            "topology": record["topology"]["class"],
                        }
                    )
        entries.append(_circuit_entry(circuit, names, frozen[:, position], measured))
        payload["circuits"] = entries
        payload["manifest_table"] = _manifest_table(entries)
        payload["runtime"] = {"wall_s": perf_counter() - started}
        _write(directory / RECEIPT_NAME, payload)
        print(
            f"LANDED circuit {circuit} after {perf_counter() - started:.1f} s",
            flush=True,
        )

    np.savez(
        directory / STATES_NAME,
        lattice_radius=np.asarray(lattice.radius, dtype=float),
        lattice_height=np.asarray(lattice.height, dtype=float),
        wall=np.asarray(profile.operator.wall.coordinate, dtype=float),
        **states,
    )
    payload["runtime"] = {
        "wall_s": perf_counter() - started,
        "states": STATES_NAME,
        "figure": FIGURE_NAME,
        "figure_circuit": _figure_circuit(entries),
    }
    _write(directory / RECEIPT_NAME, payload)
    print("MANIFEST-TABLE " + json.dumps(payload["manifest_table"]), flush=True)
    print("CONVERGENCE " + json.dumps(payload["convergence"]), flush=True)
    return payload


def _figure_circuit(entries: list[dict[str, Any]]) -> int:
    """Return the circuit whose largest ratio the panel shows."""
    best, circuit = -np.inf, -1
    for entry in entries:
        for row in entry["central_difference_table"]:
            ratio = row["ratio_500_self_consistent_over_frozen"]
            if ratio is None or not np.isfinite(float(ratio)):
                continue
            if float(ratio) > best:
                best, circuit = float(ratio), int(entry["circuit"])
    return circuit


def _nulls(summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return one stored topology summary's axis and admitted X-point."""
    axis = np.asarray(summary["axis_m"], dtype=float)[None, :]
    point = np.asarray(summary["x_point_m"], dtype=float)[None, :]
    if not summary["diverted"]:
        point = np.empty((0, 2), dtype=float)
    return axis, point


def render(directory: Path = DIRECTORY) -> Path:
    """Draw the seed beside the largest-ratio circuit's plus-2000 A state."""
    payload = json.loads((directory / RECEIPT_NAME).read_text(encoding="utf-8"))
    states = np.load(directory / STATES_NAME)
    radius = np.asarray(states["lattice_radius"], dtype=float)
    height = np.asarray(states["lattice_height"], dtype=float)
    wall = np.asarray(states["wall"], dtype=float).reshape(-1, 2)
    shape = (radius.size, height.size)
    node_count = shape[0] * shape[1]

    circuit = int(payload["runtime"]["figure_circuit"])
    step = int(STEPS_A[-1])
    seed_flux = np.asarray(states["prime"], dtype=float)[:node_count].reshape(shape)
    driven_flux = np.asarray(states[f"circuit_{circuit:02d}_plus_{step}"], dtype=float)[
        :node_count
    ].reshape(shape)
    entry = next(
        item for item in payload["circuits"] if int(item["circuit"]) == circuit
    )
    driven = entry["solves"][str(step)]["plus"]
    seed_topology = payload["seed"]["topology"]
    driven_topology = driven["topology"]

    combined = np.concatenate(
        (
            seed_flux[np.isfinite(seed_flux)],
            driven_flux[np.isfinite(driven_flux)],
        )
    )
    levels = np.linspace(float(np.min(combined)), float(np.max(combined)), 24)

    driven_title = (
        f"{entry['label']} plus {step} A\n"
        f"residual {float(driven['terminal_residual']):.2e} "
        f"converged {bool(driven['converged'])} "
        f"topology {driven_topology['class']}"
    )
    style = ink.DEFAULT_INK
    reference_style = style.variant(
        axis_marker="^", axis_color="#888888", xpoint_marker="x", xpoint_color="#888888"
    )
    figure, axes_pair = plt.subplots(
        1, 2, figsize=(9.0, 5.4), dpi=style.figure_dpi, facecolor=style.figure_facecolor
    )
    for axes, flux, summary, label in (
        (axes_pair[0], seed_flux, seed_topology, "seed (unperturbed)"),
        (
            axes_pair[1],
            driven_flux,
            driven_topology,
            f"{entry['label']} plus {step} A",
        ),
    ):
        ink.poloidal_axes(axes, style)
        poloidal.draw_flux_contours(
            axes, radius, height, flux.T, levels, color=style.contour_color
        )
        poloidal.draw_wall(axes, radius=wall[:, 0], height=wall[:, 1])
        axis, point = _nulls(summary)
        if axes is axes_pair[1]:
            seed_axis, seed_point = _nulls(seed_topology)
            poloidal.draw_nulls(
                axes,
                magnetic_axis=seed_axis[0],
                x_points=seed_point,
                style=reference_style,
            )
        poloidal.draw_nulls(axes, magnetic_axis=axis[0], x_points=point)
        axes.set_title(
            driven_title if axes is axes_pair[1] else label,
            fontsize=style.label_fontsize,
        )
    axes_pair[0].set_xlabel(
        "grey open nulls = seed; red solid nulls = perturbed state",
        fontsize=style.label_fontsize,
    )
    figure.suptitle(
        "self-consistent turning-point response, MAST 22086/43, "
        f"{len(payload['circuits'])} free circuits",
        fontsize=style.label_fontsize,
    )
    figure.tight_layout()
    target = directory / FIGURE_NAME
    figure.savefig(target, facecolor=style.figure_facecolor)
    plt.close(figure)
    return target


def main(argv: list[str] | None = None) -> int:
    """Run the measurement, or redraw the panel from its persisted receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument(
        "--render",
        action="store_true",
        help="redraw the panel from the persisted receipt; opens no machine",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the manifest table to stdout"
    )
    arguments = parser.parse_args(argv)
    if arguments.render:
        print(f"FIGURE {render(arguments.directory)}", flush=True)
        return 0
    payload = measure(arguments.directory)
    if arguments.json:
        print(json.dumps(payload["manifest_table"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
