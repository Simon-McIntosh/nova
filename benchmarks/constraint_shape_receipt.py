"""Steer a bank equilibrium's shape through the isoflux control rows.

One converged bank row is fitted to a Miller-parametrised target boundary and
then re-solved twice from that equilibrium, once with the target's geometric
height raised and once with its elongation raised.  Each arm reports the
trips it took, the compensating current every driven circuit carried, the
achieved boundary against the commanded one, and how far the magnetic axis
and the X-point moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import settled_mask_stall as settled
from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintPair,
    IsofluxConstraint,
    sample_lattice_flux,
    derive_circuit_compensators,
    miller_boundary_points,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = (
    ROOT / "docs/figures/constraint-augmented-newton-krylov/shape-control"
)
#: The bank row the shape rows are steered on.
ROW = (22086, 43)
#: Poloidal control points on the target boundary.
CONTROL_POINTS = 6
#: Commanded moves: the geometric height raised by two centimetres and the
#: elongation raised by five percent, both from the fitted target.
VERTICAL_STEP_M = 0.02
ELONGATION_FRACTION = 0.05


def _strict_float(value: Any) -> float | None:
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _boundary_polygon(profile, flux, *, angles=181):
    """Ray-cast the achieved boundary contour outward from the magnetic axis."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    axis = np.asarray(topology.axis, dtype=float)
    level = float(np.asarray(topology.boundary_flux))
    polarity = float(profile.operator.polarity)
    lattice = profile.lattice
    grid = jnp.reshape(jnp.asarray(flux)[: lattice.node_count], lattice.shape)
    reach = 0.5 * min(
        float(lattice.radius[-1] - lattice.radius[0]),
        float(lattice.height[-1] - lattice.height[0]),
    )
    theta = 2.0 * np.pi * np.arange(angles) / angles
    points = []
    for angle in theta:
        ray = np.asarray([np.cos(angle), np.sin(angle)])
        low, high = 0.0, reach
        for _step in range(48):
            middle = 0.5 * (low + high)
            value = float(
                np.asarray(
                    sample_lattice_flux(lattice, grid, jnp.asarray(axis + middle * ray))
                )
            )
            if polarity * (value - level) > 0.0:
                low = middle
            else:
                high = middle
        points.append(axis + 0.5 * (low + high) * ray)
    return np.asarray(points)


def _miller_figures(boundary: np.ndarray) -> dict[str, float]:
    """Fit the Miller shape figures to one achieved boundary polygon."""
    radius, height = boundary[:, 0], boundary[:, 1]
    inner, outer = float(np.min(radius)), float(np.max(radius))
    lower, upper = float(np.min(height)), float(np.max(height))
    minor = 0.5 * (outer - inner)
    geometric_radius = 0.5 * (outer + inner)
    return {
        "geometric_radius": geometric_radius,
        "geometric_height": 0.5 * (upper + lower),
        "minor_radius": minor,
        "elongation": 0.5 * (upper - lower) / minor,
        "triangularity": (geometric_radius - float(radius[int(np.argmax(height))]))
        / minor,
    }


def _control_points(figures: dict[str, float]) -> np.ndarray:
    """Return the commanded control points of one set of shape figures."""
    return np.asarray(miller_boundary_points(count=CONTROL_POINTS, **figures))


def _reference_point(profile, flux) -> np.ndarray:
    """Return the point whose flux the isoflux rows are measured against."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    saddle = np.asarray(topology.x_point, dtype=float)
    if np.all(np.isfinite(saddle)):
        return saddle
    return np.asarray(topology.wall_point, dtype=float)


def _isoflux_pair(profile, flux, points, *, span, tolerance_wb):
    """Return an isoflux pair whose direction the response matrix decided."""
    rows = int(points.shape[0])
    payload = jnp.concatenate(
        (
            jnp.asarray(points, dtype=jnp.float64),
            jnp.asarray(_reference_point(profile, flux))[None],
        )
    )
    return ConstraintPair(
        functional=IsofluxConstraint(point_count=rows, reference="reference_point"),
        unknown=CircuitCurrentUnknown(
            direction=jnp.zeros((_circuit_count(profile), rows)).at[0].set(1.0),
            ampere_scale=jnp.full((rows,), 1.0e3),
        ),
        binding=ConstraintBinding(
            target=jnp.zeros(rows),
            tolerance=jnp.full((rows,), tolerance_wb),
            scale=jnp.full((rows,), span),
            initial_unknown=jnp.zeros(rows),
            payload=payload,
        ),
    )


def _circuit_count(profile) -> int:
    """Return the number of prescribed circuit columns the operator carries."""
    field = profile.operator.prescribed_current_field
    if field is None:
        raise RuntimeError("the persisted prescribed-current field is unavailable")
    return int(field.circuit_count)


def _landmarks(profile, flux) -> dict[str, float | None]:
    """Return the axis and saddle positions one flux state reads."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    axis = np.asarray(topology.axis, dtype=float)
    saddle = np.asarray(topology.x_point, dtype=float)
    return {
        "axis_radius_m": _strict_float(axis[0]),
        "axis_height_m": _strict_float(axis[1]),
        "x_point_radius_m": _strict_float(saddle[0]),
        "x_point_height_m": _strict_float(saddle[1]),
        "axis_flux_wb": _strict_float(topology.axis_flux),
        "boundary_flux_wb": _strict_float(topology.boundary_flux),
    }


def _arm(
    name,
    profile,
    previous_flux,
    figures,
    *,
    target_current,
    requested,
    circuits,
    names,
    span,
    tolerance_wb,
):
    """Solve one commanded shape move warm-started from the previous state."""
    points = _control_points(figures)
    pair = _isoflux_pair(
        profile, previous_flux, points, span=span, tolerance_wb=tolerance_wb
    )
    (derived,), selection = derive_circuit_compensators(
        profile,
        (pair,),
        jnp.asarray(previous_flux),
        requested_class=requested,
        target_current=target_current,
        circuits=circuits,
    )
    branch = profile.solve_branch(
        jnp.asarray(previous_flux),
        requested,
        target_current=target_current,
        constraint_pairs=(derived,),
    )
    branch.equilibrium.flux.block_until_ready()
    equilibrium = branch.equilibrium
    record = equilibrium.constraints[0]
    direction = np.asarray(derived.unknown.direction)
    delta = direction @ np.asarray(record.physical_unknown)
    achieved = _boundary_polygon(profile, equilibrium.flux)
    error = np.asarray(record.physical_residual) / span
    return {
        "arm": name,
        "commanded_figures": {key: float(value) for key, value in figures.items()},
        "control_points_rz_m": points.tolist(),
        "selection": {
            "rule": selection.rule.name.lower(),
            "competing_rows": bool(selection.competing),
            "singular_values_row_scales_per_ampere": [
                float(value) for value in np.asarray(selection.singular_values)
            ],
            "leading_circuits_per_row": [
                [
                    {"circuit": int(index), "family": names.get(int(index))}
                    for index in selection.leading_circuits(row, count=3)
                ]
                for row in range(int(points.shape[0]))
            ],
        },
        "qualified": bool(np.asarray(branch.converged)),
        "topology_consistent": bool(np.asarray(branch.topology_consistent)),
        "terminal_residual": _strict_float(branch.residual),
        "trips": int(np.asarray(equilibrium.fixed_point.active_set_iterations)),
        "termination": settled._termination_name(
            equilibrium.fixed_point.termination_reason
        ),
        "rows_qualified": bool(np.all(np.asarray(record.qualified))),
        "row_flux_error_wb": [
            float(value) for value in np.asarray(record.physical_residual)
        ],
        "row_tolerance_wb": float(np.asarray(record.tolerance)[0]),
        "row_relative_error": [float(value) for value in error],
        "max_row_relative_error": float(np.max(np.abs(error))),
        "compensating_current_a": [
            {
                "circuit": int(index),
                "family": names.get(int(index)),
                "current_a": float(delta[index]),
            }
            for index in np.argsort(np.abs(delta))[::-1][:10]
            if abs(float(delta[index])) > 1.0e-6 * float(np.max(np.abs(delta)))
        ],
        "compensating_current_norm_a": float(np.linalg.norm(delta)),
        "achieved_figures": _miller_figures(achieved),
        "landmarks": _landmarks(profile, equilibrium.flux),
        "achieved_boundary_rz_m": achieved.tolist(),
    }


def _render(receipt, output: Path):
    """Overlay the commanded, achieved and previous boundary for both arms."""
    arms = receipt["arms"]
    figure, axes = plt.subplots(1, len(arms), figsize=(5.6 * len(arms), 5.6))
    axes = np.atleast_1d(axes)
    previous = np.asarray(receipt["converged"]["boundary_rz_m"])
    for axis, arm in zip(axes, arms, strict=True):
        commanded = np.asarray(
            miller_boundary_points(count=361, **arm["commanded_figures"])
        )
        achieved = np.asarray(arm["achieved_boundary_rz_m"])
        control = np.asarray(arm["control_points_rz_m"])
        axis.plot(
            previous[:, 0],
            previous[:, 1],
            color="0.55",
            linewidth=1.2,
            label="previous boundary",
        )
        axis.plot(
            commanded[:, 0],
            commanded[:, 1],
            color="tab:orange",
            linestyle="--",
            linewidth=1.4,
            label="commanded target",
        )
        axis.plot(
            achieved[:, 0],
            achieved[:, 1],
            color="tab:blue",
            linewidth=1.6,
            label="achieved boundary",
        )
        axis.plot(
            control[:, 0],
            control[:, 1],
            "o",
            color="tab:orange",
            markersize=5,
            label="control points",
        )
        axis.set_aspect("equal")
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.grid(alpha=0.2)
        axis.set_title(
            f"{arm['arm']}\n{arm['trips']} trips, "
            f"{arm['compensating_current_norm_a'] / 1.0e3:.2f} kA compensator"
        )
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        f"Isoflux shape steering on {receipt['identity']}: "
        "commanded Miller target against the achieved boundary",
        y=0.98,
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.86, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def measure(*, directory: Path, cache_root: Path | None = None):
    """Steer the bank row's shape through two commanded moves."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
    )
    response_cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    selected_row, qualification = selected[ROW]
    case, context = settled._mast_case_from_selection(
        settled.SHOT_STORE, selected_row, qualification
    )
    passive_case, profile, policy = settled._passive_inclusive_case(
        case, context, response_cache
    )
    names = {
        int(item["stored_circuit"]) - 1: str(item["family"])
        for item in policy["active_mapping"]
    }
    circuits = sorted(names)
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    seed = jnp.asarray(passive_case["state"])
    span = float(passive_case["span_wb"])
    tolerance_wb = 1.0e-6 * span
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    identity = f"{ROW[0]}/{ROW[1]}"
    print(f"SHAPE-CONTROL {identity} unconstrained solve", flush=True)
    base = profile.solve_branch(seed, requested, target_current=target_current)
    base.equilibrium.flux.block_until_ready()
    base_flux = base.equilibrium.flux
    boundary = _boundary_polygon(profile, base_flux)
    fitted = _miller_figures(boundary)
    receipt = {
        "receipt": "shape-control rows steered on one bank equilibrium",
        "identity": identity,
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "route": "ForwardProfile.solve_branch public defaults",
            "constraint_policy": "imposed",
            "control_points": CONTROL_POINTS,
            "isoflux_reference": "reference_point at the read saddle",
            "compensating_direction": "derive_circuit_compensators restricted to "
            "the machine active mapping; the singular-distribution rule takes "
            "over from dominant authority when the rows compete",
            "drivable_circuits": [
                {"circuit": int(index), "family": names[index]} for index in circuits
            ],
            "prescribed_circuit_count": _circuit_count(profile),
            "row_tolerance_wb": tolerance_wb,
            "flux_span_wb": span,
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "inputs": {"carrier_evidence": carrier_evidence},
        "converged": {
            "qualified": bool(np.asarray(base.converged)),
            "topology_consistent": bool(np.asarray(base.topology_consistent)),
            "terminal_residual": _strict_float(base.residual),
            "trips": int(
                np.asarray(base.equilibrium.fixed_point.active_set_iterations)
            ),
            "fitted_figures": fitted,
            "landmarks": _landmarks(profile, base_flux),
            "boundary_rz_m": boundary.tolist(),
        },
        "arms": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "converged.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    commands = (
        (
            "geometric height plus 2 cm",
            "vertical-shift",
            dict(fitted, geometric_height=fitted["geometric_height"] + VERTICAL_STEP_M),
        ),
        (
            "elongation plus 5 percent",
            "elongation-raise",
            dict(
                fitted,
                elongation=fitted["elongation"] * (1.0 + ELONGATION_FRACTION),
            ),
        ),
    )
    for label, slug, figures in commands:
        print(f"SHAPE-CONTROL {identity} {slug}", flush=True)
        arm = _arm(
            label,
            profile,
            base_flux,
            figures,
            target_current=target_current,
            requested=requested,
            circuits=circuits,
            names=names,
            span=span,
            tolerance_wb=tolerance_wb,
        )
        receipt["arms"].append(arm)
        (directory / f"{slug}.json").write_text(
            json.dumps({**receipt, "arms": [arm]}, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "SHAPE-CONTROL-ARM "
            + json.dumps(
                {
                    key: value
                    for key, value in arm.items()
                    if key not in ("achieved_boundary_rz_m", "control_points_rz_m")
                },
                sort_keys=True,
            ),
            flush=True,
        )
    (directory / "two-arms.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    _render(receipt, directory / "shape-steering.png")
    print("SHAPE-CONTROL-DONE", flush=True)
    return receipt


def main(argv=None):
    """Run the shape-steering receipt from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--cache-root", type=Path, default=None)
    arguments = parser.parse_args(argv)
    measure(directory=arguments.directory, cache_root=arguments.cache_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
