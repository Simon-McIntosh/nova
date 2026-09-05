"""Steer a bank equilibrium's shape through the bounding-box control rows.

One converged bank row is given the bounding-box rows read from its own
achieved boundary's turning points — boundary flux at the outer, upper,
inner and lower points, zero radial field at the outer and inner points,
zero vertical field at the upper and lower points, and the two-gradient
null row at the X-point — and is then re-solved twice from that equilibrium,
once with the upper turning point raised two centimetres and once with the
elongation raised five percent through the upper and lower points.  Each arm
reports the trips it took, the compensating current every driven circuit
carried, the achieved turning points against the commanded ones, and how far
the magnetic axis and the X-point moved.
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
    BoundingBoxTarget,
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintPair,
    FieldComponentConstraint,
    IsofluxConstraint,
    XPointConstraint,
    derive_circuit_compensators,
    sample_lattice_flux,
)
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
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
#: Commanded moves: the upper turning point raised by two centimetres and the
#: elongation raised by five percent through the upper and lower points.
VERTICAL_STEP_M = 0.02
ELONGATION_FRACTION = 0.05
#: Budget the constrained arms are given. A commanded shape move opens rows
#: the public default budget was never sized for, so the arms state their own.
NEWTON_STEPS = 16
ACTIVE_SET_STEPS = 24


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


def _reference_point(profile, flux) -> np.ndarray:
    """Return the point whose flux the isoflux rows are measured against."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    saddle = np.asarray(topology.x_point, dtype=float)
    if np.all(np.isfinite(saddle)):
        return saddle
    return np.asarray(topology.wall_point, dtype=float)


def _circuit_count(profile) -> int:
    """Return the number of prescribed circuit columns the operator carries."""
    field = profile.operator.prescribed_current_field
    if field is None:
        raise RuntimeError("the persisted prescribed-current field is unavailable")
    return int(field.circuit_count)


def _row_scale(profile, span, kind) -> float:
    """Return the residual scale one bounding-box row kind is measured in."""
    if kind == "flux":
        return span
    if kind == "field":
        lattice = profile.lattice
        return span / (
            TOTAL_FLUX_FACTOR * float(lattice.radius[0]) * float(lattice.radial_step)
        )
    return span / float(profile.lattice.radial_step)


def _bounding_box_pairs(
    profile, target, *, span, ampere_scale=1.0e3
) -> tuple[ConstraintPair, ...]:
    """Assemble the isoflux, field and X-point pairs one target produces."""
    flux_points = np.asarray(target.flux_points, dtype=float)
    radial = np.asarray(target.radial_field_points, dtype=float)
    vertical = np.asarray(target.vertical_field_points, dtype=float)
    x_point = np.asarray(target.x_point, dtype=float)
    circuits = _circuit_count(profile)

    def pair(functional, points, kind):
        rows = functional.row_count
        scale = _row_scale(profile, span, kind)
        return ConstraintPair(
            functional=functional,
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((circuits, rows)).at[0].set(1.0),
                ampere_scale=jnp.full((rows,), ampere_scale),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(rows),
                tolerance=jnp.full((rows,), 1.0e-6 * scale),
                scale=jnp.full((rows,), scale),
                initial_unknown=jnp.zeros(rows),
                payload=jnp.asarray(points, dtype=jnp.float64),
            ),
        )

    flux_payload = jnp.concatenate(
        (
            jnp.asarray(flux_points, dtype=jnp.float64),
            jnp.asarray(target.reference_point, dtype=jnp.float64)[None],
        )
    )
    pairs = [
        ConstraintPair(
            functional=IsofluxConstraint(
                point_count=flux_points.shape[0], reference="reference_point"
            ),
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((circuits, flux_points.shape[0])).at[0].set(1.0),
                ampere_scale=jnp.full((flux_points.shape[0],), ampere_scale),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(flux_points.shape[0]),
                tolerance=jnp.full(
                    (flux_points.shape[0],), 1.0e-6 * _row_scale(profile, span, "flux")
                ),
                scale=jnp.full((flux_points.shape[0],), span),
                initial_unknown=jnp.zeros(flux_points.shape[0]),
                payload=flux_payload,
            ),
        ),
        pair(
            FieldComponentConstraint(
                components=("radial",) * radial.shape[0]
                + ("vertical",) * vertical.shape[0]
            ),
            np.concatenate((radial, vertical)),
            "field",
        ),
    ]
    if x_point.shape == (2,):
        pairs.append(pair(XPointConstraint(), x_point[None, :], "xpoint"))
    return tuple(pairs)


def _moved_target(target, *, upper_delta=0.0, lower_delta=0.0):
    """Return the same target with turning points moved by the stated deltas.

    The flux rows are ordered outer, upper, inner, lower and the vertical
    field rows upper, lower, so an upper command touches rows one and the
    first vertical-field row, a lower command rows three and the second.
    """
    flux = np.asarray(target.flux_points, dtype=float).copy()
    vertical = np.asarray(target.vertical_field_points, dtype=float).copy()
    flux[1] = flux[1] + np.asarray([0.0, upper_delta])
    flux[3] = flux[3] + np.asarray([0.0, lower_delta])
    vertical[0] = vertical[0] + np.asarray([0.0, upper_delta])
    vertical[1] = vertical[1] + np.asarray([0.0, lower_delta])
    return BoundingBoxTarget(
        flux_points=jnp.asarray(flux),
        radial_field_points=target.radial_field_points,
        vertical_field_points=jnp.asarray(vertical),
        x_point=target.x_point,
        reference_point=target.reference_point,
    )


def _achieved_turning_points(profile, flux):
    """Return the achieved boundary's own turning points as a target."""
    return BoundingBoxTarget.from_boundary(
        _boundary_polygon(profile, flux),
        x_point=None,
        reference_point=None,
    )


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


def _turning_error(target, achieved):
    """Return each commanded turning-point row against the achieved extrema."""
    commanded = np.asarray(target.flux_points, dtype=float)
    reached = np.asarray(achieved.flux_points, dtype=float)
    return np.linalg.norm(reached - commanded, axis=1).tolist()


def _arm(
    name,
    profile,
    previous_flux,
    target,
    *,
    span,
    target_current,
    requested,
    circuits,
    names,
):
    """Solve one commanded shape move warm-started from the previous state."""
    pairs = _bounding_box_pairs(profile, target, span=span)
    derived, selection = derive_circuit_compensators(
        profile,
        pairs,
        jnp.asarray(previous_flux),
        requested_class=requested,
        target_current=target_current,
        circuits=circuits,
    )
    branch = profile.solve_branch(
        jnp.asarray(previous_flux),
        requested,
        target_current=target_current,
        constraint_pairs=derived,
        newton_steps=NEWTON_STEPS,
        active_set_steps=ACTIVE_SET_STEPS,
    )
    branch.equilibrium.flux.block_until_ready()
    equilibrium = branch.equilibrium
    achieved = _achieved_turning_points(profile, equilibrium.flux)
    compensated = []
    for pair, record in zip(derived, equilibrium.constraints, strict=True):
        direction = np.asarray(pair.unknown.direction)
        delta = direction @ np.asarray(record.physical_unknown)
        compensated.append(
            {
                "row_kind": pair.functional.__class__.__name__,
                "current_a": [
                    {
                        "circuit": int(index),
                        "family": names.get(int(index)),
                        "current_a": float(delta[index]),
                    }
                    for index in np.argsort(np.abs(delta))[::-1][:6]
                    if abs(float(delta[index])) > 1.0e-6 * float(np.max(np.abs(delta)))
                ],
                "current_norm_a": float(np.linalg.norm(delta)),
            }
        )
    return {
        "arm": name,
        "commanded_turning_points": {
            "outer_m": [float(value) for value in np.asarray(target.flux_points)[0]],
            "upper_m": [float(value) for value in np.asarray(target.flux_points)[1]],
            "inner_m": [float(value) for value in np.asarray(target.flux_points)[2]],
            "lower_m": [float(value) for value in np.asarray(target.flux_points)[3]],
        },
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
                for row in range(sum(int(pair.functional.row_count) for pair in pairs))
            ],
        },
        "qualified": bool(np.asarray(branch.converged)),
        "topology_consistent": bool(np.asarray(branch.topology_consistent)),
        "terminal_residual": _strict_float(branch.residual),
        "trips": int(np.asarray(equilibrium.fixed_point.active_set_iterations)),
        "termination": settled._termination_name(
            equilibrium.fixed_point.termination_reason
        ),
        "rows_qualified": bool(
            np.all(
                [
                    bool(np.all(np.asarray(record.qualified)))
                    for record in equilibrium.constraints
                ]
            )
        ),
        "rows": [
            {
                "row_kind": pair.functional.__class__.__name__,
                "qualified": bool(np.all(np.asarray(record.qualified))),
                "max_relative_error": float(
                    np.max(np.abs(np.asarray(record.scaled_residual)))
                ),
                "physical_residual": [
                    float(value) for value in np.asarray(record.physical_residual)
                ],
            }
            for pair, record in zip(derived, equilibrium.constraints, strict=True)
        ],
        "compensated_circuits": compensated,
        "compensating_current_norm_a": float(
            np.linalg.norm(
                np.concatenate(
                    [np.asarray(item["current_norm_a"]) for item in compensated]
                )
            )
        ),
        "achieved_turning_points": {
            "outer_m": [float(value) for value in np.asarray(achieved.flux_points)[0]],
            "upper_m": [float(value) for value in np.asarray(achieved.flux_points)[1]],
            "inner_m": [float(value) for value in np.asarray(achieved.flux_points)[2]],
            "lower_m": [float(value) for value in np.asarray(achieved.flux_points)[3]],
        },
        "turning_point_error_m": _turning_error(target, achieved),
        "landmarks": _landmarks(profile, equilibrium.flux),
        "achieved_boundary_rz_m": _boundary_polygon(profile, equilibrium.flux).tolist(),
    }


def _render(receipt, output: Path):
    """Overlay the commanded, achieved and previous boundary for both arms."""
    arms = receipt["arms"]
    figure, axes = plt.subplots(1, len(arms), figsize=(5.6 * len(arms), 5.6))
    axes = np.atleast_1d(axes)
    previous = np.asarray(receipt["converged"]["boundary_rz_m"])
    anchor = np.asarray(receipt["converged"]["target"]["flux_points_rz_m"])
    for axis, arm in zip(axes, arms, strict=True):
        commanded = np.asarray(list(arm["commanded_turning_points"].values())).reshape(
            -1, 2
        )
        achieved = np.asarray(arm["achieved_boundary_rz_m"])
        axis.plot(
            previous[:, 0],
            previous[:, 1],
            color="0.55",
            linewidth=1.2,
            label="previous boundary",
        )
        axis.plot(
            achieved[:, 0],
            achieved[:, 1],
            color="tab:blue",
            linewidth=1.6,
            label="achieved boundary",
        )
        axis.plot(
            commanded[:, 0],
            commanded[:, 1],
            "o",
            color="tab:orange",
            markersize=6,
            label="commanded turning points",
        )
        axis.plot(
            anchor[:, 0],
            anchor[:, 1],
            "x",
            color="0.35",
            markersize=6,
            label="previous turning points",
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
        f"Bounding-box shape steering on {receipt['identity']}: "
        "commanded turning points against the achieved boundary",
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
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    identity = f"{ROW[0]}/{ROW[1]}"
    print(f"SHAPE-CONTROL {identity} unconstrained solve", flush=True)
    base = profile.solve_branch(seed, requested, target_current=target_current)
    base.equilibrium.flux.block_until_ready()
    base_flux = base.equilibrium.flux
    boundary = _boundary_polygon(profile, base_flux)
    x_point = np.asarray(
        profile.operator.read(jnp.asarray(base_flux))[1].x_point, dtype=float
    )
    reference = _reference_point(profile, base_flux)
    target = BoundingBoxTarget.from_boundary(
        boundary, x_point=x_point, reference_point=reference
    )
    vertical_reference = float(np.asarray(target.flux_points)[1][1]) - float(
        np.asarray(target.flux_points)[3][1]
    )
    receipt = {
        "receipt": "bounding-box shape rows steered on one bank equilibrium",
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
            "row_set": "boundary flux at the four bounding-box points, radial "
            "field at the outer and inner points, vertical field at the upper "
            "and lower points, two-gradient null at the X-point",
            "isoflux_reference": "reference_point at the read saddle",
            "compensating_direction": "derive_circuit_compensators restricted to "
            "the machine active mapping; the singular-distribution rule takes "
            "over from dominant authority when the rows compete",
            "drivable_circuits": [
                {"circuit": int(index), "family": names[index]} for index in circuits
            ],
            "prescribed_circuit_count": _circuit_count(profile),
            "row_tolerance_fraction": 1.0e-6,
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
            "target": {
                "flux_points_rz_m": np.asarray(target.flux_points).tolist(),
                "x_point_rz_m": np.asarray(target.x_point).tolist(),
                "reference_point_rz_m": np.asarray(target.reference_point).tolist(),
            },
            "landmarks": _landmarks(profile, base_flux),
            "boundary_rz_m": boundary.tolist(),
        },
        "arms": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "converged.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    elongation_delta_m = 0.5 * ELONGATION_FRACTION * vertical_reference
    commands = (
        (
            "upper point plus 2 cm",
            "vertical-shift",
            _moved_target(target, upper_delta=VERTICAL_STEP_M),
        ),
        (
            "elongation plus 5 percent through the upper and lower points",
            "elongation-raise",
            _moved_target(
                target,
                upper_delta=elongation_delta_m,
                lower_delta=-elongation_delta_m,
            ),
        ),
    )
    for label, slug, moved in commands:
        print(f"SHAPE-CONTROL {identity} {slug}", flush=True)
        arm = _arm(
            label,
            profile,
            base_flux,
            moved,
            span=span,
            target_current=target_current,
            requested=requested,
            circuits=circuits,
            names=names,
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
                    if key not in ("achieved_boundary_rz_m",)
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
