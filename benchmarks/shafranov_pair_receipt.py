"""Impose the Shafranov row on the bank rows against the profiles' own moments.

Each bank row is read for the combination the extracted profiles imply,
``beta_p + l_i/2`` from the moment observation on the row's own reference
state and plasma current.  That value is then imposed as a constraint target
on the external-magnetics Shafranov row, whose compensating unknown is the
profile normalisation the source term already carries.  The receipt records,
per row, the combination the row achieved, the compensating fraction, the
outer step count, the terminal residual and the converged flag, so an unmet
target is reported rather than fitted away.

One poloidal panel per row is written under the project's plotting rules:
line contours only, no axes or grid, both the reference and the terminal
state's nulls drawn in their own styles, and the wall on every panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from benchmarks import settled_mask_stall as settled
from nova.equilibrium.constraint import (
    ConstraintBinding,
    ConstraintContext,
    ConstraintPair,
    ExternalShafranovConstraint,
    ProfileAmplitudeUnknown,
)
from nova.equilibrium.source import PolynomialFluxFunction
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.equilibrium.wall_mask import WallUnit
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "docs/figures/constraint-augmented-newton-krylov/shafranov"
#: Row tolerance on the combination, stated in the row's own physical scale.
ROW_TOLERANCE = 1.0e-6
#: Display raster resolution for the per-row panels.
RASTER_SAMPLES = 181


def _strict_float(value: Any) -> float | None:
    """Return a finite float or ``None`` so JSON carries no NaN."""
    if value is None:
        return None
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _selection() -> dict[tuple[int, int], Any]:
    return {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }


def _minor_radius(boundary: np.ndarray) -> float:
    """Return the reference boundary's geometric minor radius [m].

    The stored last closed flux surface is the geometry the row is stated
    against, so its own horizontal half-width is the radius that enters the
    Shafranov logarithm.  A figure of the plasma, not an operand: the row
    carries it in the payload and reports it beside every receipt.
    """
    points = np.asarray(boundary, dtype=float).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 2:
        raise ValueError("the stored boundary carries too few finite nodes")
    return 0.5 * float(np.ptp(points[:, 0]))


def _combination(profile, flux, target_current) -> float:
    """Return ``beta_p + l_i/2`` the extracted profiles imply at one state."""
    observation = profile.integral_observation(flux, target_current)
    beta = float(np.asarray(observation.poloidal_beta))
    inductance = float(np.asarray(observation.internal_inductance))
    return beta + 0.5 * inductance


def _external_image(profile) -> np.ndarray:
    """Return the prescribed conductor flux on the lattice's own nodes [Wb]."""
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None:
        raise RuntimeError("the row needs a prescribed conductor field")
    flux = np.asarray(prescribed.flux(), dtype=float)
    return flux[: profile.lattice.node_count]


def _pair(profile, *, target: float, minor_radius: float) -> tuple[Any, np.ndarray]:
    """Return the Shafranov row and the external image it is stated against."""
    external = _external_image(profile)
    functional = ExternalShafranovConstraint(
        minor_radius=jnp.asarray(minor_radius),
    )
    binding = ConstraintBinding(
        target=jnp.atleast_1d(jnp.asarray(target)),
        tolerance=jnp.asarray([ROW_TOLERANCE]),
        scale=jnp.asarray([1.0]),
        initial_unknown=jnp.asarray([0.0]),
        payload=(jnp.asarray(external), jnp.asarray(minor_radius)),
        policy="imposed",
    )
    unknown = ProfileAmplitudeUnknown("pressure_gradient", jnp.asarray([1.0]))
    return ConstraintPair(functional, unknown, binding), external


def _amplitude_availability(profile, component: str) -> str | None:
    """Return the refusal text when the amplitude is not a free unknown.

    The compensator moves the state through the source term's own scalar
    normalisation, and only a polynomial flux function carries one.  An
    extracted profile is an arbitrary callable, so the row states a constraint
    nothing on that row can move and must be refused rather than fitted.
    """
    field = "p_prime" if component == "pressure_gradient" else "ff_prime"
    function = getattr(profile.source.core, field, None)
    if isinstance(function, PolynomialFluxFunction):
        return None
    return (
        f"the source core's {field} is "
        f"{type(function).__name__}, not a PolynomialFluxFunction, so the "
        "profile amplitude is not a free unknown of this solve"
    )


def _observed_combination(profile, pair, flux) -> float:
    """Return the combination the row itself reads at one state."""
    context = ConstraintContext(jnp.asarray(flux), None, None, None)
    observed = pair.functional.observed(profile, context, pair.binding.payload)
    return float(np.asarray(jnp.atleast_1d(observed))[0])


def _topology(operator, state) -> dict[str, Any]:
    """Return the read nulls and boundary, or a recorded refusal."""
    try:
        _masks, topology = operator.read(jnp.asarray(state))
    except NoQualifiedAxisError as error:
        return {"read_status": "no_qualified_axis", "exception_text": str(error)}
    return {
        "read_status": "qualified",
        "class": str(topology.topology_class),
        "boundary_rz_m": np.asarray(topology.boundary, dtype=float).tolist(),
        "axis_rz_m": np.asarray(topology.axis, dtype=float).reshape(-1)[:2].tolist(),
        "x_point_rz_m": np.asarray(topology.x_point, dtype=float)
        .reshape(-1, 2)
        .tolist(),
    }


def _wall_units(operator) -> tuple[Any, ...]:
    """Return the operator's wall as its own typed units.

    The wall is stored flat with unit offsets and per-unit closure and kind,
    so a panel can draw every unit on its own terms rather than one invented
    ring: an open material unit is dashed and is never joined to a neighbour.
    """
    coordinate = np.asarray(operator.wall.coordinate, dtype=float).reshape(-1, 2)
    offsets = np.asarray(operator.wall_unit_offsets, dtype=int)
    closed = np.asarray(operator.wall_unit_closed, dtype=bool)
    kinds = tuple(operator.wall_unit_kinds)
    return tuple(
        WallUnit(
            coordinate[start:stop, 0],
            coordinate[start:stop, 1],
            kind=kinds[index],
            closed=bool(closed[index]),
        )
        for index, (start, stop) in enumerate(
            zip(offsets[:-1], offsets[1:], strict=True)
        )
    )


def _raster(profile, state, units, *, samples: int = RASTER_SAMPLES):
    """Interpolate one state onto a display raster for line contours.

    The state vector is a stitched target: the plasma grid block first, then
    the wall nodes and any direct sample rows.  The lattice coordinate spans
    the grid block alone, so the grid prefix is the block to contour.  The
    raster is framed on the grid nodes and the wall together, so the vessel
    is not clipped out of its own panel.
    """
    points = np.asarray(profile.lattice.coordinate, dtype=float)
    field = np.asarray(state, dtype=float).reshape(-1)[: points.shape[0]]
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(field)
    points, field = points[finite], field[finite]
    if points.shape[0] < 3:
        raise ValueError("the state carries too few finite samples to contour")
    limits = np.vstack(
        (points, *[np.asarray(unit.vertices, dtype=float) for unit in units])
    )
    radial = np.linspace(
        float(np.min(limits[:, 0])), float(np.max(limits[:, 0])), samples
    )
    height = np.linspace(
        float(np.min(limits[:, 1])), float(np.max(limits[:, 1])), samples
    )
    radius_grid, height_grid = np.meshgrid(radial, height)
    raster = LinearNDInterpolator(points, field, fill_value=np.nan)(
        radius_grid, height_grid
    )
    return radial, height, np.asarray(raster, dtype=float)


def _render(profile, *, reference, terminal, units, path: Path, title: str) -> dict:
    """Draw the reference and terminal states as shared-level line contours."""
    radial, height, reference_field = _raster(profile, reference, units)
    _, _, terminal_field = _raster(profile, terminal, units)
    levels = poloidal.contour_levels(reference_field, count=12)
    reference_topology = _topology(profile.operator, reference)
    terminal_topology = _topology(profile.operator, terminal)
    figure, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    poloidal.draw_flux_contours(
        axis, radial, height, reference_field, levels, color="#3366cc"
    )
    poloidal.draw_flux_contours(
        axis, radial, height, terminal_field, levels, color="#cc7722"
    )
    poloidal.draw_wall(axis, units=units)
    for topology, style in (
        (
            reference_topology,
            DEFAULT_INK.variant(
                axis_color="#3366cc",
                xpoint_color="#3366cc",
                axis_marker="^",
                xpoint_marker="P",
            ),
        ),
        (
            terminal_topology,
            DEFAULT_INK.variant(
                axis_color="#cc7722",
                xpoint_color="#cc7722",
                axis_marker="^",
                xpoint_marker="X",
            ),
        ),
    ):
        if topology.get("read_status") != "qualified":
            continue
        poloidal.draw_nulls(
            axis,
            magnetic_axis=topology["axis_rz_m"],
            x_points=np.asarray(topology["x_point_rz_m"], dtype=float),
            style=style,
            contain=units,
        )
    poloidal_axes(axis)
    axis.set_title(
        f"{title}\nreference blue / terminal orange, shared levels", fontsize=8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "filesystem_path": str(path),
        "project_absolute_src": (
            f"/nova/figures/constraint-augmented-newton-krylov/shafranov/{path.name}"
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _row_receipt(
    profile,
    *,
    identity: str,
    reference_state,
    target_current,
    minor_radius: float,
    requested,
    directory: Path,
) -> dict[str, Any]:
    """Impose the Shafranov row on one bank row and report the outcome.

    The row is stated against the reference boundary's own minor radius and
    the prescribed conductor image.  Where the profile amplitude is not a free
    unknown of the solve the row is refused, and the receipt records the
    refusal beside the gap between the combination the external magnetics
    imply and the combination the extracted profiles carry -- the number the
    row exists to close.
    """
    target = _combination(profile, reference_state, target_current)
    pair, _external = _pair(profile, target=target, minor_radius=minor_radius)
    refusal = _amplitude_availability(profile, "pressure_gradient")
    print(f"SHAFRANOV {identity} target={target!r} refusal={refusal!r}", flush=True)
    observed_at_reference = _observed_combination(profile, pair, reference_state)
    entry: dict[str, Any] = {
        "identity": identity,
        "status": "refused" if refusal is not None else "imposed",
        "refusal_reason": refusal,
        "target_combination": _strict_float(target),
        "minor_radius_m": _strict_float(minor_radius),
        "plasma_current_a": _strict_float(target_current),
        "target_source": (
            "poloidal_beta + internal_inductance/2 from "
            "ForwardProfile.integral_observation on the stored reference state"
        ),
        "observed_combination_at_reference": _strict_float(observed_at_reference),
        "reference_combination_gap": _strict_float(observed_at_reference - target),
        "target_error": None,
        "achieved_combination": None,
        "compensating_amplitude_fraction": None,
        "terminal_profile_combination": _strict_float(target),
        "outer_steps": 0,
        "terminal_residual": None,
        "topology_consistent": None,
        "converged": False,
        "termination": None,
    }
    terminal = np.asarray(reference_state)
    if refusal is None:
        branch = profile.solve_branch(
            jnp.asarray(reference_state),
            requested,
            target_current=target_current,
            constraint_pairs=(pair,),
        )
        equilibrium = branch.equilibrium
        flux = equilibrium.flux
        flux.block_until_ready()
        terminal = np.asarray(flux)
        records = list(equilibrium.constraints)
        record = records[0] if records else None
        terminal_combination = _combination(profile, flux, target_current)
        entry.update(
            {
                "target_error": (
                    None
                    if record is None
                    else _strict_float(abs(record.physical_residual[0]))
                ),
                "achieved_combination": (
                    None if record is None else _strict_float(record.observed[0])
                ),
                "compensating_amplitude_fraction": (
                    None
                    if record is None
                    else _strict_float(record.physical_unknown[0])
                ),
                "terminal_profile_combination": _strict_float(terminal_combination),
                "outer_steps": int(
                    np.asarray(equilibrium.fixed_point.active_set_iterations)
                ),
                "terminal_residual": _strict_float(branch.residual),
                "topology_consistent": bool(np.asarray(branch.topology_consistent)),
                "converged": bool(np.asarray(branch.converged)),
                "termination": settled._termination_name(
                    equilibrium.fixed_point.termination_reason
                ),
            }
        )
    units = _wall_units(profile.operator)
    slug = identity.replace("/", "-")
    entry["figure"] = _render(
        profile,
        reference=np.asarray(reference_state),
        terminal=terminal,
        units=units,
        path=directory / f"row-{slug}.png",
        title=f"MAST {identity}: beta_p + l_i/2 row",
    )
    return entry


def measure(*, directory: Path, cache_root: Path | None = None) -> dict[str, Any]:
    """Impose the row on every row the decomposition bank qualifies."""
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
    selected = _selection()
    directory.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "receipt": "Shafranov beta_p + l_i/2 row imposed on the MAST bank rows",
        "row_set": "every row the decomposition bank qualifies; the receipt "
        "records the full selection rather than a stated count",
        "rows": [list(key) for key in sorted(selected)],
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "route": "ForwardProfile.solve_branch public defaults",
            "constraint_policy": "imposed",
            "row": "ExternalShafranovConstraint: the external magnetics' "
            "large-aspect-ratio vertical-field identity inverted for "
            "beta_p + l_i/2 at the plasma's current centroid",
            "compensating_unknown": "ProfileAmplitudeUnknown on the "
            "pressure-gradient normalisation",
            "row_tolerance": ROW_TOLERANCE,
            "row_scale": 1.0,
            "prescribed_circuit_count": None,
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "inputs": {"carrier_evidence": carrier_evidence},
        "rows_receipt": [],
    }
    for shot, row_index in sorted(selected):
        key = (shot, row_index)
        selected_row, qualification = selected[key]
        case, context = settled._mast_case_from_selection(
            settled.SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = settled._passive_inclusive_case(
            case, context, response_cache
        )
        receipt["configuration"]["prescribed_circuit_count"] = int(
            getattr(profile.operator.prescribed_current_field, "circuit_count", 0)
        )
        reference_state = jnp.asarray(passive_case["state"])
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        minor_radius = _minor_radius(np.asarray(case["boundary"], dtype=float))
        requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
        entry = _row_receipt(
            profile,
            identity=f"{shot}/{row_index}",
            reference_state=reference_state,
            target_current=target_current,
            minor_radius=minor_radius,
            requested=requested,
            directory=directory,
        )
        entry["plasma_current_a"] = target_current
        receipt["rows_receipt"].append(entry)
        (directory / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        (directory / f"row-{shot}-{row_index}.json").write_text(
            json.dumps(entry, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "SHAFRANOV-ROW "
            + json.dumps(
                {key: value for key, value in entry.items() if key != "figure"},
                sort_keys=True,
            ),
            flush=True,
        )
    (directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print("SHAFRANOV-DONE", flush=True)
    return receipt


def main(argv=None):
    """Run the Shafranov-row receipt from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--cache-root", type=Path, default=None)
    arguments = parser.parse_args(argv)
    measure(directory=arguments.directory, cache_root=arguments.cache_root)


if __name__ == "__main__":
    main()
