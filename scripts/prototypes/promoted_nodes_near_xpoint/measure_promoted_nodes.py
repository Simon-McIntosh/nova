"""Measure what the promoted nodes cost the converged near-X-point geometry.

The reduced solve closes each active-set trip by keeping its trial state on the
nodes the residual-shadow rule selects, ``mapped = jnp.where(promoted, state,
image)`` in :mod:`nova.equilibrium.reduced_newton`.  Those nodes carry a piece
of the iterate that no field of currents reproduces, so a converged state's
near-X-point flux is partly held rather than solved.  This script re-solves the
converged MAST boundary-coil edit through ``benchmarks/coil_edit_latency.py``'s
own preparation and reports

* the promoted nodes themselves -- position, distance to the polished X-point
  and to the drawn separatrix, the domain label the solve's own topology read
  gives each, and the residual-shadow rule that promoted it;
* ``image - state`` at those nodes, with the same difference over the nodes the
  solve does claim as its control;
* the displacement of the polished X-point, and of the separatrix's lobe and
  leg directions, when the X-point is re-polished on the field of the currents
  alone: the image everywhere.

Everything is written as one JSON receipt plus one NPZ of the fields the panel
draws, and the panel is rendered from that NPZ beside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import benchmarks.coil_edit_latency as bench  # noqa: E402
from nova.equilibrium import display_flux  # noqa: E402
from nova.equilibrium.domain import PlasmaDomain  # noqa: E402
from nova.equilibrium.flux_surface_connectivity import (  # noqa: E402
    polish_stationary_points,
)
from nova.equilibrium.separatrix_branches import (  # noqa: E402
    assemble_separatrix_branches,
)
from nova.equilibrium.topology import TopologyClass  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402
from nova.linalg.tensor_spline import fit_tensor_spline  # noqa: E402

DOMAIN_NAMES = {
    int(PlasmaDomain.CORE): "core",
    int(PlasmaDomain.PRIVATE_FLUX): "private flux",
    int(PlasmaDomain.COMMON_SOL): "open (common SOL)",
    int(PlasmaDomain.EXCLUDED_MATERIAL): "wall shadow (outside material)",
}


def _stationary_saddle(values, radius, height, seed):
    """Polish the stationary point of ``values`` nearest the seed cell.

    The polish runs on the field's own tensor spline, so the returned point is
    a stationary point of that field and never a coordinate carried in from
    another.  A polish that does not converge, or whose Hessian is not of
    saddle type, is reported absent rather than as an extremum.
    """
    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(values)
    )
    polished = polish_stationary_points(
        surface, jnp.asarray(np.asarray(seed).reshape(2))[None, :], jnp.ones((1,), bool)
    )
    position = np.asarray(polished["position_rz"], dtype=np.float64).reshape(2)
    value = float(np.asarray(polished["value"]).reshape(()))
    converged = bool(np.asarray(polished["converged"]).reshape(()))
    hessian_type = int(np.asarray(polished["hessian_type"]).reshape(()))
    if converged and hessian_type < 0 and np.all(np.isfinite(position)):
        return position, value, True
    return np.full(2, np.nan), float("nan"), False


def _branches(values, radius, height, level, axis):
    """Assemble the field's own separatrix: one closed lobe and its legs."""
    assembled = jax.device_get(
        assemble_separatrix_branches(
            jnp.asarray(values),
            jnp.asarray(radius),
            jnp.asarray(height),
            jnp.asarray(level),
            jnp.asarray(axis),
        )
    )
    return {
        "closed_controls_rz": np.asarray(assembled["closed_controls_rz"], float),
        "closed_valid": np.asarray(assembled["closed_valid"], bool),
        "open_controls_rz": np.asarray(assembled["open_controls_rz"], float),
        "open_valid": np.asarray(assembled["open_valid"], bool),
        "open_branch_valid": np.asarray(assembled["open_branch_valid"], bool),
        "well_formed": bool(np.asarray(assembled["well_formed"])),
        "boundary_flux": float(level),
    }


def _cubic_points(controls, valid, samples=8):
    """Sample the valid cubic segments of one branch into an ordered polyline."""
    points = []
    for index in np.nonzero(np.asarray(valid, dtype=bool).reshape(-1))[0]:
        control = np.asarray(controls[index], dtype=np.float64).reshape(4, 2)
        parameter = np.linspace(0.0, 1.0, samples, endpoint=index == 0)
        one_minus = 1.0 - parameter
        weight = np.stack(
            (
                one_minus**3,
                3 * one_minus**2 * parameter,
                3 * one_minus * parameter**2,
                parameter**3,
            ),
            axis=1,
        )
        points.append(weight @ control)
    if not points:
        return np.empty((0, 2))
    return np.concatenate(points)


def _segmented_distance(points, query):
    """Return the shortest distance from every query point to a polyline."""
    if points.shape[0] < 2:
        return np.full(query.shape[0], np.nan)
    start = points[:-1]
    end = points[1:]
    segment = end - start
    length_squared = np.einsum("ij,ij->i", segment, segment)
    length_squared = np.where(length_squared > 0.0, length_squared, 1.0)
    offset = query[:, None, :]
    parameter = np.einsum("qij,ij->qi", offset - start[None], segment)
    parameter = np.clip(parameter / length_squared[None], 0.0, 1.0)
    closest = start[None] + parameter[..., None] * segment[None]
    return np.min(np.linalg.norm(query[:, None, :] - closest, axis=2), axis=1)


def _direction(vector):
    """Return the unit direction of a 2-vector and its norm."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return None, norm
    return np.asarray(vector, dtype=np.float64) / norm, norm


def _angle_degrees(left, right):
    """Return the angle between two unit vectors, in degrees."""
    if left is None or right is None:
        return float("nan")
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _landmarks(branches, saddle, axis):
    """Return the lobe and leg directions one separatrix assembly defines.

    The lobe direction runs from the magnetic axis to the point of the closed
    branch furthest from it -- the lobe tip.  Each leg direction runs from the
    polished saddle to the far end of that open branch, which is where the leg
    leaves the traced field.
    """
    landmark = {"lobe_direction": None, "leg_directions": [], "lobe_tip": None}
    closed = _cubic_points(branches["closed_controls_rz"], branches["closed_valid"])
    if closed.shape[0] > 2 and np.all(np.isfinite(axis)):
        offsets = closed - np.asarray(axis, dtype=np.float64)
        radii = np.linalg.norm(offsets, axis=1)
        tip = int(np.argmax(radii))
        landmark["lobe_tip"] = closed[tip].tolist()
        landmark["lobe_direction"] = _direction(offsets[tip])[0]
    legs = []
    valid = branches["open_valid"]
    for slot in range(np.asarray(valid).shape[0]):
        points = _cubic_points(branches["open_controls_rz"][slot], valid[slot])
        if points.shape[0] < 2 or not np.all(np.isfinite(saddle)):
            continue
        reach = np.linalg.norm(points - np.asarray(saddle, float), axis=1)
        far = int(np.argmax(reach))
        legs.append(_direction(points[far] - np.asarray(saddle, float))[0])
    landmark["leg_directions"] = legs
    return landmark


def measure(edit_index: int, carrier: Path) -> tuple[dict, dict]:
    """Re-solve one sweep edit and measure its promoted nodes."""
    started = time.perf_counter()
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("extended precision is not enabled")

    profile, prepared, _carrier = bench._prepare_case(carrier)
    lattice = profile.lattice
    operator = profile.operator
    target_current = prepared["target_current"]
    requested_class = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    base_current = np.asarray(prepared["prescribed_current"], dtype=np.float64)
    circuit_index = int(prepared["circuit_index"])
    constraint_pair = prepared["vertical_centroid_pair"]
    constraint_pairs = () if constraint_pair is None else (constraint_pair,)

    state = prepared["initial"]
    program = None
    chain = []
    for index in range(edit_index + 1):
        fraction = float(bench.EDIT_FRACTIONS[index])
        values = base_current.copy()
        values[circuit_index] *= 1.0 + fraction
        current = jnp.asarray(values)
        result = bench._compiled_edit(
            profile,
            state,
            current,
            requested_class,
            target_current,
            program,
            constraint_pairs=constraint_pairs,
        )
        jax.block_until_ready(result.state)
        chain.append(
            {
                "edit_index": index,
                "edit_fraction": fraction,
                "terminal_residual": float(np.asarray(result.terminal_residual)),
                "converged": bool(np.asarray(result.converged)),
                "promoted_count": int(
                    np.count_nonzero(np.asarray(result.promoted_mask))
                ),
            }
        )
        state = result.state
        program = result.program
        terminal = result

    grid_count = int(lattice.node_count)
    mask = np.asarray(terminal.promoted_mask, dtype=bool).reshape(-1)
    # The residual shadow is composed as the interior flood over the plasma
    # grid followed by the private-wall shadow over the wall's own nodes, so
    # the mask spans both and the wall nodes carry their own coordinates.
    coordinates = np.concatenate(
        (
            np.asarray(lattice.coordinate, dtype=np.float64).reshape(-1, 2),
            np.asarray(operator.wall.coordinate, dtype=np.float64).reshape(-1, 2),
        ),
        axis=0,
    )
    if mask.size != coordinates.shape[0]:
        raise RuntimeError(
            f"mask {mask.size} names {mask.size} nodes but the grid ({grid_count}) "
            f"and wall ({coordinates.shape[0] - grid_count}) carry "
            f"{coordinates.shape[0]} coordinates"
        )
    node_count = mask.size
    state_nodes = np.asarray(state, dtype=np.float64)[:node_count]

    # The map of the currents on the lattice nodes: the field the sweep's own
    # promoted paste replaces at every node it selects.
    moments = operator.normalised_current_moments(
        state, jnp.asarray(target_current, dtype=jnp.float64), requested_class
    )[0]
    # The raster carries the plasma grid only, so the wall nodes the shadow
    # also spans are evaluated through the machine coupling the panel draws
    # with, and the grid half is checked against the raster the solve itself
    # uses: the two disagree only at the display's own reproduction error.
    raster = np.asarray(
        operator.raster_image(
            moments,
            current=jnp.zeros_like(operator.external_current),
            prescribed_current=jnp.asarray(current),
        ),
        dtype=np.float64,
    )
    coupling = bench._panel_display_coupling(
        prepared["circuit_context"]["group"],
        prepared["circuit_context"]["row"],
        lattice,
        np.asarray(current),
    )
    nodewise = (
        moments.cell_current,
        moments.radial_moment,
        moments.vertical_moment,
    )
    image = display_flux._flux(coupling, coordinates, nodewise, 4096)
    scale = max(float(np.max(np.abs(raster))), 1.0e-30)
    grid_error = float(np.max(np.abs(image[:grid_count] - raster[:grid_count])))
    reproduction = grid_error / scale
    difference = image - state_nodes

    masks = operator.current_domain_masks(state, requested_class)
    grid_labels = np.asarray(masks.label).reshape(-1)
    labels = np.concatenate(
        (
            grid_labels
            if grid_labels.size == grid_count
            else np.full(grid_count, -1, dtype=np.int64),
            np.full(node_count - grid_count, int(PlasmaDomain.EXCLUDED_MATERIAL)),
        )
    )

    _masks_read, topology = operator.read(state, requested_class)
    axis = np.asarray(topology.axis, dtype=np.float64).reshape(2)
    x_points = np.asarray(topology.x_point, dtype=np.float64).reshape(-1, 2)
    x_flux = np.asarray(topology.x_point_flux, dtype=np.float64).reshape(-1)
    boundary_flux = float(np.asarray(topology.boundary_flux))
    saddle_index = int(np.argmin(np.abs(x_flux - boundary_flux))) if x_flux.size else 0
    seed = x_points[saddle_index] if x_points.shape[0] else axis

    shape = tuple(int(value) for value in np.asarray(lattice.shape))
    radius_nodes = np.asarray(lattice.radius, dtype=np.float64)
    height_nodes = np.asarray(lattice.height, dtype=np.float64)
    state_field = state_nodes[:grid_count].reshape(shape).T
    image_field = image[:grid_count].reshape(shape).T

    state_saddle, state_saddle_flux, state_saddle_ok = _stationary_saddle(
        state_field, radius_nodes, height_nodes, seed
    )
    image_saddle, image_saddle_flux, image_saddle_ok = _stationary_saddle(
        image_field, radius_nodes, height_nodes, seed
    )

    state_branches = _branches(
        state_field, radius_nodes, height_nodes, boundary_flux, axis
    )
    image_level = image_saddle_flux if image_saddle_ok else boundary_flux
    image_branches = _branches(
        image_field, radius_nodes, height_nodes, image_level, axis
    )

    state_polyline = _cubic_points(
        state_branches["closed_controls_rz"], state_branches["closed_valid"]
    )
    image_polyline = _cubic_points(
        image_branches["closed_controls_rz"], image_branches["closed_valid"]
    )
    state_landmarks = _landmarks(state_branches, state_saddle, axis)
    image_landmarks = _landmarks(image_branches, image_saddle, axis)

    promoted_index = np.nonzero(mask)[0]
    reference = state_saddle if state_saddle_ok else seed
    to_x_point = np.linalg.norm(coordinates[promoted_index] - reference, axis=1)
    to_separatrix = _segmented_distance(state_polyline, coordinates[promoted_index])

    rows = []
    for position, index in enumerate(promoted_index):
        label = int(labels[index]) if 0 <= index < labels.size else -1
        rows.append(
            {
                "node": int(index),
                "radius_m": float(coordinates[index, 0]),
                "height_m": float(coordinates[index, 1]),
                "distance_to_x_point_mm": float(1.0e3 * to_x_point[position]),
                "distance_to_separatrix_mm": float(1.0e3 * to_separatrix[position]),
                "domain_label": DOMAIN_NAMES.get(label, "unlabelled"),
                "image_minus_state_wb": float(difference[index]),
                "state_wb": float(state_nodes[index]),
                "image_wb": float(image[index]),
            }
        )

    solved = ~mask
    solved_abs = np.abs(difference[solved])
    order = np.argsort(to_x_point)
    receipt = {
        "schema": "nova.promoted-nodes-near-xpoint",
        "edit_index": edit_index,
        "edit_fraction": float(bench.EDIT_FRACTIONS[edit_index]),
        "shot": int(bench.SHOT),
        "slice_index": int(bench.SLICE_INDEX),
        "coil_circuit_index": circuit_index,
        "target_current_a": float(target_current),
        "chain": chain,
        "terminal_residual_wb": float(np.asarray(terminal.terminal_residual)),
        "converged": bool(np.asarray(terminal.converged)),
        "image_reproduction_relative_error": reproduction,
        "promoted_node_count": int(np.count_nonzero(mask)),
        "lattice_node_count": grid_count,
        "physical_node_count": node_count,
        "wall_node_count": node_count - grid_count,
        "promoted_wall_node_count": int(np.count_nonzero(mask[grid_count:])),
        "lattice_shape": list(shape),
        "promoted": rows,
        "nearest_promoted_to_x_point_mm": float(1.0e3 * to_x_point[order[0]]),
        "promoted_distance_quantiles_mm": {
            f"q{int(level * 100)}": float(1.0e3 * np.quantile(to_x_point, level))
            for level in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "domain_counts": {
            name: int(sum(1 for row in rows if row["domain_label"] == name))
            for name in sorted({row["domain_label"] for row in rows})
        },
        "image_minus_state": {
            "promoted_max_abs_wb": float(np.max(np.abs(difference[mask])))
            if mask.any()
            else 0.0,
            "promoted_rms_wb": float(np.sqrt(np.mean(difference[mask] ** 2)))
            if mask.any()
            else 0.0,
            "solved_max_abs_wb": float(np.max(solved_abs)) if solved.any() else None,
            "solved_rms_wb": float(np.sqrt(np.mean(difference[solved] ** 2)))
            if solved.any()
            else None,
            "whole_lattice_max_abs_wb": float(np.max(np.abs(difference))),
        },
        "x_point": {
            "seed_rz": seed.tolist(),
            "state_rz": state_saddle.tolist(),
            "state_polished": state_saddle_ok,
            "state_flux_wb": state_saddle_flux,
            "image_rz": image_saddle.tolist(),
            "image_polished": image_saddle_ok,
            "image_flux_wb": image_saddle_flux,
            "displacement_mm": float(
                1.0e3 * np.linalg.norm(image_saddle - state_saddle)
            ),
            "displacement_r_mm": float(1.0e3 * (image_saddle[0] - state_saddle[0])),
            "displacement_z_mm": float(1.0e3 * (image_saddle[1] - state_saddle[1])),
        },
        "separatrix": {
            "state_boundary_flux_wb": boundary_flux,
            "image_level_wb": image_level,
            "state_well_formed": state_branches["well_formed"],
            "image_well_formed": image_branches["well_formed"],
            "lobe_direction_shift_deg": _angle_degrees(
                state_landmarks["lobe_direction"], image_landmarks["lobe_direction"]
            ),
            "leg_direction_shifts_deg": [
                _angle_degrees(left, right)
                for left, right in zip(
                    state_landmarks["leg_directions"],
                    image_landmarks["leg_directions"],
                    strict=False,
                )
            ],
            "state_lobe_tip_rz": state_landmarks["lobe_tip"],
            "image_lobe_tip_rz": image_landmarks["lobe_tip"],
            "lobe_tip_shift_mm": (
                float(
                    1.0e3
                    * np.linalg.norm(
                        np.asarray(image_landmarks["lobe_tip"], float)
                        - np.asarray(state_landmarks["lobe_tip"], float)
                    )
                )
                if state_landmarks["lobe_tip"] and image_landmarks["lobe_tip"]
                else float("nan")
            ),
            "state_closed_vertex_count": int(state_polyline.shape[0]),
            "image_closed_vertex_count": int(image_polyline.shape[0]),
        },
        "lattice_spacing_m": float(
            0.5
            * (
                float(radius_nodes[1] - radius_nodes[0])
                + float(height_nodes[1] - height_nodes[0])
            )
        ),
        "wall_units": [
            {
                "radius": np.asarray(unit.r, dtype=float).tolist(),
                "height": np.asarray(unit.z, dtype=float).tolist(),
                "kind": unit.kind,
                "closed": bool(unit.closed),
            }
            for unit in bench._wall_units(operator)
        ],
        "nulls": {
            "axis_rz": axis.tolist(),
            "x_points_rz": x_points.tolist(),
            "x_point_flux_wb": x_flux.tolist(),
            "saddle_index": saddle_index,
        },
        "device": {
            "host": jax.devices()[0].device_kind,
            "platform": jax.devices()[0].platform,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    fields = {
        "radius": radius_nodes,
        "height": height_nodes,
        "state_field": state_field,
        "image_field": image_field,
        "promoted_coordinate": coordinates[promoted_index],
        "state_saddle": state_saddle,
        "image_saddle": image_saddle,
        "axis": axis,
        "state_closed_polyline": state_polyline,
        "image_closed_polyline": image_polyline,
        "state_open_controls": state_branches["open_controls_rz"],
        "state_open_valid": state_branches["open_valid"],
        "image_open_controls": image_branches["open_controls_rz"],
        "image_open_valid": image_branches["open_valid"],
        "state_boundary_flux": np.asarray(boundary_flux),
        "image_level": np.asarray(image_level),
    }
    return receipt, fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit-index", type=int, default=12)
    parser.add_argument(
        "--carrier", type=Path, default=bench.response_carrier.DEFAULT_CARRIER
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    arguments = parser.parse_args()
    receipt, fields = measure(arguments.edit_index, arguments.carrier)
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(arguments.fields, **fields)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "promoted_node_count",
                    "terminal_residual_wb",
                    "nearest_promoted_to_x_point_mm",
                    "image_minus_state",
                    "x_point",
                    "separatrix",
                )
            },
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
