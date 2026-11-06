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
from nova.equilibrium.boundary_comparison import (  # noqa: E402
    _point_to_segment_distances,
)
from nova.equilibrium.domain import PlasmaDomain  # noqa: E402
from nova.equilibrium.topology import TopologyClass  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402
from nova.linalg.tensor_spline import fit_tensor_spline  # noqa: E402
from nova.media.poloidal import sample_cubic_controls  # noqa: E402

DOMAIN_NAMES = {
    int(PlasmaDomain.CORE): "core",
    int(PlasmaDomain.PRIVATE_FLUX): "private flux",
    int(PlasmaDomain.COMMON_SOL): "open (common SOL)",
    int(PlasmaDomain.EXCLUDED_MATERIAL): "wall shadow (outside material)",
}


def _json_default(value):
    """Render the receipt's array-valued coordinates as plain JSON numbers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


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


def _bearing_degrees(direction):
    """Return a 2-vector's bearing in the (radius, height) plane, in degrees."""
    if direction is None:
        return float("nan")
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def _wrap_degrees(angle):
    """Return an angle difference folded into (-180, 180]."""
    return float((angle + 180.0) % 360.0 - 180.0)


def _landmarks(branches, saddle, axis):
    """Return the lobe and leg directions one separatrix assembly defines.

    The lobe direction runs from the magnetic axis to the point of the closed
    branch furthest from it -- the lobe tip.  Each leg direction runs from the
    polished saddle to the far end of that open branch, which is where the leg
    leaves the traced field.  Both are reported as unit vectors and as a
    bearing measured from the +R axis in the (R, Z) plane.
    """
    landmark = {
        "lobe_direction": None,
        "lobe_bearing_deg": float("nan"),
        "leg_directions": [],
        "leg_bearings_deg": [],
        "leg_far_rz": [],
        "lobe_tip": None,
    }
    closed = sample_cubic_controls(
        branches["closed_controls_rz"], branches["closed_valid"]
    )
    if closed.shape[0] > 2 and np.all(np.isfinite(axis)):
        offsets = closed - np.asarray(axis, dtype=np.float64)
        radii = np.linalg.norm(offsets, axis=1)
        tip = int(np.argmax(radii))
        landmark["lobe_tip"] = closed[tip].tolist()
        landmark["lobe_direction"] = _direction(offsets[tip])[0]
        landmark["lobe_bearing_deg"] = _bearing_degrees(landmark["lobe_direction"])
    legs = []
    bearings = []
    far_points = []
    valid = branches["open_valid"]
    for slot in range(np.asarray(valid).shape[0]):
        points = sample_cubic_controls(
            np.asarray(branches["open_controls_rz"])[slot], valid[slot]
        )
        if points.shape[0] < 2 or not np.all(np.isfinite(saddle)):
            continue
        reach = np.linalg.norm(points - np.asarray(saddle, float), axis=1)
        far = int(np.argmax(reach))
        offset, norm = _direction(points[far] - np.asarray(saddle, float))
        if offset is None or norm < 1.0e-6:
            continue
        legs.append(offset)
        bearings.append(_bearing_degrees(offset))
        far_points.append(points[far].tolist())
    landmark["leg_directions"] = legs
    landmark["leg_bearings_deg"] = bearings
    landmark["leg_far_rz"] = far_points
    return landmark


def _pair_legs(state_legs, image_legs):
    """Pair each state leg with the image leg that ends nearest its own end."""
    pairs = []
    for index, direction in enumerate(state_legs["leg_directions"]):
        far = np.asarray(state_legs["leg_far_rz"][index], dtype=np.float64)
        if not image_legs["leg_directions"]:
            continue
        distance = [
            float(np.linalg.norm(far - np.asarray(point, dtype=np.float64)))
            for point in image_legs["leg_far_rz"]
        ]
        partner = int(np.argmin(distance))
        pairs.append(
            {
                "state_bearing_deg": state_legs["leg_bearings_deg"][index],
                "image_bearing_deg": image_legs["leg_bearings_deg"][partner],
                "shift_deg": _angle_degrees(
                    direction, image_legs["leg_directions"][partner]
                ),
                "far_end_separation_mm": 1.0e3 * min(distance),
            }
        )
    return pairs


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
    # The constrained row folds a compensating conductor current into the
    # prescribed vector before imaging, and that current carries flux of its
    # own.  The field of the currents is therefore drawn from the compensated
    # vector, and the same vector is handed to the raster the reproduction
    # check compares against.
    compensating = np.asarray(terminal.compensating_current, dtype=np.float64)
    compensated = np.asarray(current, dtype=np.float64) + compensating
    coupling = bench._panel_display_coupling(
        prepared["circuit_context"]["group"],
        prepared["circuit_context"]["row"],
        lattice,
        np.asarray(current),
    )
    complete_coupling = bench._panel_display_coupling(
        prepared["circuit_context"]["group"],
        prepared["circuit_context"]["row"],
        lattice,
        compensated,
    )
    nodewise = (
        moments.cell_current,
        moments.radial_moment,
        moments.vertical_moment,
    )
    image = display_flux._flux(complete_coupling, coordinates, nodewise, 4096)
    image_withheld = display_flux._flux(coupling, coordinates, nodewise, 4096)
    # The raster carries the plasma grid only, so the wall nodes the shadow
    # also spans are evaluated through the machine coupling the panel draws
    # with, and the grid half is checked against the raster the solve itself
    # uses: the two disagree only at the display's own reproduction error.
    raster = np.asarray(
        operator.raster_image(
            moments,
            current=jnp.zeros_like(operator.external_current),
            prescribed_current=jnp.asarray(compensated),
        ),
        dtype=np.float64,
    )
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
    spacing_m = 0.5 * (
        float(radius_nodes[1] - radius_nodes[0])
        + float(height_nodes[1] - height_nodes[0])
    )

    state_saddle, state_saddle_flux, state_saddle_ok = display_flux._stationary_saddle(
        state_field, radius_nodes, height_nodes, seed
    )

    # The complete field's saddle and separatrix are read by the module's own
    # builder on the fine display grid, with the solve's promoted mask and the
    # compensated current vector, so no part of the polish or the branch
    # assembly is restated here.
    complete = display_flux.build_display_flux_map(
        profile,
        state,
        coupling=complete_coupling,
        moments=moments,
        spacing=display_flux.DEFAULT_SPACING,
        current=jnp.zeros_like(operator.external_current),
        prescribed_current=jnp.asarray(compensated),
        requested_class=requested_class,
        axis_rz=axis,
        xpoint_rz=seed,
        promoted=mask[:grid_count],
        chunk=4096,
    )
    complete_state_surface = fit_tensor_spline(
        jnp.asarray(radius_nodes),
        jnp.asarray(height_nodes),
        jnp.asarray(state_field),
    )
    radius_grid, height_grid = np.meshgrid(
        np.asarray(complete.radius, dtype=np.float64),
        np.asarray(complete.height, dtype=np.float64),
        indexing="ij",
    )
    image_saddle = np.asarray(complete.saddle_rz, dtype=np.float64).reshape(2)
    image_saddle_flux = float(complete.saddle_flux)
    image_saddle_ok = bool(complete.saddle_converged)
    image_branches = complete.branches

    state_branches = display_flux._branches(
        state_field, radius_nodes, height_nodes, boundary_flux, axis
    )
    image_level = image_saddle_flux if image_saddle_ok else boundary_flux

    state_polyline = sample_cubic_controls(
        state_branches["closed_controls_rz"], state_branches["closed_valid"]
    )
    image_polyline = sample_cubic_controls(
        image_branches["closed_controls_rz"], image_branches["closed_valid"]
    )
    state_landmarks = _landmarks(state_branches, state_saddle, axis)
    image_landmarks = _landmarks(image_branches, image_saddle, axis)

    promoted_index = np.nonzero(mask)[0]
    reference = state_saddle if state_saddle_ok else seed
    to_x_point = np.linalg.norm(coordinates[promoted_index] - reference, axis=1)
    to_separatrix = _point_to_segment_distances(
        coordinates[promoted_index], state_polyline
    )

    # The residual-shadow rule composes two independent components.  Reading
    # them apart is what tells a promoted node's own source; the concatenated
    # mask alone labels every appended wall coordinate the same way whether
    # the flood or the private-wall band selected it.
    flood_mask, wall_mask = operator.residual_shadow_components(state, requested_class)
    flood_mask = np.asarray(flood_mask, dtype=bool).reshape(-1)
    wall_mask = np.asarray(wall_mask, dtype=bool).reshape(-1)

    def _source(index: int) -> str:
        if index < grid_count:
            return "interior flood (private flux)" if flood_mask[index] else "unflooded"
        wall_index = index - grid_count
        if wall_index < wall_mask.size and wall_mask[wall_index]:
            return "private-wall band"
        return "wall coordinate, flood-promoted"

    rows = []
    for position, index in enumerate(promoted_index):
        label = int(labels[index]) if 0 <= index < labels.size else -1
        rows.append(
            {
                "node": int(index),
                "radius_m": float(coordinates[index, 0]),
                "height_m": float(coordinates[index, 1]),
                "promoted_source": _source(int(index)),
                "distance_to_x_point_mm": float(1.0e3 * to_x_point[position]),
                "distance_to_separatrix_mm": float(1.0e3 * to_separatrix[position]),
                "domain_label": DOMAIN_NAMES.get(label, "unlabelled"),
                "image_minus_state_wb": float(difference[index]),
                "operator_image_minus_state_wb": (
                    float(raster[index] - state_nodes[index])
                    if index < grid_count
                    else float("nan")
                ),
                "state_wb": float(state_nodes[index]),
                "image_wb": float(image[index]),
            }
        )

    halves = {}
    heights = coordinates[promoted_index, 1]
    for name, selector in (
        ("upper", heights > 0.0),
        ("lower", heights <= 0.0),
    ):
        values = difference[promoted_index][selector]
        halves[name] = {
            "count": int(np.count_nonzero(selector)),
            "max_abs_wb": float(np.max(np.abs(values))) if values.size else None,
            "rms_wb": float(np.sqrt(np.mean(values**2))) if values.size else None,
            "flux_min_wb": float(np.min(state_nodes[promoted_index][selector]))
            if values.size
            else None,
            "flux_max_wb": float(np.max(state_nodes[promoted_index][selector]))
            if values.size
            else None,
        }
    # Every lattice node outside the promoted set, per half: the control that
    # says what an unexcluded residual looks like at this convergence.
    solved_grid = ~mask[:grid_count]
    grid_heights = coordinates[:grid_count, 1]
    for name, selector in (
        ("upper", grid_heights > 0.0),
        ("lower", grid_heights <= 0.0),
    ):
        control = np.abs(image[:grid_count] - state_nodes[:grid_count])[
            solved_grid & selector
        ]
        halves[name]["solved_grid_max_abs_wb"] = (
            float(np.max(control)) if control.size else None
        )
        halves[name]["solved_grid_count"] = int(
            np.count_nonzero(solved_grid & selector)
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
        "constraint_compensation": {
            "compensating_current_a": float(
                np.asarray(terminal.constraints[0].physical_unknown).reshape(-1)[0]
            )
            if getattr(terminal, "constraints", ())
            else None,
            "compensating_current_norm_a": float(np.linalg.norm(compensating)),
            "compensating_current_wb_per_a_applied": True,
            "max_abs_flux_on_physical_nodes_wb": float(
                np.max(np.abs(image - image_withheld))
            ),
        },
        "complete_field": {
            "builder": "nova.equilibrium.display_flux.build_display_flux_map",
            "promoted_node_count": int(complete.promoted_node_count),
            "spacing_m": float(complete.spacing_m),
            "reproduction_relative_error": float(complete.reproduction_relative_error),
            "max_node_mismatch_wb": float(complete.max_node_mismatch_wb),
            "max_solved_node_mismatch_wb": float(complete.max_solved_node_mismatch_wb),
            "saddle_offset_mm": float(complete.saddle_offset_mm),
            "nearest_promoted_to_x_point_mm": float(
                1.0e3 * complete.nearest_promoted_to_xpoint_m
            ),
            "well_formed": bool(complete.well_formed),
            "build_seconds": float(complete.build_seconds),
        },
        "promoted_node_count": int(np.count_nonzero(mask)),
        "lattice_node_count": grid_count,
        "physical_node_count": node_count,
        "wall_node_count": node_count - grid_count,
        "promoted_wall_node_count": int(np.count_nonzero(mask[grid_count:])),
        "lattice_shape": list(shape),
        "promoted": rows,
        "promoted_halves": halves,
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
            "displacement_lattice_cells": float(
                np.linalg.norm(image_saddle - state_saddle) / spacing_m
            ),
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
            "state_lobe_bearing_deg": state_landmarks["lobe_bearing_deg"],
            "image_lobe_bearing_deg": image_landmarks["lobe_bearing_deg"],
            "lobe_bearing_change_deg": _wrap_degrees(
                image_landmarks["lobe_bearing_deg"]
                - state_landmarks["lobe_bearing_deg"]
            ),
            "state_leg_count": len(state_landmarks["leg_directions"]),
            "image_leg_count": len(image_landmarks["leg_directions"]),
            "state_leg_bearings_deg": state_landmarks["leg_bearings_deg"],
            "image_leg_bearings_deg": image_landmarks["leg_bearings_deg"],
            "legs": _pair_legs(state_landmarks, image_landmarks),
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
        "lattice_spacing_m": spacing_m,
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
        "display_radius": np.asarray(complete.radius, dtype=np.float64),
        "display_height": np.asarray(complete.height, dtype=np.float64),
        "display_flux": np.asarray(complete.flux, dtype=np.float64),
        # The converged state on the same display grid, read from the same
        # tensor spline its own saddle is polished on, so both fields' contour
        # panels share one level array and one comparison grid.
        "display_state_flux": np.asarray(
            complete_state_surface(np.c_[radius_grid.ravel(), height_grid.ravel()]),
            dtype=np.float64,
        ).reshape(complete.flux.shape),
        "promoted_source": np.asarray(
            [row["promoted_source"] for row in rows], dtype="U40"
        )
        if rows
        else np.asarray([], dtype="U40"),
    }
    return receipt, fields


def _print_headlines(receipt: dict) -> None:
    """Write the headline numbers to the log before the receipt is written.

    The receipt is a serialisation step after the measurement, and a failure
    there once lost a complete solve. These lines are emitted first so the
    numbers survive a writer defect and can be read straight out of the job log.
    """
    difference = receipt["image_minus_state"]
    x_point = receipt["x_point"]
    separatrix = receipt["separatrix"]
    lines = [
        ("solved_max_abs_wb", difference["solved_max_abs_wb"]),
        ("solved_rms_wb", difference["solved_rms_wb"]),
        ("promoted_max_abs_wb", difference["promoted_max_abs_wb"]),
        ("promoted_rms_wb", difference["promoted_rms_wb"]),
        ("whole_lattice_max_abs_wb", difference["whole_lattice_max_abs_wb"]),
        ("x_point_displacement_mm", x_point["displacement_mm"]),
        ("x_point_displacement_lattice_cells", x_point["displacement_lattice_cells"]),
        ("x_point_displacement_r_mm", x_point["displacement_r_mm"]),
        ("x_point_displacement_z_mm", x_point["displacement_z_mm"]),
        ("lobe_direction_shift_deg", separatrix["lobe_direction_shift_deg"]),
        ("lobe_tip_shift_mm", separatrix["lobe_tip_shift_mm"]),
    ]
    lines.extend(
        (f"leg_direction_shift_{index}_deg", value)
        for index, value in enumerate(separatrix["leg_direction_shifts_deg"])
    )
    lines.extend(
        [
            ("lobe_bearing_state_deg", separatrix["state_lobe_bearing_deg"]),
            ("lobe_bearing_image_deg", separatrix["image_lobe_bearing_deg"]),
            ("lobe_bearing_change_deg", separatrix["lobe_bearing_change_deg"]),
            ("state_leg_count", separatrix["state_leg_count"]),
            ("image_leg_count", separatrix["image_leg_count"]),
        ]
    )
    for index, pair in enumerate(separatrix["legs"]):
        lines.append((f"leg_{index}_shift_deg", pair["shift_deg"]))
    for half, block in receipt["promoted_halves"].items():
        lines.append((f"promoted_{half}_max_abs_wb", block["max_abs_wb"]))
    for name, value in lines:
        print(f"HEADLINE {name} {value}", flush=True)


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
    _print_headlines(receipt)
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(
        json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=True, default=_json_default
        )
        + "\n",
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
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
