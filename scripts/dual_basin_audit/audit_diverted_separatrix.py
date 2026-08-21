"""Audit the digest-pinned diverted oracle's separatrix geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import brentq

from nova.equilibrium.topology import boundary_mode
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures.measure import (
    FIXTURE_REQUESTS,
    WALL_POINT_COUNT,
    analytic_case,
    cached_machine,
    forward_operator,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPOSITORY_ROOT / "scripts" / "dual_basin_fixtures"
BANK_PATH = FIXTURE_ROOT / "diverted-state.npz"
SOURCE_RECEIPT_PATH = FIXTURE_ROOT / "diverted-receipt.json"
OUTPUT_ROOT = REPOSITORY_ROOT / "scripts" / "dual_basin_audit"
FIGURE_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "figures"
    / "dual-basin-solve"
    / "diverted-separatrix-audit.png"
)
RECEIPT_PATH = OUTPUT_ROOT / "diverted-separatrix-audit.json"
LOCAL_RADIUS_M = 0.02
CLEARANCE_PITCH_FLOOR = 2.0
FLUX_MARGIN_FRACTION_FLOOR = 0.05
LOCATOR_ERROR_PITCH_CEILING = 0.5


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _identity(values: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "sha256": _digest(values),
    }


def _load_bank() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load the bank only after every stored array matches its pinned identity."""
    source_receipt = json.loads(SOURCE_RECEIPT_PATH.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    with np.load(BANK_PATH, allow_pickle=False) as stored:
        if set(stored.files) != set(source_receipt["arrays"]):
            raise ValueError("bank array names do not match the source receipt")
        for name in stored.files:
            values = np.asarray(stored[name])
            if _identity(values) != source_receipt["arrays"][name]:
                raise ValueError(f"banked {name} does not match its SHA-256 identity")
            arrays[name] = values
    return arrays, source_receipt


def _flux(coordinates: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    radius = np.asarray(coordinates, dtype=np.float64)[..., 0]
    height = np.asarray(coordinates, dtype=np.float64)[..., 1]
    alpha, beta, gauge, radius2_term, height_term, mixed_term, quartic = coefficients
    radius2 = radius**2
    height2 = height**2
    return np.asarray(
        alpha * radius2**2
        + beta * height2
        + gauge
        + radius2_term * radius2
        + height_term * height
        + mixed_term * radius2 * height
        + quartic * (radius2**2 - 4.0 * radius2 * height2),
        dtype=np.float64,
    )


def _hessian(point: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    radius, height = point
    alpha, beta, _gauge, radius2_term, _height_term, mixed_term, quartic = coefficients
    cross = 2.0 * mixed_term * radius - 16.0 * quartic * radius * height
    return np.array(
        [
            [
                12.0 * alpha * radius**2
                + 2.0 * radius2_term
                + 2.0 * mixed_term * height
                + quartic * (12.0 * radius**2 - 8.0 * height**2),
                cross,
            ],
            [cross, 2.0 * beta - 8.0 * quartic * radius**2],
        ],
        dtype=np.float64,
    )


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    segment = end - start
    projection = np.dot(point - start, segment) / np.dot(segment, segment)
    closest = start + np.clip(projection, 0.0, 1.0) * segment
    return float(np.linalg.norm(point - closest))


def _wall_clearance(point: np.ndarray, wall: np.ndarray) -> float:
    closed = np.vstack((wall, wall[0]))
    return min(
        _point_segment_distance(point, start, end)
        for start, end in zip(closed[:-1], closed[1:], strict=True)
    )


def _wall_intersections(
    wall: np.ndarray, coefficients: np.ndarray, level: float
) -> list[list[float]]:
    closed = np.vstack((wall, wall[0]))
    intersections: list[np.ndarray] = []
    for start, end in zip(closed[:-1], closed[1:], strict=True):
        start_value, end_value = _flux(np.vstack((start, end)), coefficients) - level
        if start_value * end_value > 0.0:
            continue

        def segment_flux(fraction: float) -> float:
            coordinate = start + fraction * (end - start)
            return float(_flux(coordinate[None, :], coefficients)[0] - level)

        if start_value == 0.0:
            point = start
        elif end_value == 0.0:
            point = end
        else:
            fraction = brentq(segment_flux, 0.0, 1.0, xtol=1.0e-14)
            point = start + fraction * (end - start)
        if (
            not intersections
            or min(np.linalg.norm(point - existing) for existing in intersections)
            > 1.0e-8
        ):
            intersections.append(point)
    return [point.tolist() for point in intersections]


def _circle_leg_angles(
    saddle: np.ndarray,
    coefficients: np.ndarray,
    level: float,
    radius: float,
) -> np.ndarray:
    samples = np.linspace(0.0, 2.0 * np.pi, 4097)

    def circle_flux(angle: float) -> float:
        offset = radius * np.array([np.cos(angle), np.sin(angle)])
        return float(_flux((saddle + offset)[None, :], coefficients)[0] - level)

    values = np.asarray([circle_flux(angle) for angle in samples])
    roots: list[float] = []
    for left, right, left_value, right_value in zip(
        samples[:-1], samples[1:], values[:-1], values[1:], strict=True
    ):
        if left_value * right_value < 0.0:
            roots.append(brentq(circle_flux, left, right, xtol=1.0e-14))
    angles = np.mod(np.asarray(roots), 2.0 * np.pi)
    angles.sort()
    if len(angles) != 4:
        raise AssertionError(
            f"expected four local separatrix legs, found {len(angles)}"
        )
    return angles


def _carrier_pitch(machine) -> float:
    stencil = np.asarray(machine.interior_stencil)
    node = np.asarray(machine.node)
    centres = node[stencil[:, :1]]
    ring = node[stencil[:, 1:]]
    return float(np.median(np.linalg.norm(ring - centres, axis=2)))


def _production_read(operator, state: np.ndarray) -> dict[str, object]:
    grid_flux, _wall_flux = operator.topology.split_flux_map(jnp.asarray(state))
    extrema, saddles = operator.topology.grid(grid_flux)
    _masks, topology = operator.read(jnp.asarray(state))
    finite_extrema = np.asarray(extrema)[np.isfinite(np.asarray(extrema)[:, 0])]
    finite_saddles = np.asarray(saddles)[np.isfinite(np.asarray(saddles)[:, 0])]
    return {
        "class": boundary_mode(topology).value,
        "axis_m": np.asarray(topology.axis, dtype=np.float64).tolist(),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_m": np.asarray(topology.boundary, dtype=np.float64).tolist(),
        "boundary_flux_wb": float(topology.boundary_flux),
        "wall_extremum_m": np.asarray(topology.wall_point, dtype=np.float64).tolist(),
        "wall_extremum_flux_wb": float(topology.wall_point_flux),
        "selected_saddle_m": np.asarray(topology.x_point, dtype=np.float64).tolist(),
        "selected_saddle_flux_wb": float(topology.x_point_flux),
        "finite_axis_count": len(finite_extrema),
        "finite_saddle_count": len(finite_saddles),
        "boundary_saddle_distance_m": float(
            np.linalg.norm(np.asarray(topology.boundary) - topology.x_point)
        ),
        "boundary_saddle_flux_difference_wb": float(
            topology.boundary_flux - topology.x_point_flux
        ),
    }


def _draw(
    wall: np.ndarray,
    coefficients: np.ndarray,
    axis: np.ndarray,
    saddle: np.ndarray,
    located_saddle: np.ndarray,
    saddle_flux: float,
    leg_angles: np.ndarray,
) -> None:
    margin = 0.055
    radius = np.linspace(wall[:, 0].min() - margin, wall[:, 0].max() + margin, 900)
    height = np.linspace(wall[:, 1].min() - margin, wall[:, 1].max() + margin, 900)
    radius_grid, height_grid = np.meshgrid(radius, height)
    coordinates = np.stack((radius_grid, height_grid), axis=-1)
    field = _flux(coordinates, coefficients)
    axis_flux = float(_flux(axis[None, :], coefficients)[0])
    interior_levels = saddle_flux + (axis_flux - saddle_flux) * np.array(
        [0.1, 0.3, 0.55, 0.8]
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    for plot_axis in axes:
        plot_axis.contour(
            radius_grid,
            height_grid,
            field,
            levels=interior_levels,
            colors="0.68",
            linewidths=0.8,
        )
        plot_axis.contour(
            radius_grid,
            height_grid,
            field,
            levels=[saddle_flux],
            colors="C3",
            linewidths=2.0,
        )
        plot_axis.plot(
            np.r_[wall[:, 0], wall[0, 0]],
            np.r_[wall[:, 1], wall[0, 1]],
            color="0.12",
            linewidth=1.4,
        )
        plot_axis.scatter(*axis, color="C0", s=38, zorder=5)
        plot_axis.scatter(*saddle, color="C3", marker="x", s=64, linewidth=2, zorder=6)
        plot_axis.scatter(
            *located_saddle,
            facecolor="none",
            edgecolor="C1",
            marker="o",
            s=66,
            linewidth=1.5,
            zorder=5,
        )
        plot_axis.set_aspect("equal", adjustable="box")
        plot_axis.spines[["top", "right"]].set_visible(False)
        plot_axis.set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    axes[0].set_title("Full wall and separatrix")
    axes[0].set_xlim(radius[0], radius[-1])
    axes[0].set_ylim(height[0], height[-1])
    axes[0].annotate("axis", axis, xytext=(8, 8), textcoords="offset points")
    axes[0].annotate(
        "analytic saddle", saddle, xytext=(8, -16), textcoords="offset points"
    )

    zoom = 0.115
    axes[1].set_title("Local saddle geometry (equal scale)")
    axes[1].set_xlim(saddle[0] - zoom, saddle[0] + zoom)
    axes[1].set_ylim(saddle[1] - zoom, saddle[1] + zoom)
    circle = plt.Circle(saddle, LOCAL_RADIUS_M, fill=False, color="0.45", linewidth=0.8)
    axes[1].add_patch(circle)
    leg_points = saddle + LOCAL_RADIUS_M * np.column_stack(
        (np.cos(leg_angles), np.sin(leg_angles))
    )
    axes[1].scatter(leg_points[:, 0], leg_points[:, 1], s=18, color="C3", zorder=7)
    axes[1].annotate(
        "analytic saddle", saddle, xytext=(8, -18), textcoords="offset points"
    )
    axes[1].annotate(
        "production locator",
        located_saddle,
        xytext=(8, 9),
        textcoords="offset points",
    )
    figure.text(
        0.5,
        0.01,
        "red: exact saddle-flux contour   blue: axis   red x: analytic saddle   "
        "orange ring: production locator",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    figure.savefig(FIGURE_PATH, dpi=190)
    plt.close(figure)


def audit() -> dict[str, object]:
    configure_dtypes()
    arrays, source_receipt = _load_bank()
    state = arrays["state"]
    coefficients = arrays["coefficients"]
    axis, saddle = arrays["stationary_points"]

    case = analytic_case()
    machine = cached_machine(
        case, FIXTURE_REQUESTS["fine"], wall_nodes=WALL_POINT_COUNT
    )
    if not machine.cache["hit"]:
        raise RuntimeError("the audit requires the warm fine semantic carrier")
    if machine.cache["semantic_key"] != source_receipt["carrier"]["cache_semantic_key"]:
        raise ValueError("carrier semantic key does not match the fixture receipt")
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    roundtrip_error = float(np.max(np.abs(state - _flux(coordinates, coefficients))))
    if roundtrip_error != 0.0:
        raise AssertionError(
            "banked state is not the exact coefficient-array evaluation"
        )

    operator = forward_operator(case, machine)
    production = _production_read(operator, state)
    saddle_hessian = _hessian(saddle, coefficients)
    eigenvalues, eigenvectors = np.linalg.eigh(saddle_hessian)
    determinant = float(np.linalg.det(saddle_hessian))
    saddle_flux = float(_flux(saddle[None, :], coefficients)[0])
    axis_flux = float(_flux(axis[None, :], coefficients)[0])
    flux_span = abs(axis_flux - saddle_flux)
    boundary_wall_margin = abs(
        production["boundary_flux_wb"] - production["wall_extremum_flux_wb"]
    )
    margin_fraction = boundary_wall_margin / flux_span
    grid_count = source_receipt["carrier"]["state_layout"][0][1]
    wall_count = source_receipt["carrier"]["state_layout"][1][1]
    sampled_wall_flux = state[grid_count : grid_count + wall_count]
    sampled_wall_peak = float(np.max(sampled_wall_flux))
    exact_saddle_wall_margin = abs(saddle_flux - sampled_wall_peak)
    wall_fit_overshoot = production["wall_extremum_flux_wb"] - sampled_wall_peak
    wall_clearance = _wall_clearance(saddle, machine.wall_node)
    carrier_pitch = _carrier_pitch(machine)
    clearance_pitches = wall_clearance / carrier_pitch
    located_saddle = np.asarray(production["selected_saddle_m"])
    locator_error = float(np.linalg.norm(located_saddle - saddle))
    locator_error_pitches = locator_error / carrier_pitch

    leg_angles = _circle_leg_angles(saddle, coefficients, saddle_flux, LOCAL_RADIUS_M)
    gaps = np.diff(np.r_[leg_angles, leg_angles[0] + 2.0 * np.pi])
    leg_points = saddle + LOCAL_RADIUS_M * np.column_stack(
        (np.cos(leg_angles), np.sin(leg_angles))
    )
    wall_intersections = _wall_intersections(
        machine.wall_node, coefficients, saddle_flux
    )

    genuine = (
        eigenvalues[0] < 0.0 < eigenvalues[1]
        and len(leg_angles) == 4
        and production["class"] == "diverted"
        and production["finite_saddle_count"] == 1
        and production["boundary_saddle_distance_m"] == 0.0
        and locator_error_pitches < LOCATOR_ERROR_PITCH_CEILING
    )
    well_separated = (
        genuine
        and clearance_pitches >= CLEARANCE_PITCH_FLOOR
        and margin_fraction >= FLUX_MARGIN_FRACTION_FLOOR
    )
    if not genuine:
        verdict = "locator-artifact"
    elif well_separated:
        verdict = "genuine-well-separated"
    else:
        verdict = "genuine-marginal"

    _draw(
        machine.wall_node,
        coefficients,
        axis,
        saddle,
        located_saddle,
        saddle_flux,
        leg_angles,
    )
    return {
        "schema": "nova.diverted-separatrix-audit",
        "schema_version": 1,
        "verdict": verdict,
        "input_policy": {
            "bank": str(BANK_PATH.relative_to(REPOSITORY_ROOT)),
            "source_receipt": str(SOURCE_RECEIPT_PATH.relative_to(REPOSITORY_ROOT)),
            "digest_validated_arrays": {
                name: _identity(values) for name, values in arrays.items()
            },
            "stored_topology_label_used_as_input": False,
            "state_relabelled": False,
            "flux_values_from_digest_pinned_bank_only": True,
            "carrier_geometry_semantic_key": machine.cache["semantic_key"],
            "bank_state_roundtrip_max_abs_wb": roundtrip_error,
        },
        "production_read_recomputed_from_state": production,
        "numeric_discriminators": {
            "analytic_saddle_m": saddle.tolist(),
            "production_saddle_m": located_saddle.tolist(),
            "production_locator_error_m": locator_error,
            "production_locator_error_carrier_pitches": locator_error_pitches,
            "saddle_to_wall_segment_clearance_m": wall_clearance,
            "median_carrier_pitch_m": carrier_pitch,
            "saddle_to_wall_clearance_carrier_pitches": clearance_pitches,
            "boundary_minus_wall_flux_margin_abs_wb": boundary_wall_margin,
            "fixture_flux_span_wb": flux_span,
            "boundary_minus_wall_flux_margin_fraction_of_span": margin_fraction,
            "sampled_wall_peak_flux_wb": sampled_wall_peak,
            "exact_saddle_minus_sampled_wall_margin_abs_wb": (
                exact_saddle_wall_margin
            ),
            "exact_saddle_minus_sampled_wall_margin_fraction_of_span": (
                exact_saddle_wall_margin / flux_span
            ),
            "production_quadratic_wall_fit_overshoot_wb": wall_fit_overshoot,
            "production_quadratic_wall_fit_overshoot_fraction_of_span": (
                wall_fit_overshoot / flux_span
            ),
        },
        "local_hessian": {
            "matrix_wb_per_m2": saddle_hessian.tolist(),
            "eigenvalues_wb_per_m2": eigenvalues.tolist(),
            "orthonormal_eigenvectors_columns": eigenvectors.tolist(),
            "determinant_wb2_per_m4": determinant,
            "signature": "negative-positive",
        },
        "separatrix_leg_geometry": {
            "probe_radius_m": LOCAL_RADIUS_M,
            "leg_count": len(leg_angles),
            "leg_angles_degrees": np.rad2deg(leg_angles).tolist(),
            "leg_points_m": leg_points.tolist(),
            "successive_opening_angles_degrees": np.rad2deg(gaps).tolist(),
            "minimum_opening_angle_degrees": float(np.rad2deg(gaps).min()),
            "wall_intersection_count": len(wall_intersections),
            "wall_intersections_m": wall_intersections,
        },
        "adjudication_rule": {
            "artifact_if": (
                "the Hessian is not indefinite, the local level set does not have "
                "four legs, the production boundary is not its sole saddle, or "
                "locator error reaches half a carrier pitch"
            ),
            "well_separated_if": (
                "genuine and both wall clearance is at least two median carrier "
                "pitches and boundary-wall flux margin is at least 5% of span"
            ),
            "marginal_if": "genuine but either separation floor is missed",
            "thresholds": {
                "locator_error_pitch_ceiling": LOCATOR_ERROR_PITCH_CEILING,
                "wall_clearance_pitch_floor": CLEARANCE_PITCH_FLOOR,
                "flux_margin_fraction_floor": FLUX_MARGIN_FRACTION_FLOOR,
            },
        },
        "artifacts": {
            "figure": str(FIGURE_PATH.relative_to(REPOSITORY_ROOT)),
            "receipt": str(RECEIPT_PATH.relative_to(REPOSITORY_ROOT)),
        },
    }


def main() -> None:
    receipt = audit()
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics = receipt["numeric_discriminators"]
    print(
        f"VERDICT {receipt['verdict']} "
        f"clearance_pitches={metrics['saddle_to_wall_clearance_carrier_pitches']:.6g} "
        "flux_margin_fraction="
        f"{metrics['boundary_minus_wall_flux_margin_fraction_of_span']:.6g} "
        f"receipt={RECEIPT_PATH} figure={FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()
