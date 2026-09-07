"""Attribute the magnetic-axis position of one MAST slice to a solve factor.

The archive shows two nova solves of the same slice (shot 21978, efm row 36,
t = 0.195 s) disagreeing by more than either disagrees with EFIT: the labeller
configuration resolves the axis at R 0.86478, Z -0.01485 while the wide-grid
parity route resolves R 0.90611, Z +0.01869 against an EFIT axis at
R 0.90593, Z +0.00048.  This driver re-solves that slice on a CPU compute node,
reproducing both banked endpoints and varying exactly one factor per arm:

* grid extent and resolution  (the frozen carrier mesh versus a finer mesh)
* seed policy                 (previous-terminal continuation versus the
                               explicit reconstruction seed of the slice)
* solve route with its residual tolerance (the writer's 1e-8 regime versus
                               the parity route's 1e-12 finish)

Each arm records the resolved magnetic axis major radius and height, the final
residual, the topology class and the minimum and maximum curvature of the
selected axis candidate from the stationary-point census.  No solver module is
edited; this is a measurement over the production entry points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.interpolate import RectBivariateSpline

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    FIXED_POINT_CRITERION,
    TOTAL_FLUX_FACTOR,
    _mast_case_from_selection,
    _passive_inclusive_case,
)
from benchmarks.forward_labeller_throughput import (
    NEWTON_STEPS,
    _requested_class,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium import fixed_point, reduced_newton
from nova.equilibrium.stencil_nulls import (
    _fit_selected_centres,
    critical_point_candidates_batch,
)
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

ROOT = Path(__file__).resolve().parents[1]
SHOT = 21978
TARGET_TIME_S = 0.195
#: EFIT's published magnetic axis for this slice (efm/magnetic_axis_r/z row 36).
#: The banked nova results the arms are quoted against:
BANKED = {
    "efit": (0.9059291481971741, 0.0004827221855521202),
    "labeller_configuration": (0.864784, -0.014849),
    "wide_grid_parity_route": (0.9061140733600253, 0.01869058210819947),
}
OUTPUT_DIRECTORY = ROOT / "docs/figures/playable-forward-solve/axis-configuration"
OUTPUT_JSON = OUTPUT_DIRECTORY / "axis-configuration-receipt.json"
OUTPUT_FIGURE = OUTPUT_DIRECTORY / "axis-configuration.png"

#: The frozen-six carrier operator is calibrated on the 22086/43 keyframe.
CARRIER_SHOT = 22086
CARRIER_SLICE = 43
#: Tighter residual finish of the parity route (it reached 1.6e-12).
TIGHT_TOLERANCE = 1.0e-12
#: Finer solve mesh used to probe the grid-resolution factor (same extent).
FINE_GRID_POINTS = 65


def _finite(value: Any) -> float | None:
    """Return a strict-JSON scalar or null for a non-finite value."""
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def _point(value: Any) -> list[float] | None:
    """Return one JSON-safe R-Z point, or ``None``."""
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return point.tolist()


def _sha256(path: Path) -> str:
    """Return the stable byte identity of one evidence input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_for(
    shot: int,
    row: int,
    response_cache: dict[str, Any] | None,
    *,
    mesh_points: int | None = None,
) -> tuple[Any, Any]:
    """Build one passive-inclusive forward profile and its case context."""
    selected = {"shot": shot, "slice_index": row}
    qualification = {"note": "axis-configuration attribution driver"}
    case, context = _mast_case_from_selection(
        SHOT_STORE, selected, qualification, grid_points=mesh_points
    )
    _case, profile, _policy = _passive_inclusive_case(case, context, response_cache)
    return profile, case


def _slices_seed_on_mesh(
    group: zarr.Group, row: int, radius: np.ndarray, height: np.ndarray
) -> np.ndarray:
    """Return a reconstruction-flux seed on an arbitrary rectangular mesh."""
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    reference_full = TOTAL_FLUX_FACTOR * _live_flux_map(group, row, len(full_r)).T
    radius_grid, height_grid = np.meshgrid(radius, height, indexing="ij")
    spline = RectBivariateSpline(full_r, full_z, reference_full, kx=3, ky=3, s=0.0)
    reference = spline(radius, height)
    limiter = np.column_stack(
        [
            np.asarray(group["limiterr"], dtype=float),
            np.asarray(group["limiterz"], dtype=float),
        ]
    )
    return np.r_[reference.ravel(), spline.ev(limiter[:, 0], limiter[:, 1])]


def _live_flux_map(group, row: int, size: int) -> np.ndarray:
    """Return the stored total poloidal flux map of one slice on its grid."""
    from benchmarks.efit_topology_boundary_score import _live_flux_map as _map

    return _map(group, row, size)


def _row_seed_on_mesh(profile: Any, group: zarr.Group, row: int) -> np.ndarray:
    """Return the mesh-aligned reconstruction seed of one slice.

    The carrier profile's lattice lives on the GRID_STRIDE-subsampled stored
    65-point axes; ``forward_labeller_throughput._slices_seed`` is the
    canonical seed for that mesh and is reused verbatim.  For an arbitrary
    mesh (the grid-resolution arm) the same reconstruction is interpolated
    onto that mesh.
    """
    radius = np.asarray(profile.lattice.radius, dtype=np.float64)
    height = np.asarray(profile.lattice.height, dtype=np.float64)
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    canonical_mesh = len(radius) == len(full_r[::2]) and np.allclose(
        radius, full_r[::2]
    )
    if canonical_mesh:
        from benchmarks.forward_labeller_throughput import _slices_seed

        return _slices_seed(group, row, full_r, np.asarray(group["gridz"], dtype=float))
    return _slices_seed_on_mesh(group, row, radius, height)


def equilibrium_from_reduced(
    profile: Any, result: Any, requested_class, inputs, current
):
    """Lift one reduced result through the same receipt path as the writer."""
    residuals = jnp.asarray(result.active_set_residuals, dtype=jnp.float64)
    differences = jnp.asarray(result.active_set_mask_differences, dtype=jnp.int32)
    history = fixed_point.FixedPointResult(
        state=result.state,
        residual=jnp.asarray(result.terminal_residual, dtype=jnp.float64),
        trace=residuals,
        converged=jnp.asarray(result.converged),
        termination_reason=jnp.asarray(result.termination_reason, dtype=jnp.int32),
        active_set_iterations=jnp.asarray(
            result.active_set_iterations, dtype=jnp.int32
        ),
        active_set_residuals=residuals,
        active_set_mask_differences=differences,
        shadow_mask_changes=differences,
    )
    return profile._receipt(
        result.state,
        history,
        requested_class,
        abs(inputs["reference_plasma_current"]),
        None,
        jnp.asarray(current),
        constraints=tuple(getattr(result, "constraints", ())),
    )


def _scalar(value: Any) -> float:
    """Return one Python scalar from any 0-d or 1-d array-shaped value."""
    return float(np.asarray(value).reshape(-1)[0])


def _axis_census_curvature(profile: Any, state: jax.Array) -> dict[str, Any]:
    """Resolve the axis candidate on the solved map and its curvatures.

    Runs the production stationary-point census on the solved grid flux and
    fits the exact-offset quadratic at the selected candidate's native cell,
    which is the census's own curvature readout (the absolute Hessian
    eigenvalues in cell-pitch-normalised coordinates).  A small minimum
    curvature relative to the maximum is the flat-extremum signature that
    makes the subgrid fit ill-conditioned.
    """
    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
    radius_device, height_device, shape = operator.connectivity_grid_axes()
    radial_count, vertical_count = shape
    radius = np.asarray(radius_device, dtype=np.float64)
    height = np.asarray(height_device, dtype=np.float64)
    flux = np.asarray(grid_flux.reshape((radial_count, vertical_count)).T)
    material = np.asarray(operator.inside_material, dtype=bool)
    material_2d = material.reshape((radial_count, vertical_count)).T
    fields = np.asarray(flux, dtype=jnp.float64).reshape(
        1, vertical_count, radial_count
    )
    fields_tensor = jnp.asarray(fields)
    census = critical_point_candidates_batch(
        fields_tensor,
        jnp.asarray(radius, dtype=jnp.float64),
        jnp.asarray(height, dtype=jnp.float64),
        jnp.asarray(material_2d, dtype=bool),
        k_slots=8,
        target_index=1,
        material_dilate=0,
    )
    present = np.asarray(census["present"][0], dtype=bool)
    ntype = np.asarray(census["ntype"][0], dtype=float)
    oriented = present & (ntype == 1)
    if not np.any(oriented):
        return {
            "found": False,
            "axis_rz_m": None,
            "confidence": None,
            "minimum_curvature": None,
            "maximum_curvature": None,
        }
    selected = int(np.flatnonzero(oriented)[0])
    centre_row = jnp.asarray([census["fit_center_row"][0, selected]], dtype=jnp.int32)
    centre_column = jnp.asarray(
        [census["fit_center_column"][0, selected]], dtype=jnp.int32
    )
    fit = _fit_selected_centres(
        fields_tensor,
        jnp.asarray(radius, dtype=jnp.float64),
        jnp.asarray(height, dtype=jnp.float64),
        centre_row,
        centre_column,
    )
    return {
        "found": True,
        "axis_rz_m": [
            _scalar(census["r"][0, selected]),
            _scalar(census["z"][0, selected]),
        ],
        "ntype": int(ntype[selected]),
        "confidence": _scalar(census["confidence"][0, selected]),
        "minimum_curvature": _scalar(fit["minimum_curvature"][0]),
        "maximum_curvature": _scalar(fit["maximum_curvature"][0]),
        "candidate_count": int(np.asarray(census["candidate_count"][0]).reshape(-1)[0]),
    }


def _solve_continuation(
    profile: Any,
    group: zarr.Group,
    target_row: int,
    *,
    tolerance: float,
    case_state_seed: bool = False,
) -> tuple[Any, Any, dict[str, Any], list[dict[str, Any]]]:
    """Drive the writer's per-slice continuation up to the target row.

    With ``case_state_seed`` the target row is solved directly from its own
    reconstruction seed instead of the previous terminal, all earlier rows
    being skipped.  Mirrors ``scripts/labeller_batch/shard.py`` free-route
    behaviour per row and returns the target result, its inputs, the terminal
    readout and the trajectory.
    """
    state = None
    program = None
    trajectory: list[dict[str, Any]] = []
    target = None
    rows = (target_row,) if case_state_seed else range(target_row + 1)
    for row in rows:
        if case_state_seed and row != target_row:
            continue
        current = np.asarray(group["fcoil_c"][row], dtype=np.float64)
        scalars = {
            "time_s": float(group["time"][row]),
            "target_centroid_z": float(group["current_centrd_z"][row]),
            "reference_plasma_current": float(group["plasma_current_c"][row]),
        }
        if not np.all(np.isfinite(current)) or not all(
            np.isfinite(value) for value in scalars.values()
        ):
            state = None
            trajectory.append({"slice_index": row, "excluded": True})
            continue
        seed = _row_seed_on_mesh(profile, group, row)
        if not np.all(np.isfinite(seed)):
            state = None
            trajectory.append(
                {
                    "slice_index": row,
                    "excluded": True,
                    "reason": "non-finite reconstruction flux seed",
                }
            )
            continue
        seed_source = "reconstruction" if state is None else "previous_terminal"
        initial = jnp.asarray(seed if state is None else state)
        requested_class = jnp.asarray(_requested_class(group, row), dtype=jnp.int8)
        target_current = abs(scalars["reference_plasma_current"])
        started = time.perf_counter()
        result = reduced_newton.solve_reduced_newton(
            profile.operator,
            initial,
            requested_class=requested_class,
            target_current=target_current,
            prescribed_current=jnp.asarray(current),
            tolerance=tolerance,
            newton_steps=NEWTON_STEPS,
            program=program,
            stream=False,
        )
        jax.block_until_ready(result.state)
        wall_seconds = time.perf_counter() - started
        program = result.program
        state_values = np.asarray(result.state, dtype=np.float64)
        converged = bool(result.converged) and bool(np.all(np.isfinite(state_values)))
        item = {
            "slice_index": row,
            "time_s": scalars["time_s"],
            "excluded": False,
            "seed_source": seed_source,
            "requested_topology_class": int(np.asarray(requested_class)),
            "target_current_a": target_current,
            "converged": converged,
            "terminal_residual": _finite(result.terminal_residual),
            "active_set_iterations": int(result.active_set_iterations),
            "wall_seconds": wall_seconds,
        }
        trajectory.append(item)
        state = result.state if converged else None
        if row == target_row:
            target = (result, requested_class, scalars, current)
    if target is None:
        raise RuntimeError("the selected slice was excluded by the input gate")
    return target, trajectory


def _read_arm(
    profile: Any,
    result: Any,
    requested_class,
    inputs: dict[str, Any],
    current: np.ndarray,
) -> dict[str, Any]:
    """Read the terminal axis, residual, class and census curvature."""
    equilibrium = None
    topology_error = None
    try:
        equilibrium = equilibrium_from_reduced(
            profile, result, requested_class, inputs, current
        )
    except Exception as error:
        topology_error = f"{type(error).__name__}: {error}"
    axis = None
    axis_flux = None
    boundary_flux = None
    diverted = None
    if equilibrium is not None:
        topology = equilibrium.topology
        axis = _point(np.asarray(topology.axis, dtype=np.float64))
        axis_flux = _finite(topology.axis_flux)
        boundary_flux = _finite(topology.boundary_flux)
        diverted = bool(np.asarray(topology.diverted))
    census_error = None
    try:
        census = _axis_census_curvature(profile, result.state)
    except Exception as error:
        census_error = f"{type(error).__name__}: {error}"
        census = {
            "found": False,
            "axis_rz_m": None,
            "minimum_curvature": None,
            "maximum_curvature": None,
        }
    census["error"] = census_error
    return {
        "axis_rz_m": axis,
        "axis_flux_wb": axis_flux,
        "boundary_flux_wb": boundary_flux,
        "diverted": diverted,
        "final_residual": _finite(result.terminal_residual),
        "topology_class": int(np.asarray(requested_class)),
        "converged": bool(result.converged),
        "topology_error": topology_error,
        "census": census,
    }


def _run_arm(
    name: str,
    profile: Any,
    group: zarr.Group,
    target_row: int,
    *,
    tolerance: float,
    case_state_seed: bool,
) -> dict[str, Any]:
    """Run one arm and return its receipt block."""
    (result, requested_class, inputs, current), trajectory = _solve_continuation(
        profile,
        group,
        target_row,
        tolerance=tolerance,
        case_state_seed=case_state_seed,
    )
    readout = _read_arm(profile, result, requested_class, inputs, current)
    return {
        "arm": name,
        "selected_slice": target_row,
        "selected_time_s": inputs["time_s"],
        "seed_policy": (
            "explicit case-state reconstruction"
            if case_state_seed
            else "previous-terminal continuation"
        ),
        "route": "reduced_newton",
        "tolerance": tolerance,
        "trajectory": trajectory,
        **readout,
    }


def _offset_cm(
    value: tuple[float, float] | list[float] | None, reference: tuple[float, float]
) -> float | None:
    """Return the radial offset in centimetres from a reference axis."""
    if value is None or not np.all(np.isfinite(value)):
        return None
    return 100.0 * float(value[0] - reference[0])


def _draw_figure(receipt: dict[str, Any], path: Path) -> None:
    """Draw per-arm axis positions against EFIT and the banked references."""
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    colours = ["#1b6ca8", "#1b6ca8", "#b8461b", "#2e7d32", "#8e44ad", "#7f6b29"]
    axis_plot = axes[0]
    reference = BANKED["efit"]
    axis_plot.scatter(*reference, marker="*", s=160, color="#111111", label="EFIT")
    for label, position in (
        ("labeller config", BANKED["labeller_configuration"]),
        ("wide-grid parity", BANKED["wide_grid_parity_route"]),
    ):
        axis_plot.scatter(*position, marker="+", s=120, color="#999999")
        axis_plot.annotate(
            label,
            position,
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=8,
            color="#555555",
        )
    radial_offsets = []
    for index, arm in enumerate(receipt["arms"]):
        axis = arm.get("axis_rz_m")
        if axis is None:
            continue
        axis_plot.scatter(
            axis[0],
            axis[1],
            marker=markers[index],
            s=64,
            color=colours[index],
            edgecolors="#111111",
            linewidths=0.8,
            zorder=3,
        )
        axis_plot.annotate(
            arm["arm"],
            axis,
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
            color=colours[index],
        )
        radial_offsets.append((arm["arm"], 100.0 * (axis[0] - reference[0])))
    axis_plot.set_xlabel("R [m]")
    axis_plot.set_ylabel("Z [m]")
    axis_plot.set_title("Magnetic axis per arm (shot 21978, row 36, t = 0.195 s)")
    axis_plot.grid(alpha=0.25)
    axis_plot.set_aspect("equal", adjustable="box")

    offset_axis = axes[1]
    names = [item[0] for item in radial_offsets]
    values = [item[1] for item in radial_offsets]
    arm_order = ("L", "P", "S", "R", "W", "G")
    colours_for = {
        arm["arm"]: colours[arm_order.index(arm["arm"])] for arm in receipt["arms"]
    }
    offset_axis.bar(
        names, values, color=[colours_for[name] for name in names], alpha=0.85
    )
    offset_axis.axhline(1.0, color="#b2182b", ls="--", lw=0.9)
    offset_axis.axhline(0.0, color="#111111", lw=0.8)
    offset_axis.axhline(-4.11, color="#999999", ls=":", lw=0.9)
    offset_axis.set_ylabel("axis R offset from EFIT [cm]")
    offset_axis.set_title(">1 cm line dashed red; labeller banked offset dotted grey")
    offset_axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(
    output_json: Path = OUTPUT_JSON,
    output_figure: Path = OUTPUT_FIGURE,
    arms_to_run: tuple[str, ...] = ("L", "P", "S", "R", "W", "G"),
) -> dict:
    """Run the attribution arms and write their evidence artifacts.

    The receipt is persisted after each arm so a job cut short by its time
    fence still leaves the arms that completed.  The freshly built
    grid-resolution arm is ordered last because it is the only one whose
    response the shared carrier does not serve.
    """
    configure_dtypes()
    cache_root = os.environ.get("NOVA_JAX_CACHE_DIR")
    if cache_root is None:
        cache_root = str(default_persistent_compilation_cache_root())
    configure_persistent_compilation_cache(cache_root)
    print(f"stage: compilation cache {cache_root}", flush=True)
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    print("stage: shot store", flush=True)
    group = zarr.open_group(str(SHOT_STORE / f"{SHOT}.zarr"), mode="r")["efm"]
    times = np.asarray(group["time"], dtype=np.float64)
    row = int(np.argmin(np.abs(times - TARGET_TIME_S)))
    selected_time = float(times[row])
    if row != 36:
        raise AssertionError(f"expected row 36, selected {row}")
    efit_axis = (
        float(group["magnetic_axis_r"][row]),
        float(group["magnetic_axis_z"][row]),
    )
    mesh_overview = {
        "carrier_native_points": 33,
        "carrier_extent_m": [0.06, 2.0, -2.0, 2.0],
        "fine_points": FINE_GRID_POINTS,
    }

    specs = {
        "L": dict(
            shot=CARRIER_SHOT,
            slice_index=CARRIER_SLICE,
            response_cache=response_cache,
            mesh_points=None,
            tolerance=FIXED_POINT_CRITERION,
            case_state_seed=False,
            description="labeller reproduction: carrier operator, continuation"
            + "writer tolerance",
        ),
        "P": dict(
            shot=SHOT,
            slice_index=row,
            response_cache=response_cache,
            mesh_points=None,
            tolerance=FIXED_POINT_CRITERION,
            case_state_seed=False,
            description="parity reproduction: 21978-native operator, continuation",
        ),
        "G": dict(
            shot=CARRIER_SHOT,
            slice_index=CARRIER_SLICE,
            response_cache=None,
            mesh_points=FINE_GRID_POINTS,
            tolerance=FIXED_POINT_CRITERION,
            case_state_seed=False,
            description="grid factor: carrier anchor on a finer solve mesh",
        ),
        "S": dict(
            shot=CARRIER_SHOT,
            slice_index=CARRIER_SLICE,
            response_cache=response_cache,
            mesh_points=None,
            tolerance=FIXED_POINT_CRITERION,
            case_state_seed=True,
            description="seed factor: explicit case-state reconstruction seed",
        ),
        "R": dict(
            shot=CARRIER_SHOT,
            slice_index=CARRIER_SLICE,
            response_cache=response_cache,
            mesh_points=None,
            tolerance=TIGHT_TOLERANCE,
            case_state_seed=False,
            description="route factor: carrier operator with the tight residual finish",
        ),
        "W": dict(
            shot=SHOT,
            slice_index=row,
            response_cache=response_cache,
            mesh_points=None,
            tolerance=FIXED_POINT_CRITERION,
            case_state_seed=True,
            description="full parity configuration: native operator, case-state seed",
        ),
    }
    missing = [name for name in arms_to_run if name not in specs]
    if missing:
        raise AssertionError(f"unknown arms {missing}")
    selected = tuple(
        name for name in ("L", "P", "S", "R", "W", "G") if name in arms_to_run
    )

    def write_payload(payload: dict[str, Any]) -> None:
        """Persist the current receipt and figure at their destination paths."""
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _draw_figure(payload, output_figure)

    prior: list[dict[str, Any]] = []
    if output_json.is_file():
        try:
            prior = json.loads(output_json.read_text()).get("arms", [])
        except json.JSONDecodeError:
            prior = []
    arms = [arm for arm in prior if arm["arm"] not in selected]
    for name in selected:
        spec = specs[name]
        started = time.perf_counter()
        profile, _case = _profile_for(
            spec["shot"],
            spec["slice_index"],
            spec["response_cache"],
            mesh_points=spec["mesh_points"],
        )
        print(f"stage: {name} profile built", flush=True)
        arm = _run_arm(
            name,
            profile,
            group,
            row,
            tolerance=spec["tolerance"],
            case_state_seed=spec["case_state_seed"],
        )
        arm["description"] = spec["description"]
        arm["wall_seconds"] = time.perf_counter() - started
        arm["offset_from_efit_cm"] = _offset_cm(arm["axis_rz_m"], efit_axis)
        arms.append(arm)
        print(
            f"arm {name} done in {arm['wall_seconds']:.1f} s: axis "
            f"{arm['axis_rz_m']}, residual {arm['final_residual']:.4g}",
            flush=True,
        )
        payload = {
            "artifact": (
                "single-factor attribution of the magnetic-axis position on one "
                "MAST slice"
            ),
            "driver_sha256": _sha256(Path(__file__)),
            "shot": SHOT,
            "slice_index": row,
            "time_s": selected_time,
            "efit_axis_rz_m": list(efit_axis),
            "banked_nova_results_rz_m": {
                "labeller_configuration": list(BANKED["labeller_configuration"]),
                "wide_grid_parity_route": list(BANKED["wide_grid_parity_route"]),
            },
            "factors_varied": [
                "grid_extent_and_resolution",
                "seed_policy",
                "route_with_residual_tolerance",
            ],
            "mesh": mesh_overview,
            "evidence_inputs": {
                "shot_store": str(SHOT_STORE / f"{SHOT}.zarr"),
                "response_carrier": carrier_evidence,
            },
            "arms": arms,
            "complete": len(arms) == 6,
        }
        write_payload(payload)
    print(
        f"PASS: {len(arms)} arms run on 21978/{row} at {selected_time:.4f} s; "
        f"axis offsets from EFIT: "
        + "; ".join(
            f"{arm['arm']}={arm['offset_from_efit_cm']:.2f} cm" for arm in arms
        ),
        flush=True,
    )
    return {"arms": arms, "complete": len(arms) == 6}


def main() -> None:
    """Run from the command line with optional artifact destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-figure", type=Path, default=OUTPUT_FIGURE)
    parser.add_argument(
        "--arms",
        type=str,
        default="L,P,G,S,R,W",
        help="comma-separated arms to run (default all six)",
    )
    args = parser.parse_args()
    run(args.output_json, args.output_figure, tuple(args.arms.split(",")))


if __name__ == "__main__":
    main()
