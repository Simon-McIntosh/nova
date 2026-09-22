"""Re-apply the forward flux map once at a persisted terminal state.

Reloads whole-cell production-row terminal flux from its persisted part,
rebuilds the operator with the re-posed exterior exactly as the row driver
did, applies the production map once at the terminal state, and reports the
fixed-point residual by three independent definitions:

* the solver's own relative sup-norm scalar ``max|g-x| / max|g|``,
* the rms and sup of ``map - state`` over the residual-entering carriers as a
  fraction of the terminal flux span, and
* the same measure restricted to the profile-owned cells.

It also reports the residual shadow census (total, entering and excluded
carriers and the flood/wall/sample decomposition), re-evaluates the unmasked
physical map at the terminal state to decide whether the terminal is a genuine
fixed point of the map or whether the recorded scalar came from the shadow
copy-through emptying the entering set, and traces the qualification bound to
the policy field and code line that set it.

The weak and moderate 2500-cell rows are measured identically.  This is an
evidence node: it rebuilds and re-applies, never re-solves.

``--render-only`` rebuilds the panels from the arrays the measurement pass
persisted under ``--output-root/parts/render``; it applies no map, touches no
solver and runs anywhere in seconds.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from time import perf_counter
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.jax.config import configure_dtypes
from nova.media import ink
from nova.media import poloidal
from nova.media.ink import poloidal_axes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "docs/figures/cut-cell-current-attribution/zero-residual"
ROW_ORDER = (
    ("weak-rotation-reactor-static", -2500),
    ("moderate-rotation-conventional-static", -2500),
)
CLIP_MODE = "chord"
COORDINATE_REBUILD_TOLERANCE_M = 1.0e-6
FIGURE_STEM = "zero-residual-map"
FLUX_LEVEL_COUNT = 10
RESIDUAL_LEVEL_COUNT = 8

# The reference (analytic) null set is drawn beside the solved one in its own
# style: a hollow triangle for the reference axis, so a reader can see the
# position error of the solve against the fixture rather than one glyph alone.
REFERENCE_NULL_INK = replace(
    ink.DEFAULT_INK,
    xpoint_marker="^",
    xpoint_color="#0b7285",
    xpoint_markersize=ink.DEFAULT_INK.axis_markersize,
    xpoint_markeredgewidth=1.4,
)


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _driver_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _device_record() -> dict[str, Any]:
    device = jax.devices()[0]
    return {
        "platform": device.platform,
        "kind": device.device_kind,
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(certificate._strict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _row_part_path(case_name: str) -> Path:
    return (
        ROOT
        / "docs"
        / "figures"
        / "cut-cell-current-attribution"
        / "wholecell-rows"
        / "parts"
        / "rows"
        / f"{case_name}-production-route-cells-2500.json"
    )


@contextmanager
def _monitor_live_peak() -> Iterator[list[int]]:
    """Sample live device bytes in the background over one row."""
    peak: list[int] = [0]
    stopped = threading.Event()

    def sample() -> None:
        while not stopped.wait(0.25):
            try:
                stats = jax.devices()[0].memory_stats() or {}
                live = int(stats.get("bytes_in_use", 0))
            except Exception:  # noqa: BLE001 - absent on the host backend
                return
            if live > peak[0]:
                peak[0] = live

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield peak
    finally:
        stopped.set()
        worker.join()


def _load_part(case_name: str) -> dict[str, Any]:
    path = _row_part_path(case_name)
    if not path.exists():
        raise RuntimeError(f"persisted row part missing: {path}")
    part = json.loads(path.read_text(encoding="utf-8"))
    if "render_data" not in part:
        raise RuntimeError(f"row part carries no render data: {path}")
    return part


def _rebuild_operator(
    case_name: str,
    part: dict[str, Any],
) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    """Rebuild the production operator at the persisted terminal geometry.

    Returns ``(machine, operator, coordinates, cache_record)``.  The rebuild
    mirrors the row driver's operator construction so the single map
    application runs against the identical re-posed exterior.
    """
    requested_cells = -2500
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = certificate.oracle_fixture.forward_operator(source_case, machine)
    _exact_physical, fixture_exterior, exterior_cache = (
        certificate.oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty_operator, oracle_state
        )
    )
    operator = certificate.oracle_fixture.forward_operator(
        source_case, machine, fixture_exterior
    )
    cache_record = {
        "machine": machine.cache,
        "fixture_exterior": exterior_cache,
    }
    return machine, operator, coordinates, cache_record


def _carrier_statistics(
    operator: Any,
    terminal: np.ndarray,
    target_current: float,
    external: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate masked and unmasked maps once and fold the three measures.

    Returns ``(statistics, physical_diff, cold_j, promoted_j)``: the folded
    dict plus the raw unmasked residual array and both shadow boolean arrays
    over all carriers, which the figure renderer needs and which would bloat
    the persisted part if folded into it.
    """

    terminal_j = jnp.asarray(terminal)
    external_j = jnp.asarray(external)
    target_j = jnp.asarray(target_current, dtype=np.float64)

    physical_j = external_j + operator.internal(terminal_j, None, target_j)
    shadow_cold = operator.residual_shadow_mask(terminal_j, None)
    shadow_promoted = operator.residual_shadow_mask(
        terminal_j, None, previous_shadow=shadow_cold
    )
    masked_cold = jnp.where(shadow_cold, terminal_j, physical_j)
    masked_promoted = jnp.where(shadow_promoted, terminal_j, physical_j)

    def scalar(mapped: jax.Array) -> jax.Array:
        return jnp.max(jnp.abs(mapped - terminal_j)) / jnp.maximum(
            jnp.max(jnp.abs(mapped)), 1.0e-30
        )

    def entering_measure(mapped: jax.Array, entering: jax.Array) -> dict[str, float]:
        if not bool(jnp.any(entering)):
            return {"count": 0, "rms": None, "sup": None}
        diff = jnp.abs(mapped - terminal_j)[entering]
        return {
            "count": int(jnp.sum(entering)),
            "rms": float(jnp.sqrt(jnp.mean(diff**2))),
            "sup": float(jnp.max(diff)),
        }

    physical = np.asarray(physical_j, dtype=np.float64)
    masks = operator.current_domain_masks(terminal_j, None)
    profile_extended = np.concatenate(
        (
            np.asarray(masks.profile_participation, dtype=bool),
            np.zeros(len(terminal) - len(masks.profile_participation), dtype=bool),
        )
    )

    cold_j = np.asarray(shadow_cold, dtype=bool)
    promoted_j = np.asarray(shadow_promoted, dtype=bool)
    entering_cold = ~cold_j
    entering_promoted = ~promoted_j
    physical_diff = np.abs(physical - terminal)

    return (
        {
            "shadow_census_cold": {
                "total": int(len(cold_j)),
                "entering": int(np.count_nonzero(entering_cold)),
                "excluded": int(np.count_nonzero(cold_j)),
                "excluded_fraction": float(np.mean(cold_j)),
            },
            "shadow_census_promoted": {
                "total": int(len(promoted_j)),
                "entering": int(np.count_nonzero(entering_promoted)),
                "excluded": int(np.count_nonzero(promoted_j)),
                "excluded_fraction": float(np.mean(promoted_j)),
            },
            "solver_scalar": {
                "persisted": None,
                "recomputed_cold": float(np.asarray(scalar(jnp.asarray(masked_cold)))),
                "recomputed_promoted": float(
                    np.asarray(scalar(jnp.asarray(masked_promoted)))
                ),
                "unmasked_all_carriers": float(np.asarray(scalar(physical_j))),
            },
            "masked_diff": {
                "cold": entering_measure(masked_cold, entering_cold),
                "promoted": entering_measure(masked_promoted, entering_promoted),
            },
            "physical_unmasked": {
                "sup_all": float(np.max(physical_diff)),
                "sup_entering_cold": float(np.max(physical_diff[entering_cold])),
                "sup_shadowed_cold": float(np.max(physical_diff[cold_j])),
                "sup_entering_promoted": float(
                    np.max(physical_diff[entering_promoted])
                ),
                "sup_shadowed_promoted": float(np.max(physical_diff[promoted_j])),
                "argmax_index": int(np.argmax(physical_diff)),
            },
            "profile_owned": {
                "entering_cold": int(
                    np.count_nonzero(entering_cold & profile_extended)
                ),
                "profile_total": int(np.count_nonzero(profile_extended)),
            },
        },
        physical_diff,
        cold_j,
        promoted_j,
    )


def _topology_record(operator: Any, terminal: np.ndarray) -> dict[str, Any]:
    return certificate._topology(operator, terminal)


def _measure_row(
    case_name: str,
    part: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load, rebuild, apply once, and fold the residual evidence for one row.

    Returns ``(row, figure_payload)``: the persisted evidence plus the arrays
    the shared-level panel renderer needs, which are too large for JSON.
    """
    row_started = perf_counter()
    live_peak: list[int] = [0]
    with _monitor_live_peak() as peak:
        machine, operator, coordinates, cache_record = _rebuild_operator(
            case_name, part
        )
        terminal = np.asarray(part["render_data"]["terminal_flux_wb"], dtype=np.float64)
        if len(terminal) != len(coordinates):
            raise RuntimeError(
                f"state length {len(terminal)} != rebuilt coordinates "
                f"{len(coordinates)}"
            )
        persisted_coordinates = np.asarray(
            part["render_data"]["coordinates_rz_m"], dtype=np.float64
        )
        coordinate_delta = float(
            np.max(np.abs(coordinates - persisted_coordinates))
            if coordinates.shape == persisted_coordinates.shape
            else np.inf
        )
        exceed_tolerance = coordinate_delta > COORDINATE_REBUILD_TOLERANCE_M
        if not np.isfinite(coordinate_delta) or exceed_tolerance:
            raise RuntimeError(
                f"rebuilt coordinates diverge from the persisted row: "
                f"max delta {coordinate_delta:g} m"
            )

        target_current = float(part["solver"]["target_current_a"])
        external = np.asarray(operator.external(None, None), dtype=np.float64)

        statistics, physical_diff, cold_j, promoted_j = _carrier_statistics(
            operator, terminal, target_current, external
        )
        topology = _topology_record(operator, terminal)
        live_peak[:] = peak
    wall_seconds = perf_counter() - row_started

    span = float(topology["flux_span_wb"])
    abs_span = abs(span)
    statistics["masked_diff"]["cold"]["relative_rms"] = (
        statistics["masked_diff"]["cold"]["rms"] / abs_span
        if statistics["masked_diff"]["cold"]["rms"] is not None
        else None
    )
    statistics["masked_diff"]["cold"]["relative_sup"] = (
        statistics["masked_diff"]["cold"]["sup"] / abs_span
        if statistics["masked_diff"]["cold"]["sup"] is not None
        else None
    )
    statistics["masked_diff"]["promoted"]["relative_rms"] = (
        statistics["masked_diff"]["promoted"]["rms"] / abs_span
        if statistics["masked_diff"]["promoted"]["rms"] is not None
        else None
    )
    statistics["masked_diff"]["promoted"]["relative_sup"] = (
        statistics["masked_diff"]["promoted"]["sup"] / abs_span
        if statistics["masked_diff"]["promoted"]["sup"] is not None
        else None
    )

    row = {
        "schema": "nova.zero-residual-check-row",
        "case": case_name,
        "requested_cells": -2500,
        "source_part_relative": str(_row_part_path(case_name).relative_to(ROOT)),
        "source_revision": _source_revision(),
        "clip_mode": support_clip_mode(),
        "state_length": len(terminal),
        "grid_node_number": int(len(machine.node)),
        "wall_node_number": int(len(machine.wall_node)),
        "sample_node_number": int(len(machine.sample_coordinates)),
        "coordinate_rebuild_max_delta_m": coordinate_delta,
        "cache": cache_record,
        "topology": topology,
        "span_wb": span,
        "resolution_scaled_position_bound_m": part["characteristic_pitch_m"],
        "statistics": statistics,
        "persisted_solver": {
            "terminal_fixed_point_residual": part["solver"][
                "terminal_fixed_point_residual"
            ],
            "qualification_bound": part["solver"]["qualification_bound"],
            "qualification": part["solver"]["qualification"],
            "converged": part["solver"]["production_telemetry"]["converged"],
        },
        "peak_live_bytes_observed": live_peak[0],
        "wall_seconds": wall_seconds,
    }
    carrier_case, _source_case, exact = certificate._case(case_name)
    boundary = certificate._boundary(case_name, exact)
    caption = {
        "case": case_name,
        "recorded_scalar": row["persisted_solver"]["terminal_fixed_point_residual"],
        "converged": row["persisted_solver"]["converged"],
        "span_wb": span,
        "entering_cold": statistics["shadow_census_cold"]["entering"],
        "entering_promoted": statistics["shadow_census_promoted"]["entering"],
        "sup_all": statistics["physical_unmasked"]["sup_all"],
        "recomputed_unmasked_scalar": statistics["solver_scalar"][
            "unmasked_all_carriers"
        ],
    }
    figure_payload = {
        "case": case_name,
        "coordinates": coordinates,
        "terminal": terminal,
        "physical_diff": physical_diff,
        "wall": np.asarray(machine.wall_node, dtype=np.float64),
        "boundary": boundary,
        "topology": topology,
        "reference_topology": part["render_data"]["analytic_topology"],
        "reference_receipt": str(_row_part_path(case_name).relative_to(ROOT)),
        "span_wb": span,
        "caption": caption,
    }
    return row, figure_payload


def _bundle_dir(output_root: Path) -> Path:
    return output_root / "parts" / "render"


def _bundle_paths(output_root: Path, case_name: str) -> tuple[Path, Path]:
    directory = _bundle_dir(output_root)
    return directory / f"{case_name}.npz", directory / f"{case_name}-meta.json"


def _write_bundle(
    output_root: Path, case_name: str, payload: dict[str, Any]
) -> dict[str, str]:
    """Persist the arrays and labels a re-render needs, with no map application.

    The measured payload is written beside the row record so a later render-only
    run rebuilds the figure from this directory alone.
    """
    array_path, meta_path = _bundle_paths(output_root, case_name)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        array_path,
        coordinates=np.asarray(payload["coordinates"], dtype=np.float64),
        terminal=np.asarray(payload["terminal"], dtype=np.float64),
        physical_diff=np.asarray(payload["physical_diff"], dtype=np.float64),
        wall=np.asarray(payload["wall"], dtype=np.float64),
        boundary=np.asarray(payload["boundary"], dtype=np.float64),
    )
    meta = {
        key: payload[key]
        for key in (
            "case",
            "topology",
            "reference_topology",
            "reference_receipt",
            "span_wb",
            "caption",
        )
    }
    _write_json(meta_path, meta)
    return {
        "arrays": str(array_path.relative_to(ROOT)),
        "meta": str(meta_path.relative_to(ROOT)),
    }


def _load_bundle(output_root: Path, case_name: str) -> dict[str, Any]:
    array_path, meta_path = _bundle_paths(output_root, case_name)
    if not array_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"render bundle missing for {case_name}: run the measurement pass first "
            f"({array_path} / {meta_path})"
        )
    arrays = np.load(array_path)
    payload: dict[str, Any] = {name: arrays[name] for name in arrays.files}
    payload.update(json.loads(meta_path.read_text(encoding="utf-8")))
    return payload


def _null_points(topology: dict[str, Any]) -> tuple[Any, Any]:
    """Return the finite ``(axis, x_points)`` of one null set, or ``None``."""
    axis = topology.get("axis_rz_m")
    x_points = topology.get("x_point_rz_m")
    axis_array = None if axis is None else np.asarray(axis, dtype=float).reshape(-1)[:2]
    if axis_array is not None and not np.all(np.isfinite(axis_array)):
        axis_array = None
    if x_points is None:
        x_array = None
    else:
        candidate = np.atleast_2d(np.asarray(x_points, dtype=float))
        finite = candidate[np.all(np.isfinite(candidate[:, :2]), axis=1)]
        x_array = finite if finite.size else None
    return axis_array, x_array


def _resolve_levels(raster: np.ndarray, count: int) -> list[float]:
    """Stated contour levels for one raster, never drawn empty.

    The two rows differ in flux span by a factor of about thirty-six, so a
    level array taken over their union carries levels inside neither: the
    narrow row then draws no contour at all. Levels are stated per raster.
    """
    finite = np.asarray(raster, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return []
    levels = poloidal.contour_levels(finite, count=count)
    return [float(level) for level in np.asarray(levels).reshape(-1)]


def _render_panels(
    output_root: Path,
    payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Render the terminal-flux and relative-residual panels for each row.

    One row of two panels per case: the persisted terminal flux as line
    contours (the state the map is re-applied to) and the unmasked relative
    residual ``|map - state| / |span|`` on the same geometry.  The contour
    levels are stated per panel, because the two rows differ in flux span by a
    factor of about thirty-six and a level array taken over their union falls
    inside neither of the narrower row's values, drawing it empty.

    The terminal-flux panel carries its stated level array, the wall, the
    boundary, the solved null set and the analytic reference null set in
    distinct styles, and both panel titles carry the recorded residual and the
    converged flag.  Every title line and the per-set glyph counts are written
    to ``render-receipt.json`` beside the panels, so a reader can check the
    figure's own claim about what it drew.
    """
    import matplotlib.pyplot as plt

    names = list(payloads)
    figure, axes = plt.subplots(
        len(names), 2, figsize=(9.5, 4.6 * len(names)), constrained_layout=True
    )
    if len(names) == 1:
        axes = axes[None, :]

    flux_rasters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rel_rasters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, payload in payloads.items():
        radial, height, flux = certificate._raster_field(
            payload["coordinates"], payload["terminal"], payload["wall"]
        )
        _, _, residual = certificate._raster_field(
            payload["coordinates"], payload["physical_diff"], payload["wall"]
        )
        flux_rasters[name] = (radial, height, flux)
        rel = residual / abs(payload["span_wb"])
        rel_rasters[name] = (radial, height, rel)

    flux_levels = {
        name: _resolve_levels(raster, FLUX_LEVEL_COUNT)
        for name, (_r, _h, raster) in flux_rasters.items()
    }
    residual_levels = {
        name: _resolve_levels(raster, RESIDUAL_LEVEL_COUNT)
        for name, (_r, _h, raster) in rel_rasters.items()
    }
    panels: list[dict[str, Any]] = []

    for row, name in enumerate(names):
        payload = payloads[name]
        radial, height, flux = flux_rasters[name]
        _, _, rel = rel_rasters[name]
        wall_units = (payload["wall"],)
        topology = payload["topology"]
        reference = payload.get("reference_topology") or {}
        caption = payload["caption"]
        residual = float(caption["recorded_scalar"])
        converged = bool(caption["converged"])
        marker = f"residual={residual:.3e}  converged={converged}"

        flux_axis = axes[row, 0]
        solved_axis, solved_x = _null_points(topology)
        reference_axis, reference_x = _null_points(reference)
        flux_level_array = flux_levels[name]
        if flux_level_array:
            flux_title = (
                f"{name} terminal flux (persisted)  levels={len(flux_level_array)}"
            )
        else:
            flux_title = (
                f"{name} terminal flux: no persisted flux raster exists "
                f"({payload.get('reference_receipt')})"
            )
        poloidal.draw_flux_contours(
            flux_axis, radial, height, flux, flux_level_array, color="#5a6bd6"
        )
        poloidal.draw_wall(flux_axis, units=wall_units)
        poloidal.draw_boundary(
            flux_axis, payload["boundary"][:, 0], payload["boundary"][:, 1]
        )
        solved_tally = poloidal.draw_nulls(
            flux_axis,
            magnetic_axis=solved_axis,
            x_points=solved_x,
            contain=wall_units,
        )
        reference_tally = poloidal.draw_nulls(
            flux_axis,
            magnetic_axis=None,
            x_points=None,
            other_x_points=(
                None if reference_x is None else np.asarray(reference_x, dtype=float)
            ),
            style=REFERENCE_NULL_INK,
        )
        reference_axis_tally = poloidal.draw_nulls(
            flux_axis,
            magnetic_axis=None,
            x_points=None,
            other_x_points=(
                None
                if reference_axis is None
                else np.asarray(reference_axis, dtype=float)[None, :]
            ),
            style=REFERENCE_NULL_INK,
        )
        poloidal_axes(flux_axis)
        flux_axis.set_title(f"{flux_title}\n{marker}", fontsize=8)

        residual_axis = axes[row, 1]
        residual_level_array = residual_levels[name]
        residual_title = (
            f"{name} |map-state|/|span|  levels={len(residual_level_array)}"
        )
        poloidal.draw_flux_contours(
            residual_axis, radial, height, rel, residual_level_array, color="#7a3e9d"
        )
        poloidal.draw_wall(residual_axis, units=wall_units)
        poloidal.draw_boundary(
            residual_axis,
            payload["boundary"][:, 0],
            payload["boundary"][:, 1],
            color="#35b9c8",
        )
        poloidal_axes(residual_axis)
        residual_axis.set_title(f"{residual_title}\n{marker}", fontsize=8)

        panels.append(
            {
                "case": name,
                "kind": "terminal_flux",
                "title": f"{flux_title}\n{marker}",
                "persisted_raster_present": bool(flux_level_array),
                "level_count": len(flux_level_array),
                "levels_wb": flux_level_array,
                "reference_receipt": payload.get("reference_receipt"),
                "null_glyphs": {
                    "solved_axis": int(solved_axis is not None),
                    "solved_x_points": int(solved_tally["x_points_drawn"]),
                    "solved_x_points_dropped_outside_wall": int(
                        solved_tally["x_points_dropped_outside_wall"]
                    ),
                    "solved_other_x_points": int(solved_tally["other_x_points_drawn"]),
                    "reference_axis": int(reference_axis_tally["other_x_points_drawn"]),
                    "reference_x_points": int(reference_tally["other_x_points_drawn"]),
                },
            }
        )
        panels.append(
            {
                "case": name,
                "kind": "relative_residual",
                "title": f"{residual_title}\n{marker}",
                "level_count": len(residual_level_array),
                "levels_relative_span": residual_level_array,
            }
        )

    figure.suptitle(
        "cut-cell whole-cell map re-applied once at the persisted terminal state",
        fontsize=10,
    )
    path = output_root / "panels" / f"{FIGURE_STEM}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    _write_json(
        output_root / "panels" / f"{FIGURE_STEM}-caption.json",
        {
            "flux_levels_wb": flux_levels,
            "residual_levels_relative_span": residual_levels,
            "series": {name: payload["caption"] for name, payload in payloads.items()},
        },
    )
    figure_record = {
        "png_relative": str(Path(os.path.abspath(path)).relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "src": (
            "/nova/figures/cut-cell-current-attribution/zero-residual/panels/"
            "zero-residual-map.png"
        ),
        "render_source": "zero_residual_check",
    }
    render_receipt = {
        "schema": "nova.zero-residual-render-receipt",
        "source_revision": _source_revision(),
        "driver_sha256": _driver_sha256(),
        "figures": {FIGURE_STEM: {"figure": figure_record, "panels": panels}},
    }
    _write_json(output_root / "render-receipt.json", render_receipt)
    return {
        "figure": figure_record,
        "panels": panels,
        "render_receipt": str(
            (output_root / "render-receipt.json").relative_to(ROOT)
        ),
    }


def _verdict(weak: dict[str, Any], moderate: dict[str, Any]) -> dict[str, Any]:
    """Decide whether the recorded zero was a genuine fixed point or the empty
    shadowed entering set, and name the code line that set the bound."""

    qual_w = weak["persisted_solver"]
    res_w = weak["statistics"]["physical_unmasked"]
    scalar_w = weak["statistics"]["solver_scalar"]
    enter_w = weak["statistics"]["shadow_census_cold"]["entering"]
    promoted_w = scalar_w["recomputed_promoted"]
    enter_p = moderate["statistics"]["shadow_census_cold"]["entering"]
    res_p = moderate["statistics"]["physical_unmasked"]
    return {
        "weak_2500": {
            "recorded_scalar": qual_w["terminal_fixed_point_residual"],
            "entering_carriers": enter_w,
            "unmasked_sup_all": res_w["sup_all"],
            "unmasked_sup_entering": res_w["sup_entering_cold"],
            "unmasked_sup_shadowed": res_w["sup_shadowed_cold"],
            "recomputed_promoted_scalar": promoted_w,
            "unmasked_relative_scalar": scalar_w["unmasked_all_carriers"],
            "qualification": qual_w["qualification"],
            "converged": qual_w["converged"],
            "verdict": (
                "zero_follows_empty_entering_set"
                if enter_w == 0
                else "genuine_fixed_point_of_the_map"
                if scalar_w["unmasked_all_carriers"] < 1.0e-12
                else "masked_fixed_point_rejects_shadowed_disagreement"
            ),
        },
        "moderate_2500": {
            "recorded_scalar": moderate["persisted_solver"][
                "terminal_fixed_point_residual"
            ],
            "entering_carriers": enter_p,
            "unmasked_sup_all": res_p["sup_all"],
            "unmasked_sup_entering": res_p["sup_entering_cold"],
            "unmasked_sup_shadowed": res_p["sup_shadowed_cold"],
            "qualification": moderate["persisted_solver"]["qualification"],
        },
        "qualification_bound_trace": {
            "bound_value": 1.0e-12,
            "source_field": "LOCKED_RECOVERY_BOUNDS['fixed_point_residual']",
            "source_line": "tests/test_solovev_recovery_gates.py:28",
            "carrier_constant": "TERMINAL_RESIDUAL_BOUND",
            "carrier_line": "benchmarks/solovev_certificate.py:129",
            "applied_as": [
                "kernel_tolerance",
                "qualification_tolerance",
            ],
            "application_lines": [
                "benchmarks/solovev_certificate.py:2101-2102",
            ],
            "evaluation_site": "_apply_joint_qualification",
            "evaluation_line": "benchmarks/solovev_certificate.py:1764-1776",
            "recorded_solver_scalar_source": (
                "equilibrium.fixed_point live_residual (active-set reconcile), "
                "relative sup max|g-x|/max|g|"
            ),
            "shadow_line": "nova/equilibrium/forward_operator.py:2611 "
            "_exclude_shadow_residual copies shadowed carriers through",
            "note": (
                "the plan comment conflates the recorded residual VALUE (0.0) "
                "with the qualification BOUND; the bound is 1e-12, the value 0.0"
            ),
        },
    }


def _run(output_root: Path, rows: list[tuple[str, int]]) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the zero-residual check requires binary64")
    if support_clip_mode() != CLIP_MODE:
        set_support_clip_mode(CLIP_MODE)
    if support_clip_mode() != CLIP_MODE:
        raise RuntimeError(f"zero-residual clip mode {CLIP_MODE!r} is not active")

    certificate.DIAGNOSTIC_ROOT = output_root / "diagnostics"
    certificate.FIGURE_ROOT = output_root / "panels"
    certificate.PART_ROOT = output_root / "parts" / "rows"

    measured: dict[str, Any] = {}
    figure_payloads: dict[str, dict[str, Any]] = {}
    receipt_path = output_root / "receipt.json"
    for case_name, requested_cells in rows:
        try:
            part = _load_part(case_name)
            measured[case_name], figure_payloads[case_name] = _measure_row(
                case_name, part
            )
        except Exception:
            _write_json(
                receipt_path,
                {
                    "schema": "nova.zero-residual-check-acceptance",
                    "source_revision": _source_revision(),
                    "driver_sha256": _driver_sha256(),
                    "clip_mode": support_clip_mode(),
                    "completed": False,
                    "rows": measured,
                },
            )
            raise
        _write_json(
            output_root / "parts" / f"{case_name}-cells-{abs(requested_cells)}.json",
            measured[case_name],
        )
        _write_bundle(output_root, case_name, figure_payloads[case_name])
    render_result = _render_panels(output_root, figure_payloads)
    verdict = (
        _verdict(measured[rows[0][0]], measured[rows[1][0]])
        if len(rows) == 2
        else {"rows_completed": [name for name, _cells in rows]}
    )
    _write_json(
        receipt_path,
        {
            "schema": "nova.zero-residual-check-acceptance",
            "source_revision": _source_revision(),
            "driver_sha256": _driver_sha256(),
            "clip_mode": support_clip_mode(),
            "device": _device_record(),
            "figure": render_result["figure"],
            "panels": render_result["panels"],
            "render_receipt": render_result["render_receipt"],
            "rows": measured,
            "verdict": verdict,
            "completed": True,
        },
    )
    for name, row in measured.items():
        print(
            f"ZERO_RESIDUAL_ROW case={name} "
            f"recorded={row['persisted_solver']['terminal_fixed_point_residual']} "
            f"entering_cold={row['statistics']['shadow_census_cold']['entering']} "
            f"sup_all={row['statistics']['physical_unmasked']['sup_all']:.3e} "
            f"span={row['span_wb']:.6g}",
            flush=True,
        )
    print("ZERO_RESIDUAL_EXIT=0", flush=True)
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def _run_render_only(
    output_root: Path, rows: list[tuple[str, int]]
) -> dict[str, Any]:
    """Rebuild the panels from the persisted bundles, applying no map.

    This is the render path: it reads the arrays the measurement pass wrote
    and the labels beside them, and it touches no solver and no device beyond
    matplotlib, so it can run anywhere and finishes in seconds.
    """
    payloads = {
        case_name: _load_bundle(output_root, case_name) for case_name, _ in rows
    }
    render_result = _render_panels(output_root, payloads)
    receipt_path = output_root / "receipt.json"
    receipt: dict[str, Any] = {}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["figure"] = render_result["figure"]
    receipt["panels"] = render_result["panels"]
    receipt["render_receipt"] = render_result["render_receipt"]
    receipt["render_only"] = {
        "rows": [name for name, _cells in rows],
        "source_revision": _source_revision(),
    }
    _write_json(receipt_path, receipt)
    for panel in render_result["panels"]:
        print(
            f"ZERO_RESIDUAL_RENDER_PANEL kind={panel['kind']} case={panel['case']} "
            f"levels={panel['level_count']}",
            flush=True,
        )
    print("ZERO_RESIDUAL_RENDER_EXIT=0", flush=True)
    return receipt


def _parse_rows(arguments: argparse.Namespace) -> list[tuple[str, int]]:
    if arguments.case:
        return [(arguments.case, -2500)]
    return list(ROW_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[name for name, _cells in ROW_ORDER],
        help="measure only this case",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="where parts, panels and the receipt land",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the row plan and clip mode without applying any map",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "rebuild the panels from the persisted render bundles in "
            "--output-root, applying no map"
        ),
    )
    arguments = parser.parse_args()

    rows = _parse_rows(arguments)
    if arguments.render_only:
        _run_render_only(arguments.output_root, rows)
        return
    if arguments.dry_run:
        configure_dtypes()
        print("ZERO_RESIDUAL_DRY_RUN rows=%d" % len(rows))
        for case_name, _requested_cells in rows:
            print(f"ZERO_RESIDUAL_DRY_RUN_ROW case={case_name} cells=2500")
        clip = f"{support_clip_mode()} default->{CLIP_MODE}"
        print(f"ZERO_RESIDUAL_DRY_RUN clip_mode={clip}")
        return
    _run(arguments.output_root, rows)


if __name__ == "__main__":
    main()
