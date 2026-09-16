"""Measure the weak-fixture centroid response to uniform exterior fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import fixture_positional_stiffness as stiffness
from benchmarks import oracle_start_newton_probe as oracle_probe
from benchmarks import solovev_certificate as certificate
from nova.equilibrium.constraint import ConstraintContext
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures.centroid_row import centroid_constraint_pair
from scripts.analytic_oracle_fixtures.measure import EXTERIOR_FIELD_COMPONENTS


ROOT = Path(__file__).resolve().parents[1]
CASE_NAME = "weak-rotation-reactor-static"
REQUESTED_CELLS = -110
REQUESTED_CLASS = int(stiffness.REQUESTED_CLASS)
FIELD_AMPLITUDES_T = np.asarray((-1.0e-2, -1.0e-3, 1.0e-3, 1.0e-2), dtype=np.float64)
FIELD_TO_CENTROID = {"vertical": "centroid_r", "radial": "centroid_z"}
CENTROID_INDEX = {"centroid_r": 0, "centroid_z": 1}


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _lane() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    device = jax.devices()[0]
    if job_id is None:
        raise RuntimeError("the response probe requires one scheduler allocation")
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"the response probe requires one H200, got {device}")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the response probe requires the betelgeuse partition")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the response probe requires gpu_0003_grpA")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "8":
        raise RuntimeError("the response probe requires eight requested CPUs")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR must be /tmp inside the allocation")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("the response probe requires JAX_PLATFORMS=cuda,cpu")
    return {
        "job_id": int(job_id),
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "reservation": os.environ["SLURM_JOB_RESERVATION"],
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "allocated_cpus": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "memory_mb": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "device": device.device_kind,
        "platform": device.platform,
        "jax_platforms": os.environ["JAX_PLATFORMS"],
        "tmpdir": os.environ["TMPDIR"],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _centroid(operator: Any, state: np.ndarray, target_current: float) -> np.ndarray:
    moments, _amplitude = operator.normalised_current_moments(
        jnp.asarray(state), target_current, REQUESTED_CLASS
    )
    current = np.asarray(jax.block_until_ready(moments.cell_current), dtype=np.float64)
    coordinates = np.asarray(operator.grid.coordinate, dtype=np.float64)
    total = float(np.sum(current))
    if not np.isfinite(total) or total == 0.0:
        raise RuntimeError("normalized current moments have no finite total current")
    return np.sum(current[:, None] * coordinates, axis=0) / total


def _directional_constraint_response(
    context: dict[str, Any],
    profile: ForwardProfile,
    pair: Any,
    field_index: int,
) -> np.ndarray:
    response = np.asarray(context["operator"].prescribed_current_field.response)
    step_t = 1.0e-6
    base_context = ConstraintContext(
        jnp.asarray(context["analytic"]),
        jnp.asarray(REQUESTED_CLASS),
        jnp.asarray(context["target_current"]),
        None,
    )
    perturbation = response[:, field_index] * step_t

    def observed(state: np.ndarray) -> np.ndarray:
        value = pair.functional.observed(
            profile, base_context._replace(flux=jnp.asarray(state)), None
        )
        return np.asarray(jax.block_until_ready(value), dtype=np.float64)

    return (
        observed(context["analytic"] + perturbation)
        - observed(context["analytic"] - perturbation)
    ) / (2.0 * step_t)


def _measure_row(
    context: dict[str, Any],
    production_map: Any,
    profile: ForwardProfile,
    pair: Any,
    directional_response: np.ndarray,
    component: str,
    amplitude_t: float,
    baseline_external: jax.Array,
    output_root: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    field_index = EXTERIOR_FIELD_COMPONENTS.index(component)
    prescribed = np.zeros(2, dtype=np.float64)
    prescribed[field_index] = amplitude_t
    external = context["operator"].external(prescribed_current=jnp.asarray(prescribed))
    state = np.asarray(context["analytic"], dtype=np.float64)
    target = context["current_centroid"]
    prediction = directional_response[:, field_index] * amplitude_t
    row = {
        "component": component,
        "field_amplitude_t": amplitude_t,
        "field_amplitude_mt": amplitude_t * 1.0e3,
        "prescribed_current": prescribed,
        "baseline_external_sha256_binary64": _digest(baseline_external),
        "analytic_centroid_m": target,
        "linear_prediction_centroid_shift_m": prediction,
        "trips": [],
        "completed": False,
    }
    one_map_state = None
    part = output_root / "parts" / f"{component}-{amplitude_t * 1.0e3:+g}mt.json"
    _write_json(part, row)
    for trip in range(1, 5):
        state = np.asarray(
            jax.block_until_ready(production_map(jnp.asarray(state), external)),
            dtype=np.float64,
        )
        if trip == 1:
            one_map_state = state.copy()
        measured = _centroid(context["operator"], state, context["target_current"])
        shift = measured - target
        prediction_index = CENTROID_INDEX[FIELD_TO_CENTROID[component]]
        predicted_value = float(prediction[prediction_index])
        measured_value = float(shift[prediction_index])
        ratio = (
            measured_value / predicted_value if predicted_value != 0.0 else float("nan")
        )
        row["trips"].append(
            {
                "trip": trip,
                "centroid_m": measured,
                "centroid_shift_m": shift,
                "centroid_shift_pitches": shift / context["pitch"],
                "target_component": FIELD_TO_CENTROID[component],
                "measured_target_shift_m": measured_value,
                "predicted_target_shift_m": predicted_value,
                "measured_to_predicted_ratio": ratio,
                "sign_agreement": bool(
                    np.sign(measured_value) == np.sign(predicted_value)
                    and measured_value != 0.0
                    and predicted_value != 0.0
                ),
                "state_sha256_binary64": _digest(state),
            }
        )
        _write_json(part, row)
    row["after_one_map"] = row["trips"][0]
    row["after_four_trips"] = row["trips"][3]
    row["completed"] = True
    _write_json(part, row)
    print(
        "CENTROID_FIELD_RESPONSE_PART "
        f"component={component} amplitude_mt={amplitude_t * 1.0e3:+g} "
        f"shift_m={row['after_one_map']['measured_target_shift_m']:.6e}",
        flush=True,
    )
    if one_map_state is None:
        raise RuntimeError("the response probe did not retain its one-map state")
    return row, one_map_state


def _draw_panel(
    context: dict[str, Any],
    component: str,
    amplitude_t: float,
    state: np.ndarray,
    path: Path,
) -> dict[str, Any]:
    wall = np.asarray(context["machine"].wall_node, dtype=np.float64)
    radial, height, analytic_field = certificate._raster_field(
        context["coordinates"], context["analytic"], wall
    )
    _, _, response_field = certificate._raster_field(
        context["coordinates"], state, wall
    )
    levels = poloidal.contour_levels(analytic_field, count=12)
    figure, axis = plt.subplots(1, 1, figsize=(5.4, 4.8), constrained_layout=True)
    poloidal.draw_flux_contours(
        axis, radial, height, analytic_field, levels, color="#3366cc"
    )
    poloidal.draw_flux_contours(
        axis, radial, height, response_field, levels, color="#cc7722"
    )
    poloidal.draw_wall(axis, units=(wall,))
    poloidal.draw_nulls(
        axis,
        magnetic_axis=context["analytic_topology"]["axis_rz_m"],
        x_points=context["analytic_topology"]["x_point_rz_m"],
        style=DEFAULT_INK.variant(
            axis_marker="^", axis_color="#3366cc", xpoint_color="#3366cc"
        ),
        contain=(wall,),
    )
    response_topology = oracle_probe._topology(context["operator"], state)
    poloidal.draw_nulls(
        axis,
        magnetic_axis=response_topology["axis_rz_m"],
        x_points=response_topology["x_point_rz_m"],
        style=DEFAULT_INK.variant(
            axis_marker="^", axis_color="#cc7722", xpoint_color="#cc7722"
        ),
        contain=(wall,),
    )
    poloidal_axes(axis)
    axis.set_title(
        f"{component} field +10 mT\nblue analytic / ochre mapped", fontsize=9
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "filesystem_path": str(path.relative_to(ROOT)),
        "project_absolute_src": f"/nova/{path.relative_to(ROOT / 'docs')}",
        "sha256": _file_digest(path),
        "component": component,
        "field_amplitude_t": amplitude_t,
        "levels_wb": levels,
    }


def _headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_rows = [
        row
        for row in rows
        if row["component"] in EXTERIOR_FIELD_COMPONENTS
        and row["field_amplitude_t"] == 1.0e-2
    ]
    return {
        "uniform_vertical_field_moves_radial_centroid": bool(
            target_rows
            and next(row for row in target_rows if row["component"] == "vertical")[
                "after_one_map"
            ]["sign_agreement"]
        ),
        "uniform_radial_field_moves_vertical_centroid": bool(
            target_rows
            and next(row for row in target_rows if row["component"] == "radial")[
                "after_one_map"
            ]["sign_agreement"]
        ),
        "all_one_map_target_signs_agree": all(
            row["after_one_map"]["sign_agreement"] for row in rows
        ),
        "all_four_trip_target_signs_agree": all(
            row["after_four_trips"]["sign_agreement"] for row in rows
        ),
    }


def _report(receipt: dict[str, Any]) -> str:
    lines = [
        "# Weak-fixture uniform-field response",
        "",
        "One compiled production map was reused with the cached analytic-clipped "
        "exterior held fixed while the prescribed uniform-field slot received "
        "signed 1 mT and 10 mT increments. The prediction is the contraction of "
        "the `CurrentCentroidConstraint` dual flux image with the same response "
        "column.",
        "",
        "| field | amplitude [mT] | one-map shift [m] | one-map shift [pitch] | "
        "four-trip shift [m] | four-trip shift [pitch] | linear prediction [m] | "
        "ratio at one map | sign | |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in receipt["rows"]:
        index = CENTROID_INDEX[FIELD_TO_CENTROID[row["component"]]]
        one = row["after_one_map"]
        four = row["after_four_trips"]
        lines.append(
            f"| {row['component']} | {row['field_amplitude_mt']:+g} | "
            f"{one['measured_target_shift_m']:.8e} | "
            f"{one['centroid_shift_pitches'][index]:+.8e} | "
            f"{four['measured_target_shift_m']:.8e} | "
            f"{four['centroid_shift_pitches'][index]:+.8e} | "
            f"{one['predicted_target_shift_m']:.8e} | "
            f"{one['measured_to_predicted_ratio']:.8e} | "
            f"{'agree' if one['sign_agreement'] else 'DISAGREE'} |"
        )
    lines.extend(
        [
            "",
            "## Panels",
            "",
            "Blue contours and null markers are analytic; ochre contours and null "
            "markers are the one-map state. Both panels use shared analytic flux "
            "levels and draw the wall.",
            "",
        ]
    )
    for panel in receipt["panels"]:
        lines.append(
            f"![{panel['component']} response]({panel['project_absolute_src']})"
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            f"The measured response headline is `{receipt['headline']}`. The "
            "receipt preserves both one-map and four-trip values because the dual "
            "image is a local linear prediction while the production map can move "
            "the state between trips.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_root: Path, report_path: Path) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the response probe requires extended precision")
    lane = _lane()
    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    started = perf_counter()
    try:
        context = stiffness._build_context(CASE_NAME, REQUESTED_CELLS)
        operator = context["operator"]
        baseline_external = operator.external()
        baseline_centroid = _centroid(
            operator, context["analytic"], context["target_current"]
        )
        baseline_centroid_error = baseline_centroid - context["current_centroid"]
        if not np.all(np.isfinite(baseline_centroid_error)):
            raise RuntimeError("the analytic centroid baseline is not finite")
        pitch = float(np.sqrt(np.median(np.asarray(context["machine"].area))))
        context["pitch"] = pitch
        pair = centroid_constraint_pair(
            context["current_centroid"],
            pitch=pitch,
            components=("centroid_r", "centroid_z"),
        )
        profile = ForwardProfile(
            operator,
            StencilMesh(
                context["machine"].node,
                context["machine"].stencil,
                context["machine"].area,
            ),
            newton_steps=certificate.recovery.NEWTON_STEPS,
        )
        directional_response = np.column_stack(
            tuple(
                _directional_constraint_response(context, profile, pair, index)
                for index in range(len(EXTERIOR_FIELD_COMPONENTS))
            )
        )
        production_map = jax.jit(
            operator.traced_flux_map(REQUESTED_CLASS, context["target_current"])
        )
        baseline_external = jax.block_until_ready(baseline_external)
        receipt_path = output_root / "receipt.json"
        receipt: dict[str, Any] = {
            "$id": "nova.weak-fixture-uniform-field-response",
            "revision": _revision(),
            "driver": {
                "path": str(Path(__file__).relative_to(ROOT)),
                "sha256": _file_digest(Path(__file__)),
            },
            "lane": lane,
            "case": CASE_NAME,
            "requested_cells": REQUESTED_CELLS,
            "realised_cells": len(context["machine"].node),
            "clip_mode": "exact",
            "persistent_compilation_cache": cache.receipt(),
            "compiled_map_reused_for_every_increment": True,
            "field_components": list(EXTERIOR_FIELD_COMPONENTS),
            "field_amplitudes_t": FIELD_AMPLITUDES_T,
            "target_centroid_m": context["current_centroid"],
            "baseline_centroid_m": baseline_centroid,
            "baseline_centroid_error_from_analytic_target_m": baseline_centroid_error,
            "characteristic_pitch_m": pitch,
            "cached_exterior": context["exteriors"]["analytic_clipped"],
            "baseline_external_sha256_binary64": _digest(baseline_external),
            "baseline_external_current": np.asarray(operator.external_current),
            "prescribed_field_current_at_baseline": np.asarray(
                operator.prescribed_current_field.current
            ),
            "response_sha256_binary64": _digest(
                operator.prescribed_current_field.response
            ),
            "constraint_response_sha256_binary64": _digest(directional_response),
            "prediction_method": (
                "central directional difference of CurrentCentroidConstraint "
                "along each prescribed exterior response column at 1 microtesla"
            ),
            "rows": [],
            "panels": [],
            "completed": False,
        }
        _write_json(receipt_path, receipt)
        panel_states: dict[str, np.ndarray] = {}
        for component in EXTERIOR_FIELD_COMPONENTS:
            for amplitude_t in FIELD_AMPLITUDES_T:
                row, one_map_state = _measure_row(
                    context,
                    production_map,
                    profile,
                    pair,
                    directional_response,
                    component,
                    float(amplitude_t),
                    baseline_external,
                    output_root,
                )
                receipt["rows"].append(row)
                if amplitude_t == 1.0e-2:
                    panel_states[component] = one_map_state
                _write_json(receipt_path, receipt)
        for component in EXTERIOR_FIELD_COMPONENTS:
            state = panel_states.get(component)
            if state is None:
                raise RuntimeError(f"missing +10 mT state for {component}")
            panel = _draw_panel(
                context,
                component,
                1.0e-2,
                state,
                output_root / "panels" / f"{component}-plus-10mt.png",
            )
            receipt["panels"].append(panel)
            _write_json(receipt_path, receipt)
        receipt["headline"] = _headline(receipt["rows"])
        receipt["elapsed_seconds"] = perf_counter() - started
        receipt["completed"] = True
        _write_json(receipt_path, receipt)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_report(receipt), encoding="utf-8")
        return receipt
    finally:
        set_support_clip_mode(previous_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.output_root, args.report)
    print(json.dumps(_strict(receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
