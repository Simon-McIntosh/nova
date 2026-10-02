"""Measure the analytic-state map and prescribed exterior flux columns."""

# Precision must be configured before importing fixture modules with arrays.
# ruff: noqa: E402

from pathlib import Path
import json
import os
import subprocess
import sys
import time

import jax
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
assert jax.default_backend() == "cpu"

import jax.numpy as jnp
import numpy as np
from benchmarks import centroid_constrained_fixture_receipt as driver
from benchmarks import plasma_cell_map_fidelity as fidelity
from nova.equilibrium.constraint import ConstraintContext
from nova.equilibrium.forward_operator import set_support_clip_mode


ROOT = Path.cwd().resolve()
OUTPUT = Path(__file__).resolve().parent


def write(name, payload):
    (OUTPUT / name).write_text(
        json.dumps(driver._strict(payload), indent=2, allow_nan=False) + "\n"
    )


def compare(actual, expected):
    scale = float(np.dot(actual, expected) / np.dot(expected, expected))
    error = actual - expected
    sup = float(np.max(np.abs(error)))
    return {
        "scale": scale,
        "sup_error_wb": sup,
        "sup_relative": sup / float(np.max(np.abs(expected))),
        "spatial_deviation_after_scale_wb": float(
            np.max(np.abs(actual - scale * expected))
        ),
        "pass": bool(np.allclose(actual, expected, rtol=2e-14, atol=2e-13)),
    }


def main():
    start = time.monotonic()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={ROOT} command={sys.argv!r}", flush=True)
    print(f"module.__file__={driver.__file__}", flush=True)
    print(
        f"cwd={ROOT} job={os.environ.get('SLURM_JOB_ID')} backend=cpu x64=True",
        flush=True,
    )
    set_support_clip_mode("exact")
    context = driver._context("weak-rotation-reactor-static", -110)
    print(
        f"CONTEXT cells={len(context['machine'].node)} "
        f"seconds={time.monotonic() - start:.3f}",
        flush=True,
    )
    assert len(context["machine"].node) == 135
    profile = context["profile"]
    operator = profile.operator
    coordinates = context["coordinates"]
    analytic = context["analytic"]
    radius, height = coordinates.T
    anchor = float(context["axis_point"][0, 0])
    # Integrate dpsi/dR = 2*pi*R*Bz and dpsi/dZ = -2*pi*R*Br.
    forms = np.column_stack(
        (np.pi * (radius**2 - anchor**2), -2 * np.pi * radius * height)
    )
    controls = {
        "analytic_self": [compare(forms[:, i], forms[:, i].copy()) for i in range(2)]
    }
    controls["reversed_sign"] = compare(-forms[:, 0], forms[:, 0])
    controls["missing_total_flux_factor"] = compare(
        forms[:, 0] / (2 * np.pi), forms[:, 0]
    )
    assert all(row["pass"] for row in controls["analytic_self"])
    assert not controls["reversed_sign"]["pass"]
    assert not controls["missing_total_flux_factor"]["pass"]
    pair = driver._certificate_pairs(context, level=True, field_scale_t=0.25)[0]
    constraint_context = ConstraintContext(
        jnp.asarray(analytic), None, jnp.asarray(context["target_current"]), None
    )
    columns = []
    results = []
    for index, name in enumerate(("vertical", "radial")):
        unit = jnp.eye(2, dtype=jnp.float64)[index] / pair.unknown.field_scale
        actual = np.asarray(
            pair.unknown.flux_delta(
                profile, constraint_context, pair.functional, pair.binding.payload, unit
            )
        )
        columns.append(actual)
        result = compare(actual, forms[:, index])
        result["component"] = name
        result["normalized_unit_amplitude"] = np.asarray(unit)
        # A unit response is a basis measurement, not an admissible 1 T command.
        try:
            pair.unknown.require_within_bound(unit)
        except ValueError as error:
            result["unit_command_refusal"] = str(error)
        else:
            raise AssertionError("the 1 T command must exceed the 0.25 T bound")
        prescribed = jnp.zeros(3, dtype=jnp.float64).at[index].set(1.0)
        exterior_delta = np.asarray(
            operator.external(prescribed_current=prescribed) - operator.external()
        )
        result["exterior_composition"] = compare(exterior_delta, forms[:, index])
        assert result["pass"] and result["exterior_composition"]["pass"]
        results.append(result)
    write(
        "field-response.json",
        {
            "revision": revision,
            "module": driver.__file__,
            "cwd": str(ROOT),
            "case": context["case_name"],
            "realised_cells": 135,
            "flux_unit": "Wb total poloidal flux",
            "anchor_radius_m": anchor,
            "vertical_closed_form": "pi * B_z * (R**2 - R_axis**2)",
            "radial_closed_form": "-2*pi*B_r*R*Z; B_z = -B_r*Z/R",
            "controls": controls,
            "columns": results,
            "coordinates_rz_m": coordinates,
            "expected_unit_flux_wb": forms,
            "actual_unit_flux_wb": np.column_stack(columns),
        },
    )
    print("FIELD_RESPONSE " + json.dumps(driver._strict(results)), flush=True)
    # This is the same map callable as ForwardProfile.flux_map; passing the
    # operator as an argument keeps mesh-sized arrays out of closure constants.
    request = driver._certificate_request(context)
    evaluate = jax.jit(operator.traced_flux_map(None, request.target_current))
    external = operator.external(request.current, request.prescribed_current)
    mapped = np.asarray(
        evaluate(jnp.asarray(analytic), external, operator, request.target_current)
    )
    residual = mapped - analytic
    print("MAP " + json.dumps(fidelity.norms(residual, analytic)), flush=True)
    shifted = coordinates.copy()
    shifted[:, 1] -= context["pitch"]
    shifted_state = driver.certificate._exact_state(
        context["case_name"], context["exact"], shifted
    )
    shifted_map = np.asarray(
        evaluate(jnp.asarray(shifted_state), external, operator, request.target_current)
    )
    shifted_error = fidelity.norms(shifted_map - analytic, analytic)
    assert shifted_error["sup_wb"] > 10 * np.max(np.abs(residual))
    observe = jax.jit(
        lambda state: pair.functional.observed(
            profile, constraint_context._replace(flux=state), None
        )
    )
    baseline = np.asarray(observe(jnp.asarray(analytic)))
    leverage = []
    for index in range(2):
        delta = jnp.asarray(columns[index]) * 1e-6
        leverage.append(
            np.asarray(
                (
                    observe(jnp.asarray(analytic) + delta)
                    - observe(jnp.asarray(analytic) - delta)
                )
                / 2e-6
            )
        )
    leverage = np.column_stack(leverage)
    historical = json.loads(
        (
            ROOT
            / "docs/figures/centroid-constrained-oracle-solve"
            / "field-response/receipt.json"
        ).read_text()
    )
    converged = json.loads(
        (
            ROOT
            / "docs/figures/centroid-constrained-oracle-solve"
            / "cap-factor-repair/control-positive.json"
        ).read_text()
    )
    prior = json.loads(
        (
            ROOT
            / "docs/figures/plasma-cell-read-fidelity/map-fidelity"
            / "weak-rotation-reactor-static-cells-110-exact.json"
        ).read_text()
    )
    offset = baseline - context["centroid"]
    need = -offset[0] / leverage[0, 0]
    old_offset = historical["baseline_centroid_error_from_analytic_target_m"][0]
    old_leverage = (
        historical["rows"][0]["linear_prediction_centroid_shift_m"][0]
        / historical["rows"][0]["field_amplitude_t"]
    )
    old_need = -old_offset / old_leverage
    field = converged["solve"]["compensating_field_t"][0]
    wall = np.asarray(operator.wall.coordinate)
    grid_count = len(context["machine"].node)
    shadow = np.asarray(operator.residual_shadow_mask(jnp.asarray(analytic), None))
    receipt = {
        "revision": revision,
        "module": driver.__file__,
        "cwd": str(ROOT),
        "case": context["case_name"],
        "requested_cells": -110,
        "realised_cells": grid_count,
        "state_nodes": len(analytic),
        "clip_mode": "exact",
        "target_current_a": context["target_current"],
        "cache": context["cache"],
        "gauge": "fixture analytic exterior; no re-zeroing",
        "all_nodes": fidelity.norms(residual, analytic),
        "cell_nodes": fidelity.norms(residual[:grid_count], analytic[:grid_count]),
        "residual_min_wb": float(residual.min()),
        "residual_max_wb": float(residual.max()),
        "max_error_coordinate_rz_m": coordinates[np.argmax(np.abs(residual))],
        "shadow_copied_nodes": int(shadow.sum()),
        "shifted_input_control": shifted_error,
        "reference_nulls": fidelity.nulls(operator, analytic),
        "mapped_nulls": fidelity.nulls(operator, mapped),
        "prior_map_fidelity": {
            k: prior[k]
            for k in (
                "source_revision",
                "realised_cells",
                "mismatch",
                "jax_backend",
                "clip_mode",
            )
        },
        "centroid": {
            "analytic_target_m": context["centroid"],
            "discrete_analytic_state_m": baseline,
            "offset_m": offset,
            "leverage_m_per_t": leverage,
            "local_cancel_field_t": need,
            "historical_offset_m": old_offset,
            "historical_leverage_m_per_t": old_leverage,
            "historical_local_cancel_field_t": old_need,
            "converged_field_t": field,
            "converged_over_local_prediction": field / need,
            "converged_over_historical_prediction": field / old_need,
            "converged_source_revision": converged["source_revision"],
        },
        "coordinates_rz_m": coordinates,
        "analytic_flux_wb": analytic,
        "mapped_flux_wb": mapped,
        "residual_wb": residual,
        "shadow_mask": shadow,
        "wall_rz_m": wall,
        "wall_unit_offsets": np.asarray(operator.wall_unit_offsets),
        "wall_unit_closed": np.asarray(operator.wall_unit_closed),
        "completed": True,
        "seconds": time.monotonic() - start,
    }
    write("map-residual.json", receipt)
    print("CENTROID " + json.dumps(driver._strict(receipt["centroid"])), flush=True)
    print(f"COMPLETE seconds={time.monotonic() - start:.3f}", flush=True)


if __name__ == "__main__":
    main()
