"""Locate the analytic fixture's centroid derivative discrepancy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import centroid_constrained_fixture_receipt as fixture
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.observation import MomentIntegralSupport
from nova.jax.config import configure_dtypes


OUTPUT = Path(__file__).resolve().parent / "derivative-factors.json"


def _host(tree):
    return jax.tree.map(lambda value: np.asarray(jax.block_until_ready(value)), tree)


def _metrics(values):
    return {
        "max_absolute": float(np.max(np.abs(values))),
        "sum": float(np.sum(values)),
        "l2": float(np.linalg.norm(values)),
    }


def main():
    configure_dtypes()
    assert jax.config.jax_enable_x64
    set_support_clip_mode("exact")
    context = fixture._context("weak-rotation-reactor-static", -110)
    operator = context["profile"].operator
    profile = context["profile"]
    target = context["target_current"]
    state = jnp.asarray(context["analytic"], dtype=jnp.float64)
    columns = jnp.asarray(operator.prescribed_current_field.response)
    base_partition = operator._support_partition(state, 0)
    base_support = base_partition[3]
    point = jnp.asarray(operator.grid.coordinate)

    def factors(psi, *, frozen=False):
        masks, topology, sample, support = operator._support_partition(psi, 0)
        if frozen:
            support = base_support
        raw = operator._partitioned_current_moments(
            (masks, topology, sample, support)
        ).cell_current
        raw_total = jnp.sum(raw)
        amplitude = operator.current_normalisation_amplitude(target, raw_total)
        current = amplitude * raw
        total = jnp.sum(current)
        numerator = jnp.sum(current[:, None] * point, axis=0)
        centroid = numerator / total
        area = jnp.asarray(support.area)
        first = jnp.asarray(support.first_area_moment)
        safe_area = jnp.where(area != 0.0, area, 1.0)
        cut_centre = point + first / safe_area[:, None]
        return {
            "centroid": centroid,
            "numerator": numerator,
            "total": total,
            "raw_current": raw,
            "current": current,
            "raw_total": raw_total,
            "amplitude": amplitude,
            "area": area,
            "first_area_moment": first,
            "cut_centre": cut_centre,
        }

    def production(psi):
        row = profile.current_moment_observation(
            psi,
            support=MomentIntegralSupport.ALL_DOMAIN,
            requested_class=0,
            target_current=target,
        )
        return jnp.stack((row.centroid_r, row.centroid_z))

    digest = hashlib.sha256(np.asarray(state, dtype="<f8").tobytes()).hexdigest()
    base = _host(factors(state))
    control = _host(production(state))
    if not np.allclose(base["centroid"], control, rtol=0, atol=2e-12):
        raise RuntimeError("factor reconstruction misses the production centroid")
    result = {
        "revision": "b4103b477e2807790996fbc9b7edeb07778cd433",
        "analytic_state_digest": digest,
        "clip_mode": "exact",
        "requested_class": 0,
        "realised_cells": int(point.shape[0]),
        "base_centroid_m": control.tolist(),
        "directions": {},
    }
    print(f"base verified {digest} centroid {control.tolist()}", flush=True)
    if digest != "d2c980a88374751bb6e4af9305ae9ad39bc0f19bb7f954654c08d6fa8862f8f2":
        raise RuntimeError("analytic state differs from prior receipt")

    for name, column, steps in (
        ("vertical_t", columns[:, 0], [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]),
        ("level_wb", columns[:, 2], [1e-4]),
    ):
        print(f"direction {name}: JVP", flush=True)
        _, tangent = jax.jvp(factors, (state,), (column,))
        _, frozen_tangent = jax.jvp(
            lambda psi: factors(psi, frozen=True), (state,), (column,)
        )
        _, production_tangent = jax.jvp(production, (state,), (column,))
        tangent = _host(tangent)
        frozen_tangent = _host(frozen_tangent)
        production_tangent = _host(production_tangent)
        rows = []
        for step in steps:
            plus = _host(factors(state + step * column))
            minus = _host(factors(state - step * column))
            difference = jax.tree.map(lambda a, b: (a - b) / (2 * step), plus, minus)
            plus_frozen = _host(factors(state + step * column, frozen=True))
            minus_frozen = _host(factors(state - step * column, frozen=True))
            frozen_difference = jax.tree.map(
                lambda a, b: (a - b) / (2 * step), plus_frozen, minus_frozen
            )
            production_difference = (
                _host(production(state + step * column))
                - _host(production(state - step * column))
            ) / (2 * step)
            row = {
                "step": step,
                "centroid_difference": difference["centroid"].tolist(),
                "production_difference": production_difference.tolist(),
                "raw_current_difference": _metrics(difference["raw_current"]),
                "raw_current_jvp_error": _metrics(
                    tangent["raw_current"] - difference["raw_current"]
                ),
                "frozen_current_difference": _metrics(frozen_difference["raw_current"]),
                "frozen_current_jvp_error": _metrics(
                    frozen_tangent["raw_current"] - frozen_difference["raw_current"]
                ),
                "geometry_current_difference": _metrics(
                    difference["raw_current"] - frozen_difference["raw_current"]
                ),
                "geometry_current_jvp": _metrics(
                    tangent["raw_current"] - frozen_tangent["raw_current"]
                ),
                "area_difference": _metrics(difference["area"]),
                "area_jvp_error": _metrics(tangent["area"] - difference["area"]),
                "first_area_moment_difference": _metrics(
                    difference["first_area_moment"]
                ),
                "first_area_moment_jvp_error": _metrics(
                    tangent["first_area_moment"] - difference["first_area_moment"]
                ),
                "cut_centre_difference": _metrics(difference["cut_centre"]),
                "cut_centre_jvp_error": _metrics(
                    tangent["cut_centre"] - difference["cut_centre"]
                ),
                "amplitude_difference": float(difference["amplitude"]),
                "amplitude_jvp_error": float(
                    tangent["amplitude"] - difference["amplitude"]
                ),
                "raw_total_difference": float(difference["raw_total"]),
                "raw_total_jvp_error": float(
                    tangent["raw_total"] - difference["raw_total"]
                ),
                "numerator_difference": difference["numerator"].tolist(),
                "numerator_jvp_error": (
                    tangent["numerator"] - difference["numerator"]
                ).tolist(),
                "total_difference": float(difference["total"]),
                "total_jvp_error": float(tangent["total"] - difference["total"]),
                "active_cells_minus": int(np.count_nonzero(minus["current"])),
                "active_cells_plus": int(np.count_nonzero(plus["current"])),
            }
            rows.append(row)
            print(
                f"{name} step {step}: centroid {row['centroid_difference']}", flush=True
            )
        result["directions"][name] = {
            "centroid_jvp": tangent["centroid"].tolist(),
            "production_jvp": production_tangent.tolist(),
            "raw_current_jvp": _metrics(tangent["raw_current"]),
            "frozen_current_jvp": _metrics(frozen_tangent["raw_current"]),
            "area_jvp": _metrics(tangent["area"]),
            "first_area_moment_jvp": _metrics(tangent["first_area_moment"]),
            "cut_centre_jvp": _metrics(tangent["cut_centre"]),
            "amplitude_jvp": float(tangent["amplitude"]),
            "raw_total_jvp": float(tangent["raw_total"]),
            "numerator_jvp": tangent["numerator"].tolist(),
            "total_jvp": float(tangent["total"]),
            "steps": rows,
        }
        OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
