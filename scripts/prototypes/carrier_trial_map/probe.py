"""Record the topology and moment map at banked physical trial states."""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from nova.equilibrium import fixed_points, topology
from scripts.prototypes.fast_map_production.program import (
    FusedField,
    _density_coefficients,
)
from scripts.prototypes.sub_millisecond.measure_bell_cut import bell_polygon


def measure(context):
    """Use the certificate driver's own operands without taking another step."""
    assert jax.config.jax_enable_x64
    run = Path(os.environ["MEASURE_RUN"])
    bank = Path(os.environ["MEASURE_BANK"])
    arm = os.environ["MEASURE_ARM"]
    geometry = context["geometry"]
    scale = np.asarray(context["scale"])
    reference = np.asarray(context["reference"])
    source = np.load(bank / "split-first-step.npz")
    cold = source["point"].reshape(-1, 3)
    own = reference / scale
    distance = np.linalg.norm(own[:, None] - cold[None], axis=-1)
    rows, matched = linear_sum_assignment(distance)
    assert np.array_equal(rows, np.arange(len(rows)))
    assert np.max(distance[rows, matched]) < 1e-10
    split_receipt = json.loads((bank / "split-first.json").read_text())
    common = np.asarray(split_receipt["terminal_state"]["moments"])[matched]
    native_arm = "split" if arm == "split" else "integration"
    native_receipt = json.loads((bank / f"{native_arm}-first.json").read_text())
    native_source = np.load(bank / f"{native_arm}-first-step.npz")
    distance_native = np.linalg.norm(
        own[:, None] - native_source["point"].reshape(-1, 3)[None], axis=-1
    )
    native_rows, native_match = linear_sum_assignment(distance_native)
    assert np.max(distance_native[native_rows, native_match]) < 1e-10
    native = np.asarray(native_receipt["terminal_state"]["moments"])[native_match]
    payload = {"match": matched, "cold": reference, "scale": scale}
    for name in (
        "vertices",
        "vertex_count",
        "centre",
        "pitch",
        "full_area",
        "sample_points",
        "neighbour",
        "neighbour_edge",
        "fit_inverse",
        "contour_sample_points",
        "contour_sources",
        "contour_weights",
    ):
        payload[f"geometry.{name}"] = np.asarray(getattr(geometry, name))
    for name in (
        "vertex_rz",
        "vertex_valid",
        "vertex_is_wall",
        "edges",
        "edge_valid",
        "triangles",
        "triangle_valid",
    ):
        payload[f"tree.{name}"] = np.asarray(getattr(geometry.contour_mesh, name))
    window = context["window"]
    mapped = jax.jit(context["mapped"])

    def read_operand(
        moments,
        values,
        jets,
        window,
        rule,
        fit,
        design,
        geometry,
        seeds,
        convention,
        target,
    ):
        cached = fixed_points._CachedField(
            window.mesh, window.lookup, window.rows, moments, window.exterior
        )
        roots = fixed_points.solve(
            window,
            moments,
            seeds,
            seeds,
            jnp.median(geometry.pitch),
            context["axis_sign"],
        )
        flux = values.apply(moments).reshape(rule.points.shape[:-1])
        gauss = jnp.einsum("nkq,nq->nk", fit, flux)
        slots = window.lookup[jets.sample_index]
        safe = jnp.maximum(slots, 0)
        band = window.exterior[safe, 0] + jnp.einsum(
            "nscm,cm->ns", window.rows[safe, ..., 0], moments
        )
        coefficients = jnp.where(
            jnp.all(slots >= 0, axis=1)[:, None],
            jnp.einsum("nks,ns->nk", geometry.fit_inverse, band),
            gauss,
        )
        reading = context["selected_read"](
            FusedField(cached, coefficients),
            roots,
            geometry,
            convention,
            context["policy"],
        )
        return reading, roots

    read = jax.jit(read_operand)
    print(
        "MATCH",
        arm,
        "cells",
        len(rows),
        "max_cold_distance",
        float(np.max(distance[rows, matched])),
        "scale",
        scale.tolist(),
        flush=True,
    )
    for label, moments in (("cold", reference), ("native", native), ("common", common)):
        moments = jnp.asarray(moments)
        reading, roots = read(moments, *context["operands"])
        jax.block_until_ready(reading)
        state = mapped(moments, *context["operands"])
        jax.block_until_ready(state)
        payload[f"{label}.input"] = np.asarray(moments)
        for name in reading._fields:
            value = getattr(reading, name)
            if hasattr(value, "shape"):
                payload[f"{label}.read.{name}"] = np.asarray(value)
        for name in state._fields:
            payload[f"{label}.map.{name}"] = np.asarray(getattr(state, name))
        payload[f"{label}.density"] = np.asarray(
            jax.jit(
                lambda geometry, reading: _density_coefficients(
                    geometry, reading, context["profile"]
                )
            )(geometry, reading)
        )
        for name in ("position", "valid", "status"):
            if hasattr(roots, name):
                payload[f"{label}.roots.{name}"] = np.asarray(getattr(roots, name))
        cached = fixed_points._CachedField(
            window.mesh, window.lookup, window.rows, moments, window.exterior
        )
        field = FusedField(cached, reading.field_coefficients)
        tree_values = jax.jit(topology._point_values)(
            field, geometry.contour_sample_points
        )
        payload[f"{label}.tree_values"] = np.asarray(tree_values)
        polygon, count, valid = jax.jit(bell_polygon, static_argnums=(3,))(
            geometry, reading, context["convention"].sigma, 4
        )
        payload[f"{label}.polygon"] = np.asarray(polygon)
        payload[f"{label}.polygon_count"] = np.asarray(count)
        payload[f"{label}.polygon_valid"] = np.asarray(valid)
        residual = float(jnp.max(jnp.abs(state.moment_residual / scale)))
        print(
            "STAGE",
            arm,
            label,
            "residual",
            repr(residual),
            "axis",
            np.asarray(reading.axis).tolist(),
            "boundary",
            np.asarray(reading.boundary).tolist(),
            "boundary_flux",
            float(reading.boundary_flux),
            "boundary_class",
            int(reading.boundary_class),
            "valid",
            bool(state.booking_valid),
            flush=True,
        )
        np.savez(run / f"{arm}-stages.npz", **payload)
    print("SAVED", str(run / f"{arm}-stages.npz"), flush=True)
