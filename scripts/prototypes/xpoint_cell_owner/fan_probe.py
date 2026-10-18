"""Inspect the read's own ring fan for one owner cell.

Scratch probe, not part of the node deliverable.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import current
from scripts.prototypes.fine_grid_convergence.context import build


def fan_area(rings, sizes):
    from nova.equilibrium.clip_quadrature import _quadrature_from_arrays

    rings = jnp.asarray(rings)
    counts = jnp.minimum(jnp.asarray(sizes), rings.shape[1])
    centres = rings[:, 0, :]
    points, weights = _quadrature_from_arrays(
        rings, counts, centres, jnp.ones(rings.shape[0], dtype=bool), signed=True
    )
    return float(jnp.sum(weights))


def polygon_area(vertices, count):
    vertices = np.asarray(vertices)[: int(count)]
    following = np.roll(vertices, -1, axis=0)
    return 0.5 * np.sum(vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    context = build(args.cells, args.cache)
    reading = context.reading(context.reference)
    geometry = context.geometry
    print(f"nova={nova.__file__}", flush=True)
    for idx in np.flatnonzero(np.asarray(reading.booking_cells)).tolist():
        sel = reading.fragment_selected[idx]
        iv = reading.edge_interval[idx]
        full = jnp.ones_like(iv)
        ones = jnp.ones_like(sel)
        lobe_r, lobe_s = current._normal_rings(
            geometry.vertices[idx], geometry.vertex_count[idx], geometry.centre[idx],
            reading.saddle_form, sel, iv, 4)
        all_iv_r, all_iv_s = current._normal_rings(
            geometry.vertices[idx], geometry.vertex_count[idx], geometry.centre[idx],
            reading.saddle_form, ones, iv, 4)
        full_r, full_s = current._normal_rings(
            geometry.vertices[idx], geometry.vertex_count[idx], geometry.centre[idx],
            reading.saddle_form, ones, full, 4)
        neg = reading.saddle_form._replace(
            positive=~reading.saddle_form.positive & reading.saddle_form.valid)
        neg_r, neg_s = current._normal_rings(
            geometry.vertices[idx], geometry.vertex_count[idx], geometry.centre[idx],
            neg, sel, iv, 4)
        print(f"cell {idx} poly_area={polygon_area(geometry.vertices[idx], geometry.vertex_count[idx]):.10g}")
        print(f"  lobe        area={fan_area(lobe_r, lobe_s):.10g}")
        print(f"  ones+readiv area={fan_area(all_iv_r, all_iv_s):.10g}")
        print(f"  ones+full   area={fan_area(full_r, full_s):.10g}")
        print(f"  negform     area={fan_area(neg_r, neg_s):.10g}")
        compl_r, compl_s = current._normal_rings(
            geometry.vertices[idx], geometry.vertex_count[idx], geometry.centre[idx],
            reading.saddle_form, ones, jnp.ones_like(iv) - iv, 4)
        print(f"  ones+1-readiv area={fan_area(compl_r, compl_s):.10g}")
        comp = 1.0 - iv[..., ::-1]
        cmp_r, cmp_s = current._normal_rings(
            geometry.vertices[idx], geometry.vertex_count[idx], geometry.centre[idx],
            reading.saddle_form, ones, comp, 4)
        print(f"  ones+flip   area={fan_area(cmp_r, cmp_s):.10g}")
        print(f"  poly-lobe   area={polygon_area(geometry.vertices[idx], geometry.vertex_count[idx]) - fan_area(lobe_r, lobe_s):.10g}")
        print(f"  lobe+neg    area={fan_area(lobe_r,lobe_s)+fan_area(neg_r,neg_s):.10g}")


if __name__ == "__main__":
    main()