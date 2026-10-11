"""Inspect which support predicate refuses an analytic carrier."""

from __future__ import annotations

import argparse

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402

from nova.equilibrium import topology  # noqa: E402
from nova.equilibrium.solve_request import TopologyPolicy  # noqa: E402
from tests.equilibrium import test_topology_read as fixture  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    _, field, wall, _, saddle = fixture._analytic_inputs(args.kind)
    geometry = fixture._realised_hex_geometry(wall, args.cells)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0,
        normal_form_pitch_floor=1.5,
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    result = jax.jit(topology.read)(field, geometry, convention, policy)
    jax.block_until_ready(result)
    selected = np.asarray(result.fragment_selected)
    areas = np.asarray(result.fragment_area)
    raw = np.sum(np.where(selected, areas, 0), axis=1) / np.asarray(geometry.full_area)
    local = (geometry.vertices - geometry.centre[:, None, :]) / geometry.pitch[
        :, None, None
    ]
    signed = convention.sigma * result.field_coefficients
    signed = signed.at[:, 0].add(-convention.sigma * result.boundary_flux)
    fragments = jax.vmap(topology.quadratic_cell_fragments, in_axes=(0, 0, 0, None))(
        local, geometry.vertex_count, signed, policy.fragment_capacity
    )
    valid = np.asarray(fragments.valid)
    labels = np.asarray(result.fragment_component)
    outside = (raw < -policy.edge_tolerance) | (raw > 1 + policy.edge_tolerance)
    invalid_cells = np.flatnonzero(~valid)
    outside_cells = np.flatnonzero(outside)
    print(
        f"DIAG case={args.kind} cells={args.cells} valid={bool(result.valid)} "
        f"reason={int(result.reason)} "
        f"fragment_invalid={len(invalid_cells)} "
        f"raw_nonfinite={int(np.count_nonzero(~np.isfinite(raw)))} "
        f"raw_outside={len(outside_cells)} "
        f"raw_min={float(np.nanmin(raw))} raw_max={float(np.nanmax(raw))} "
        f"selected={int(np.count_nonzero(selected))} "
        f"distinct_labels={len(np.unique(labels))} "
        f"axis={np.asarray(result.axis).tolist()} "
        f"boundary_flux={float(result.boundary_flux)}",
        flush=True,
    )
    print("INVALID_CELLS=" + repr(invalid_cells[:50].tolist()), flush=True)
    print("OUTSIDE_CELLS=" + repr(outside_cells[:50].tolist()), flush=True)


if __name__ == "__main__":
    main()
