"""Capture closed-form read outputs for compatibility checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    _, field, wall, _, saddle = fixture._analytic_inputs(args.kind)
    geometry = fixture._realised_hex_geometry(wall, args.cells)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0,
        normal_form_pitch_floor=1.5,
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    evaluate = jax.jit(topology.read)
    started = perf_counter()
    executable = evaluate.lower(field, geometry, convention, policy).compile()
    cold = perf_counter() - started
    started = perf_counter()
    result = executable(field, geometry, convention, policy)
    jax.block_until_ready(result)
    warm = perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        membership=np.asarray(result.membership),
        axis=np.asarray(result.axis),
        x_points=np.asarray(result.x_points),
        x_point_valid=np.asarray(result.x_point_valid),
        boundary_flux=np.asarray(result.boundary_flux),
        valid=np.asarray(result.valid),
        reason=np.asarray(result.reason),
    )
    print(
        f"SNAPSHOT case={args.kind} cells={args.cells} realised={len(geometry.pitch)} "
        f"valid={bool(result.valid)} reason={int(result.reason)} cold={cold:.3f} "
        f"warm={warm:.3f} path={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
