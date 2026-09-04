"""Measure nested topology components on the production profiling state."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import inspect
import json
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp

from benchmarks import trip_quantum_profile as trip_profile
from benchmarks.receipt_raster_check import _profile_and_seed
from nova.biot.null import NullBase
from nova.equilibrium import connectivity_boundary as boundary
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache
from nova.linalg.tensor_spline import fit_tensor_spline


_original_null_post_init = NullBase.__post_init__


def _sentinel_safe_null_post_init(self) -> None:
    if not hasattr(self.coordinate, "shape"):
        return
    _original_null_post_init(self)


NullBase.__post_init__ = _sentinel_safe_null_post_init


def _revision() -> str:
    root = Path(os.environ.get("NOVA_PROFILE_ROOT", Path.cwd()))
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _limiter_program(profile, state, requested):
    operator = profile.operator
    physical = state[: operator.physical_node_number]
    fixed_topology = operator._fixed_design_topology
    _masks, topology, _connected, _admitted = operator._fixed_design_read(
        physical, requested
    )
    grid_flux, wall_flux = fixed_topology.split_flux_map(physical)
    radius, height, shape = operator.connectivity_grid_axes()
    radial_count, vertical_count = shape
    values = grid_flux.reshape((radial_count, vertical_count)).T
    surface = fit_tensor_spline(radius, height, values)
    _seed, material = operator.connectivity_axis_seed(topology.axis)
    inside = material.reshape((radial_count, vertical_count)).T
    selected_x = jnp.concatenate((topology.x_point, topology.x_point_flux[None]))[
        None, :
    ]
    selected_wall = jnp.concatenate(
        (topology.wall_point, topology.wall_point_flux[None])
    )
    accepts_selected_support = (
        "selected_wall"
        in inspect.signature(boundary._select_reachable_wall_limiter).parameters
    )

    def limiter_tangency(surface_value, map_values, exact_wall, x_value, wall_value):
        psi_axis = surface_value(topology.axis[0], topology.axis[1])
        edge = jnp.concatenate(
            [map_values[0, :], map_values[-1, :], map_values[:, 0], map_values[:, -1]]
        )
        psi_out = edge[boundary._argmax_exact(jnp.abs(edge - psi_axis))]
        span = psi_out - psi_axis
        span_safe = jnp.where(jnp.abs(span) < 1.0e-30, 1.0e-30, span)
        normalized = (map_values - psi_axis) / span_safe
        x_present = jnp.all(jnp.isfinite(x_value[:, :3]), axis=1)
        x_inside = boundary._points_inside_polygon(
            x_value[:, 0],
            x_value[:, 1],
            operator.wall.coordinate[:, 0],
            operator.wall.coordinate[:, 1],
        )
        x_valid = x_present & x_inside
        safe_x_r = jnp.where(x_valid, x_value[:, 0], topology.axis[0])
        safe_x_z = jnp.where(x_valid, x_value[:, 1], topology.axis[1])
        x_flux = jnp.where(x_valid, surface_value(safe_x_r, safe_x_z), psi_axis)
        x_level = (x_flux - psi_axis) / span_safe
        x_index = boundary._argmin_exact(jnp.where(x_valid, x_level, jnp.inf))
        pre_saddle = boundary._axis_component_before_level(
            normalized,
            inside,
            radius,
            height,
            topology.axis[0],
            topology.axis[1],
            x_level[x_index],
        )
        arguments = (
            map_values,
            radius,
            height,
            inside,
            operator.wall.coordinate[:, 0],
            operator.wall.coordinate[:, 1],
            exact_wall,
            pre_saddle,
            psi_axis,
            surface_value,
            topology.axis[0],
            topology.axis[1],
        )
        if accepts_selected_support:
            return boundary._select_reachable_wall_limiter(
                *arguments, selected_wall=wall_value
            )
        return boundary._select_reachable_wall_limiter(*arguments)

    _arc, sample_r, sample_z = boundary._sample_wall_polyline(
        operator.wall.coordinate[:, 0],
        operator.wall.coordinate[:, 1],
        boundary._WALL_REACHABILITY_SAMPLES,
    )

    def line_of_sight(target_r, target_z):
        return boundary._wall_nodes_in_line_of_sight(
            topology.axis[0],
            topology.axis[1],
            target_r,
            target_z,
            operator.wall.coordinate[:, 0],
            operator.wall.coordinate[:, 1],
        )

    def tensor_spline_fit(map_values):
        return fit_tensor_spline(radius, height, map_values)

    return {
        "tensor_spline_fit": (tensor_spline_fit, (values,)),
        "limiter_tangency": (
            limiter_tangency,
            (surface, values, wall_flux, selected_x, selected_wall),
        ),
        "line_of_sight": (line_of_sight, (sample_r, sample_z)),
    }, {
        "line_of_sight_target_count": int(sample_r.size),
        "limiter_uses_selected_segment": accepts_selected_support,
    }


def run(output: Path, cache_root: Path, repeats: int, preflight: bool) -> None:
    configure_dtypes()
    case, profile, target_current, _carrier, _policy = _profile_and_seed()
    state = jnp.asarray(case["state"])
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    standard = trip_profile._component_programs(
        profile, state, requested, target_current
    )
    nested, contract = _limiter_program(profile, state, requested)
    selected = {
        "census": standard["candidate_census"],
        "census_values_by_spline": standard["spline_fits"],
        "flood_fills": standard["flood_fills"],
        "wall_reachability": standard["wall_reachability"],
        "separatrix": standard["separatrix"],
        **nested,
    }
    if preflight:
        shaped = {
            name: jax.eval_shape(function, *arguments)
            for name, (function, arguments) in selected.items()
        }
        print(
            json.dumps(
                {
                    "status": "preflight_complete",
                    "revision": _revision(),
                    "programs": sorted(shaped),
                    "contract": contract,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    trip_profile._require_h200()
    cache = configure_persistent_compilation_cache(
        cache_root, minimum_compile_seconds=0.0
    )
    probes = {}
    for name, (function, arguments) in selected.items():
        _executable, probes[name] = trip_profile._compile_probe(
            name, function, arguments, repeats
        )
    payload = {
        "schema": "nova.fine_topology_component_profile",
        "captured_at": datetime.now(UTC).isoformat(),
        "revision": _revision(),
        "scheduler": trip_profile._scheduler(),
        "runtime": trip_profile._runtime(),
        "persistent_compilation_cache": cache.receipt(),
        "contract": contract,
        "direct_probes": probes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"FINE_PROFILE_WRITTEN={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    run(arguments.output, arguments.cache_root, arguments.repeats, arguments.preflight)


if __name__ == "__main__":
    main()
