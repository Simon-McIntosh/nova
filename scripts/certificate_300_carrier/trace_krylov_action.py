"""Trace the first CPU Krylov linear action of the diverted 300-cell solve.

The qualification in ``nova/equilibrium/fixed_point.py::_qualified_krylov...``
refuses a step when ``probe_action = linear_action(probe)`` holds a non-finite
entry, where ``linear_action(v) = v - tangent(v)`` and ``tangent`` is the
directional derivative of the write-then-read flux map at the seed state.
This script reconstructs that exact action outside the solver, decomposes the
map into its stages, and names the operator, the coordinate and the stage whose
derivative first turns non-finite.

Usage: trace_krylov_action.py <out-dir>
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


def nonfinite_indices(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    return np.flatnonzero(~np.isfinite(array))


def block_of(index: int, sizes: dict[str, int]) -> tuple[str, int]:
    start = 0
    for name in ("grid", "wall", "sample"):
        stop = start + sizes[name]
        if start <= index < stop:
            return name, index - start
        start = stop
    return "beyond", index - start


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from nova.jax.config import configure_dtypes

    configure_dtypes()

    import jax
    import jax.numpy as jnp
    import nova

    assert jax.config.jax_enable_x64 is True, "x64 is required for this measurement"

    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture

    revision = os.environ.get("NOVA_REVISION", "unstamped")
    print(
        f"# revision={revision} nova.__file__={nova.__file__} cwd={Path.cwd()}",
        flush=True,
    )
    print(
        f"# jax_platforms={os.environ.get('JAX_PLATFORMS')} "
        f"backend={jax.default_backend()} x64={jax.config.jax_enable_x64}",
        flush=True,
    )

    case_name = "diverted-single-null"
    requested_cells = -300
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, fixture_exterior, _cache = oracle_fixture.cached_fixture_exterior(
        source_case, exact, machine, empty_operator, oracle_state
    )
    operator = oracle_fixture.forward_operator(source_case, machine, fixture_exterior)
    from nova.equilibrium.stencil_mesh import StencilMesh

    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    from nova.equilibrium.forward import ForwardProfile

    profile = ForwardProfile(operator, mesh, newton_steps=25)
    target_current, current_centroid, current_receipt = (
        certificate._closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
    )
    seed, requested_class, seed_receipt = certificate._production_seed(
        profile, case_name, target_current, current_centroid, current_receipt
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{case_name}:{requested_cells}",
    )
    active = profile._with_source(request.source_profile, clip_mode=request.clip_mode)
    initial_flux = request.seed_policy.resolve(active, current=request.current)
    external = active.operator.external(request.current, request.prescribed_current)
    target_value = (
        None if request.target_current is None else jnp.asarray(request.target_current)
    )
    op = active.operator
    sizes = {
        "grid": int(len(machine.node)),
        "wall": int(len(machine.wall_node)),
        "sample": int(len(machine.sample_coordinates)),
    }
    print(
        f"NODE_SIZES grid={sizes['grid']} wall={sizes['wall']} "
        f"sample={sizes['sample']} total={sum(sizes.values())} "
        f"state={int(initial_flux.size)}",
        flush=True,
    )

    def describe(tag: str, values: np.ndarray) -> dict[str, object]:
        values = np.asarray(values, dtype=np.float64)
        bad = nonfinite_indices(values)
        nan_count = int(np.count_nonzero(np.isnan(values)))
        inf_count = int(np.count_nonzero(np.isinf(values)))
        print(
            f"{tag}: size={values.size} nonfinite={len(bad)} nan={nan_count} "
            f"inf={inf_count}",
            flush=True,
        )
        record: dict[str, object] = {
            "size": int(values.size),
            "nonfinite_count": int(len(bad)),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "first_indices": [int(i) for i in bad[:16]],
            "first_blocks": [],
            "first_coordinates": [],
        }
        for index in bad[:8]:
            block, local = block_of(int(index), sizes)
            if block == "grid":
                coordinate = np.asarray(machine.node)[local]
            elif block == "wall":
                coordinate = np.asarray(machine.wall_node)[local]
            else:
                coordinate = np.asarray(machine.sample_coordinates)[local]
            record["first_blocks"].append([block, int(local)])
            record["first_coordinates"].append(
                [float(v) for v in np.asarray(coordinate, dtype=float).ravel()]
            )
            print(
                f"    {tag}[{index}] block={block} local={local} "
                f"coord={np.asarray(coordinate).ravel()[:2]} "
                f"value={values[index]!r}",
                flush=True,
            )
        return record

    report: dict[str, object] = {
        "revision": revision,
        "backend": jax.default_backend(),
        "node_sizes": sizes,
        "target_current_a": float(target_current),
        "seed_nonfinite": int(np.count_nonzero(~np.isfinite(np.asarray(initial_flux)))),
    }

    # --- the map itself at the seed ----------------------------------------
    mapped_fn = op.traced_flux_map(None, target_value)

    def map_fn(state):
        return mapped_fn(state, external, op, target_value)

    mapped_state, tangent = jax.linearize(map_fn, initial_flux)
    residual_vector = mapped_state - initial_flux
    report["map"] = describe("map(mapped_state)", mapped_state)
    report["residual_vector"] = describe("residual_vector", residual_vector)

    probe_scale = float(np.max(np.abs(np.asarray(residual_vector))))
    probe = residual_vector / jnp.asarray(probe_scale, dtype=residual_vector.dtype)
    report["probe_scale"] = probe_scale
    report["probe"] = describe("probe", probe)

    tangent_probe = tangent(probe)
    report["tangent_probe"] = describe("tangent(probe)", tangent_probe)
    probe_action = probe - tangent_probe
    report["probe_action"] = describe("probe_action", probe_action)
    report["qualification"] = (
        "NONFINITE_LINEAR_ACTION"
        if np.any(~np.isfinite(np.asarray(probe_action)))
        else "ACCEPTED"
    )
    print(f"QUALIFICATION={report['qualification']}", flush=True)

    # --- decompose the map into its stages --------------------------------
    stages: dict[str, object] = {}
    tangent_cm = None

    def cell_moments(state):
        return op.normalised_current_moments(state, target_value, None)

    try:
        value, tangent_cm = jax.linearize(cell_moments, initial_flux)
        moments, amplitude = value
        tangent_moments, tangent_amplitude = tangent_cm(probe)
        stages["normalised_current_moments"] = describe(
            "normalised_current_moments|J*probe", tangent_moments.cell_current
        )
        stages["normalised_current_moments.amplitude"] = describe(
            "normalisation_amplitude|J*probe", tangent_amplitude
        )
        stages["normalised_current_moments.value_amplitude"] = describe(
            "normalisation_amplitude|value", amplitude
        )
    except Exception as error:  # noqa: BLE001 - diagnostic
        stages["normalised_current_moments"] = {"error": repr(error)}
        print(f"STAGE_FAILED normalised_current_moments {error!r}", flush=True)

    if tangent_cm is not None:
        try:
            base_moments = cell_moments(initial_flux)
            value, tangent_img = jax.linearize(
                lambda moments: op.current_moment_image(moments), base_moments
            )
            stages["current_moment_image"] = describe(
                "current_moment_image|J*probe_moments",
                tangent_img(tangent_cm(probe)[0]),
            )
        except Exception as error:  # noqa: BLE001
            stages["current_moment_image"] = {"error": repr(error)}
            print(f"STAGE_FAILED current_moment_image {error!r}", flush=True)

    try:
        value, tangent_x = jax.linearize(
            lambda state: op._exclude_shadow_residual(
                state, external + op.internal(state, None, target_value), None, None
            ),
            initial_flux,
        )
        stages["residual"] = describe("residual_stage|J*probe", tangent_x(probe))
    except Exception as error:  # noqa: BLE001
        stages["residual"] = {"error": repr(error)}
        print(f"STAGE_FAILED residual {error!r}", flush=True)

    report["stages"] = stages
    failed = sorted(name for name, entry in stages.items() if "error" in entry)
    report["failed_stages"] = failed

    np.savez(
        out / "krylov_action.npz",
        initial_flux=np.asarray(initial_flux),
        mapped_state=np.asarray(mapped_state),
        residual_vector=np.asarray(residual_vector),
        probe=np.asarray(probe),
        tangent_probe=np.asarray(tangent_probe),
        probe_action=np.asarray(probe_action),
    )
    (out / "krylov_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    status = 1 if failed else 0
    for name in failed:
        print(f"STAGE_FAILED_STAGE {name}", flush=True)
    print(f"EXIT={status}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
