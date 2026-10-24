"""Separate the raw cell-current derivative from the normalisation divide.

The first Krylov action of the diverted 300-cell CPU solve is NaN in all 340
cell current moments and in the single normalisation amplitude.  This script
linearises the two stages separately at the same seed so the report can say
which one introduces the NaN.

Usage: trace_normalisation.py <out-dir>
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


def census(values) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(values))
    return {
        "size": int(values.size),
        "nonfinite": int(len(bad)),
        "nan": int(np.count_nonzero(np.isnan(values))),
        "inf": int(np.count_nonzero(np.isinf(values))),
        "first_bad": [int(i) for i in bad[:8]],
        "finite_max_abs": float(
            np.max(np.abs(values[np.isfinite(values)])) if bad.size < values.size else 0.0
        ),
    }


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from nova.jax.config import configure_dtypes

    configure_dtypes()

    import jax
    import jax.numpy as jnp

    assert jax.config.jax_enable_x64 is True

    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture

    case_name = "diverted-single-null"
    requested_cells = -300
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(
        case_name, carrier_case, exact, requested_cells
    )
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

    profile = ForwardProfile(operator, mesh)
    target_current, current_centroid, current_receipt = (
        certificate._closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
    )
    seed, _requested_class, _seed_receipt = certificate._production_seed(
        profile, case_name, target_current, current_centroid, current_receipt
    )
    state = jnp.asarray(seed)
    target = jnp.asarray(target_current)

    print(
        f"# revision={os.environ.get('NOVA_REVISION')} backend={jax.default_backend()} "
        f"x64={jax.config.jax_enable_x64} target={float(target_current):.17g}",
        flush=True,
    )

    report: dict[str, object] = {"target_current_a": float(target_current)}

    # The probe the solver builds from the residual at the seed.
    op = operator
    external = op.external(None, None)
    mapped_fn = op.traced_flux_map(None, target)

    def map_fn(value):
        return mapped_fn(value, external, op, target)

    mapped_state, tangent = jax.linearize(map_fn, state)
    residual = mapped_state - state
    scale = float(np.max(np.abs(np.asarray(residual))))
    probe = residual / jnp.asarray(scale, dtype=residual.dtype)
    report["probe_scale"] = scale

    def cell_moments(value):
        return op.cell_current_moments(value, None)

    raw_value, raw_tangent = jax.linearize(cell_moments, state)
    report["raw_moments_value"] = census(raw_value.cell_current)
    report["raw_moments_derivative"] = census(raw_tangent(probe).cell_current)

    def amplitude(scale_vector):
        moments = op.cell_current_moments(scale_vector, None)
        return op.current_normalisation_amplitude(
            target, jnp.sum(moments.cell_current)
        )

    amp_value, amp_tangent = jax.linearize(amplitude, state)
    report["amplitude_value"] = census(amp_value)
    report["amplitude_derivative"] = census(amp_tangent(probe))

    def total_current(value):
        return jnp.sum(op.cell_current_moments(value, None).cell_current)

    total_value, total_tangent = jax.linearize(total_current, state)
    report["total_current_value"] = census(total_value)
    report["total_current_derivative"] = census(total_tangent(probe))

    normalised_value, normalised_tangent = jax.linearize(
        lambda value: op.normalised_current_moments(value, target, None), state
    )
    moments, amplification = normalised_value
    tang_moments, tang_amplification = normalised_tangent(probe)
    report["normalised_moments_value"] = census(moments.cell_current)
    report["normalised_moments_derivative"] = census(tang_moments.cell_current)
    report["normalised_amplitude_value"] = census(amplification)
    report["normalised_amplitude_derivative"] = census(tang_amplification)

    for key, value in report.items():
        print(f"{key}: {value}", flush=True)

    (out / "normalisation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print("EXIT=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())