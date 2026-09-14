"""Lower exterior-variable forward programs without compiling or executing."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from nova.equilibrium import reduced_newton
from nova.equilibrium.forward_operator import (
    PrescribedCurrentField,
    set_support_clip_mode,
    support_clip_mode,
)
from nova.jax.config import configure_dtypes
from tests.test_forward_compile_identity import (
    CASES,
    _certificate_row,
    _digest,
    _lower_certificate_solve,
)


OUTPUT = Path(__file__).with_name("same-mesh-lowering.json")


def _compiled_slice_digests(row) -> tuple[str, str]:
    """Return StableHLO identities for two prescribed-current vectors."""
    profile, seed, requested_class, target_current, _request = row
    operator = profile.operator
    response = jnp.asarray(operator.external())[:, None]
    operator.prescribed_field = PrescribedCurrentField(
        response=response,
        current=jnp.asarray([1.0], dtype=response.dtype),
    )
    first_current = jnp.asarray([1.0], dtype=response.dtype)
    second_current = jnp.asarray([0.8], dtype=response.dtype)
    first_external = operator.external(prescribed_current=first_current)
    second_external = operator.external(prescribed_current=second_current)
    program, raw_kernels = reduced_newton._compiled_program(
        operator,
        seed,
        requested_class=requested_class,
        target_current=jnp.asarray(target_current),
        external=first_external,
        program=None,
    )
    kernels = reduced_newton._bind_dynamic_arguments(
        raw_kernels,
        first_external,
        jnp.asarray(target_current),
        requested_class,
        bind_external=False,
    )
    solver = reduced_newton._compiled_slice_solver(
        kernels,
        tolerance=1.0e-8,
        newton_steps=1,
        active_set_steps=1,
    )
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(seed, requested_class), dtype=bool)
    )
    print("STAGE compiled-slice lower first prescribed current", flush=True)
    first = solver.lower(seed, shadow, first_external)
    print("STAGE compiled-slice lower second prescribed current", flush=True)
    second = solver.lower(seed, shadow, second_external)
    assert program.external_shape == first_external.shape
    return _digest(first), _digest(second)


def main() -> int:
    print("STAGE configure CPU extended precision", flush=True)
    configure_dtypes()
    set_support_clip_mode("chord")
    assert support_clip_mode() == "chord"

    print("STAGE load weak cached 300-cell certificate row", flush=True)
    weak = _certificate_row(CASES[0])
    fixture_external = weak[0].operator.external()
    scaled_external = fixture_external * jnp.asarray(0.9, dtype=fixture_external.dtype)

    print("STAGE certificate lower fixture exterior", flush=True)
    fixture_lowered, _ = _lower_certificate_solve(weak, fixture_external)
    print("STAGE certificate lower scaled exterior", flush=True)
    scaled_lowered, _ = _lower_certificate_solve(weak, scaled_external)
    certificate_digests = (_digest(fixture_lowered), _digest(scaled_lowered))

    print("STAGE compiled-slice lowering", flush=True)
    compiled_digests = _compiled_slice_digests(weak)
    receipt = {
        "clip_mode": support_clip_mode(),
        "requested_cells": -300,
        "external_shape": list(fixture_external.shape),
        "exterior_scale": 0.9,
        "certificate": {
            "fixture_sha256": certificate_digests[0],
            "scaled_sha256": certificate_digests[1],
            "identical": certificate_digests[0] == certificate_digests[1],
        },
        "compiled_slice": {
            "first_prescribed_current_sha256": compiled_digests[0],
            "second_prescribed_current_sha256": compiled_digests[1],
            "identical": compiled_digests[0] == compiled_digests[1],
        },
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"STAGE receipt written {OUTPUT}", flush=True)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return (
        0
        if all(
            section["identical"]
            for section in (receipt["certificate"], receipt["compiled_slice"])
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
