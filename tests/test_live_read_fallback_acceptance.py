"""Acceptance coverage for the Newton route's live-read recovery."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.equilibrium import fixed_point
    from nova.equilibrium import forward
    from nova.equilibrium.forward import ForwardProfile


def test_live_read_recovery_keeps_own_mask_acceptance_callbacks(monkeypatch):
    calls = []

    def shadowed_map():
        def mapped(state, _shadow, *_arguments):
            return state + 1.0

        mapped._read_frozen_partition = lambda *_arguments: None
        mapped._map_frozen_partition = lambda state, _partition, *_arguments: (
            state + 2.0
        )
        mapped._frozen_partition_shadow = lambda partition: partition
        return mapped

    def controlled_newton(map_fn, initial, **options):
        if options["own_mask_acceptance"] and (
            options.get("promoted_shadow_mask_fn") is None
            or options.get("shadowed_map_fn") is None
        ):
            raise ValueError("own-mask acceptance requires promoted shadow masks")
        fallback_image = None
        if calls:
            shadowed = options["shadowed_map_fn"]
            arguments = options["map_arguments"]
            if hasattr(shadowed, "_map_frozen_partition"):
                fallback_image = shadowed._map_frozen_partition(
                    initial, None, *arguments
                )
            else:
                fallback_image = shadowed(initial, jnp.zeros_like(initial), *arguments)
        calls.append((options, fallback_image))
        return fixed_point.FixedPointResult(
            state=map_fn(initial, *options["map_arguments"]),
            residual=jnp.asarray(0.0),
            trace=jnp.asarray([0.0]),
            converged=jnp.asarray(len(calls) == 2),
            live_partition_reads=1,
        )

    operator = SimpleNamespace(
        clip_mode=None,
        external=lambda *_arguments: jnp.zeros(1),
        traced_flux_map=lambda *_arguments: lambda state, *_call_arguments: state,
        traced_flux_map_with_shadow=lambda *_arguments: shadowed_map(),
    )
    profile = SimpleNamespace(
        newton_steps=1,
        operator=operator,
        _accelerated_program_cache={},
    )
    profile._accelerated_history_program = MethodType(
        ForwardProfile._accelerated_history_program, profile
    )
    profile._receipt = lambda state, history, *_arguments: SimpleNamespace(
        flux=state, fixed_point=history
    )
    monkeypatch.setattr(forward.jax, "jit", lambda solve: solve)
    monkeypatch.setattr(fixed_point, "newton_krylov", controlled_newton)

    receipt = ForwardProfile._solve_accelerated(
        profile,
        "newton_krylov",
        jnp.zeros(1),
        None,
        own_mask_acceptance=True,
    )

    assert len(calls) == 2
    assert receipt is not None
    assert bool(np.asarray(receipt.fixed_point.converged))
    assert int(np.asarray(receipt.fixed_point.live_partition_reads)) == 1
    fallback, fallback_image = calls[-1]
    assert fallback["shadow_mask_fn"] is not None
    assert fallback["promoted_shadow_mask_fn"] is not None
    assert fallback["shadowed_map_fn"] is not None
    np.testing.assert_array_equal(fallback_image, [1.0])
