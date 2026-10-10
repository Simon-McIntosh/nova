"""The traced forward Grad-Shafranov map."""

from __future__ import annotations

import jax


def apply(
    psi: jax.Array,
    external: jax.Array,
    operator,
    requested_class=None,
    target_current=None,
    shadow=None,
    partition=None,
) -> jax.Array:
    """Image booked plasma current and exterior flux using operand geometry."""
    internal = (
        operator.internal(psi, requested_class, target_current)
        if partition is None
        else operator._internal_on_partition(psi, partition, target_current)
    )
    image = external + internal
    return operator._exclude_shadow_residual(psi, image, requested_class, shadow=shadow)
