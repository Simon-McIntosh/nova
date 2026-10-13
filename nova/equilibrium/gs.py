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
    moments=None,
) -> jax.Array:
    """Image booked plasma current and exterior flux using operand geometry.

    ``moments`` are physical cell-current moments already booked on a support
    read. They are imaged directly, so the operator reads no support of its
    own and ``shadow``, the residual exclusion, must be supplied with them.
    """
    if moments is not None:
        if shadow is None:
            raise ValueError("booked moments require the residual shadow of their read")
        image = external + operator.current_moment_image(
            operator.coupling_current_moments(moments)
        )
        return operator._exclude_shadow_residual(psi, image, requested_class, shadow)
    internal = (
        operator.internal(psi, requested_class, target_current)
        if partition is None
        else operator._internal_on_partition(psi, partition, target_current)
    )
    image = external + internal
    return operator._exclude_shadow_residual(psi, image, requested_class, shadow=shadow)
