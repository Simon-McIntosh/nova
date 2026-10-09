"""Broadcasting-correct Taylor rules for ``lax.min_p`` / ``lax.max_p``.

``jax.experimental.jet`` ships min/max rules that assume both operands already
share a shape: they compute ``lax.select(primal_mask, x, y)`` on the raw
primals. The analytic polygon kernel reaches ``min``/``max`` with a broadcast
scalar against an array, and the shipped rule then raises
``select cases must have the same shapes`` before the point jet is traced at
all. These replacements broadcast the primal operands and every Taylor term
pair before selecting, which is the mathematically identical rule applied in the
correct shape. The primal expression is unchanged, so the value the rule returns
is the same one ``lax.min`` returns.
"""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import jet


def _broadcast_select_rule(comparison):
    def rule(primals_in, series_in, **_):
        x, y = jnp.broadcast_arrays(*primals_in)
        mask = comparison(x, y)
        primal_out = lax.select(mask, x, y)

        def pick(x_term, y_term):
            x_term, y_term = jnp.broadcast_arrays(x_term, y_term)
            return lax.select(mask, x_term, y_term)

        series_out = [
            pick(*terms) for terms in zip(*series_in, strict=True)
        ]
        return primal_out, series_out

    return rule


def patch_broadcast_minmax() -> None:
    """Install broadcasting min/max rules on the process's jet rule table."""
    jet.jet_rules[lax.min_p] = _broadcast_select_rule(lambda x, y: x < y)
    jet.jet_rules[lax.max_p] = _broadcast_select_rule(lambda x, y: x > y)


__all__ = ["patch_broadcast_minmax"]


_ = jax  # keep the import meaningful for readers scanning the module