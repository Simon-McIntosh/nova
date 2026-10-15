"""Check that padding cannot erase an active mobile jet row."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from scripts.prototypes.read_robustness.census import axis_identity
from scripts.prototypes.read_robustness.window import mobile_window


def main():
    assert jax.config.jax_enable_x64 is True
    jets = SimpleNamespace(
        nodes=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        sample_index=jnp.array([[0, 1], [1, 2]]),
        rows=jnp.ones((4, 2, 3, 6)),
        exterior=jnp.zeros((4, 6)),
        mesh=None,
    )
    geometry = SimpleNamespace(
        centre=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        pitch=jnp.ones(2),
        neighbour=-jnp.ones((2, 6), dtype=jnp.int32),
        vertex_count=jnp.full(2, 6),
    )
    gather = jax.jit(
        lambda capacity: mobile_window(
            jets, geometry, jnp.ones(2), jnp.array([[0.0, 0.0]]), capacity
        ),
        static_argnames=("capacity",),
    )
    with_slack = gather(4)
    assert int(with_slack.required) == 3
    assert not bool(with_slack.overflow)
    assert (with_slack.cache.lookup[:3] >= 0).all()
    assert int(with_slack.cache.lookup[3]) == -1
    too_small = gather(2)
    assert bool(too_small.overflow)
    assert int(too_small.required) == 3
    expected_axis = jnp.array([0.0, 0.0])
    assert bool(axis_identity(expected_axis, expected_axis, 1.0))
    assert not bool(axis_identity(jnp.array([0.0, 3.0]), expected_axis, 1.0))
    print("WINDOW_CHECK required=3 padded_lookup=0,1,2 overflow_at=2 axis_control=red")


if __name__ == "__main__":
    main()
