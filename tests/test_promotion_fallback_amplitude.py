"""A frozen topology read may only be carried where it can scale a state.

The promotion fallback carries one topology read across the inner iterations
of a trip.  A read that admitted no finite axis leaves every normalising
scalar undefined, so revaluing a state against it turns the whole current
support into ``nan``: the cell-current sum the source amplitude divides by is
fabricated rather than measured, and the mask the active set records is the
one that read produced rather than the state's own.  These tests pin the
refusal that keeps such a read out of the carry, on both the predicate and the
solve that consumes it, and pin the un-carried rung so the guard which keeps
the carry off the target-current path cannot be removed silently either.
"""

from collections import Counter
from pathlib import Path
import shutil
import types

import jax
import jax.numpy as jnp
import pytest

from nova.equilibrium.fixed_point import newton_krylov
from nova.equilibrium.forward_operator import ForwardFluxOperator


def _frozen_read(axis_flux, boundary_flux, flux_span):
    return types.SimpleNamespace(
        topology=types.SimpleNamespace(
            axis_flux=jnp.asarray(axis_flux),
            boundary_flux=jnp.asarray(boundary_flux),
            flux_span=jnp.asarray(flux_span),
        )
    )


def _carried_solve(usable: bool) -> tuple[Counter, object]:
    """Solve a synthetic contraction with the carry either taken or refused.

    ``map_partition`` and ``shadowed_map`` are separate closures over the same
    map, so counting them says which one the solve actually evaluated rather
    than which one it was handed.
    """
    counts: Counter = Counter()

    def map_fn(state):
        return 0.5 * state

    def mask(_state):
        return jnp.zeros(1, dtype=bool)

    def shadowed_map(state, _mask):
        jax.debug.callback(lambda: counts.update(("plain",)), ordered=True)
        return map_fn(state)

    def read_partition(_state, _previous_shadow=None):
        return jnp.zeros(1)

    def map_partition(state, _partition):
        jax.debug.callback(lambda: counts.update(("carried",)), ordered=True)
        return map_fn(state)

    shadowed_map._read_frozen_partition = read_partition
    shadowed_map._map_frozen_partition = map_partition
    shadowed_map._frozen_partition_shadow = lambda _partition: jnp.zeros(1, dtype=bool)
    shadowed_map._frozen_partition_usable = lambda _partition: jnp.asarray(usable)

    solve = jax.jit(
        lambda: newton_krylov(
            map_fn,
            jnp.zeros(2),
            newton_steps=6,
            gmres_iterations=2,
            warmup=0,
            shadow_mask_fn=mask,
            promoted_shadow_mask_fn=lambda state, _previous: mask(state),
            shadowed_map_fn=shadowed_map,
            active_set_steps=2,
        )
    )
    try:
        result = solve()
        jax.block_until_ready(result.state)
    finally:
        jax.clear_caches()
    return counts, result


def test_a_finite_admitted_axis_stays_usable():
    """Positive control: a sound read must pass the refusal."""
    partition = _frozen_read(
        0.010040016591095605, -0.0038869410906113644, -0.013926957681706969
    )

    assert bool(ForwardFluxOperator.frozen_partition_usable(partition)) is True


def test_a_non_finite_axis_is_refused():
    """The diverted carrier's own read: axis and span both ``nan``."""
    partition = _frozen_read(jnp.nan, 0.004708612000369493, jnp.nan)

    assert bool(ForwardFluxOperator.frozen_partition_usable(partition)) is False


def test_a_vanishing_flux_span_is_refused():
    """A zero span divides every normalised flux by zero."""
    partition = _frozen_read(0.010040016591095605, -0.0038869410906113644, 0.0)

    assert bool(ForwardFluxOperator.frozen_partition_usable(partition)) is False


def test_a_usable_frozen_read_is_carried():
    """Positive control for the solve: a usable read reaches the carried map."""
    counts, _result = _carried_solve(True)

    assert counts["carried"] > 0
    assert counts["plain"] == 0


def test_an_unusable_frozen_read_is_refused():
    """The refused read never reaches the carried map, and the trip still runs."""
    counts, result = _carried_solve(False)

    assert counts["carried"] == 0
    assert counts["plain"] > 0
    assert bool(jnp.all(jnp.isfinite(result.state)))


@pytest.mark.slow
def test_the_diverted_rung_settles_on_its_un_carried_residual(monkeypatch):
    """The diverted rung settles where a fresh read puts it, not on the carry.

    Attaching the fallback carry to a target-normalised solve moves the
    terminal residual from ``2.248351582217136`` to ``6.3706933429737305``,
    because the carried discrete partition is not the one the state would
    read.  Pinning the un-carried value here is what makes the target-current
    guard load-bearing: removing it moves this number.

    The measurement renders its panels and receipts them against the served
    docs mount, so it writes under ``<root>/docs``; the scratch subtree is the
    test's own and is removed afterwards, because a test that leaves rendered
    panels behind dirties the tree for every peer reading it.
    """
    from benchmarks import solovev_certificate

    scratch = Path(solovev_certificate.ROOT) / "docs" / "figures" / "_fallback_scratch"
    try:
        monkeypatch.setattr(solovev_certificate, "FIGURE_ROOT", scratch)
        monkeypatch.setattr(solovev_certificate, "DIAGNOSTIC_ROOT", scratch)
        monkeypatch.setattr(
            solovev_certificate, "PART_ROOT", scratch / "production-route-parts"
        )
        row = solovev_certificate._measure("diverted-single-null", -300)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    assert row["solver"]["terminal_fixed_point_residual"] == 2.248351582217136
    assert row["solver"]["qualification"] in {"qualified", "unqualified"}
