"""The qualified Krylov step applies its operator at exactly one call site.

Every operator application of the step (the finite-action probe, the
condition Arnoldi columns, the restarted GMRES and the achieved-residual
check) is served from one slot of a loop, so the traced program carries the
operator once however many applications the solve needs. The operator is
traced under a named scope and its contraction is counted through every
nested jaxpr. The loop exits when the stream is done, and under ``vmap`` when
every member is, so a batch applies the operator no more often than its
slowest member needs.
"""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_point
from nova.jax.config import configure_dtypes

configure_dtypes()

OPERATOR_SCOPE = "krylov_operator_under_test"


def _operator_sites(jaxpr) -> int:
    count = 0
    for equation in jaxpr.eqns:
        if equation.primitive.name == "dot_general" and OPERATOR_SCOPE in str(
            equation.source_info.name_stack
        ):
            count += 1
        for parameter in equation.params.values():
            for value in (
                parameter if isinstance(parameter, tuple | list) else (parameter,)
            ):
                inner = getattr(value, "jaxpr", value)
                if hasattr(inner, "eqns"):
                    count += _operator_sites(inner)
    return count


def _qualified_step_program(size=12, iterations=8):
    rng = np.random.default_rng(3)
    matrix = jnp.asarray(np.eye(size) + 0.2 * rng.standard_normal((size, size)))
    rhs = jnp.asarray(rng.standard_normal(size))

    def operator(vector):
        with jax.named_scope(OPERATOR_SCOPE):
            return matrix @ vector

    def step():
        return fixed_point._qualified_krylov_step(
            operator,
            rhs,
            jnp.asarray(0.1),
            gmres_iterations=iterations,
            condition_ratio_limit=10.0,
            preceding_condition_baseline=jnp.asarray(2.0),
        )

    return step, matrix, rhs


def test_the_qualified_step_applies_its_operator_at_one_site():
    assert jax.config.jax_enable_x64 is True
    step, _, _ = _qualified_step_program()
    closed = jax.make_jaxpr(step)()
    assert _operator_sites(closed.jaxpr) == 1


def test_the_single_site_step_still_solves_a_full_dimension_system():
    step, matrix, rhs = _qualified_step_program(size=6, iterations=6)
    result = jax.jit(step)()
    np.testing.assert_allclose(
        np.asarray(matrix @ result.unconditioned_step), np.asarray(rhs), atol=1e-10
    )
    assert float(result.achieved_reduction) < 1e-10


def _counted_member_step(iterations=3, size=6):
    """A vmappable step whose operator reports each application to the host."""
    rng = np.random.default_rng(3)
    coupling = jnp.asarray(rng.standard_normal((size, size)))
    applications = []

    def step(scale, rhs):
        def operator(vector):
            jax.debug.callback(lambda _: applications.append(1), vector)
            return vector + scale * (coupling @ vector)

        return fixed_point._qualified_krylov_step(
            operator,
            rhs,
            jnp.asarray(0.1),
            gmres_iterations=iterations,
            condition_ratio_limit=10.0,
            preceding_condition_baseline=jnp.asarray(2.0),
        ).step

    return step, applications, rng.standard_normal((2, size))


def test_the_vmapped_stream_stops_when_every_member_is_done():
    iterations = 3
    capacity = 1 + iterations + 1 + iterations * (iterations + 1) + 1
    step, applications, rhs = _counted_member_step(iterations)
    # an identity member resolves in its first restart, a weakly coupled one
    # needs two; neither reaches the stream's worst-case slot count
    scales = jnp.asarray([0.0, 0.02])
    rhs = jnp.asarray(rhs)
    single = []
    for scale, member_rhs in zip(scales, rhs, strict=True):
        applications.clear()
        jax.jit(step)(scale, member_rhs).block_until_ready()
        single.append(len(applications))
    assert single[0] < single[1] < capacity
    applications.clear()
    batched = jax.jit(jax.vmap(step))(scales, rhs).block_until_ready()
    assert len(applications) == len(scales) * max(single)
    # a finished member's carry is held while the other runs on; batched and
    # unbatched arithmetic round differently, so the match is to rounding
    for index, scale in enumerate(scales):
        np.testing.assert_allclose(
            np.asarray(batched[index]),
            np.asarray(jax.jit(step)(scale, rhs[index])),
            rtol=1e-12,
            atol=1e-15,
        )


def _loop_holding_operator(jaxpr):
    """The innermost while loop whose body applies the operator under test."""
    for equation in jaxpr.eqns:
        for parameter in equation.params.values():
            for value in (
                parameter if isinstance(parameter, tuple | list) else (parameter,)
            ):
                inner = getattr(value, "jaxpr", value)
                if not hasattr(inner, "eqns"):
                    continue
                nested = _loop_holding_operator(inner)
                if nested is not None:
                    return nested
                if equation.primitive.name == "while" and _operator_sites(inner):
                    return inner
    return None


def test_the_vmapped_stream_runs_one_consumer_per_shared_phase():
    """Under vmap the stream's own batching rule serves the batch.

    Members sit on an explicit axis, the loop exits on a scalar any-pending
    predicate, and each slot dispatches through one conditional over the
    member-shared phase (one branch per consumer, one pass-through, one
    per-member fallback), rather than the loop's default rule selecting the
    whole carry against every member's exit.
    """
    step, _, rhs = _qualified_step_program()
    rhs = jnp.stack([rhs, 2.0 * rhs])
    matrix = jnp.asarray(
        np.eye(12) + 0.2 * np.random.default_rng(3).standard_normal((12, 12))
    )

    def member(b):
        def operator(vector):
            with jax.named_scope(OPERATOR_SCOPE):
                return matrix @ vector

        return fixed_point._qualified_krylov_step(
            operator,
            b,
            jnp.asarray(0.1),
            gmres_iterations=8,
            condition_ratio_limit=10.0,
            preceding_condition_baseline=jnp.asarray(2.0),
        ).step

    body = _loop_holding_operator(jax.make_jaxpr(jax.vmap(member))(rhs).jaxpr)
    assert body is not None
    dispatch = [
        equation
        for equation in body.eqns
        if equation.primitive.name == "cond"
        and len(equation.params["branches"]) == len(fixed_point._KrylovPhase) + 1
    ]
    assert len(dispatch) == 1
    assert dispatch[0].invars[0].aval.shape == ()
