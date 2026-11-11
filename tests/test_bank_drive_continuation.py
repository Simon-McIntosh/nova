"""Drive continuation reaches a drive change a direct bank solve cannot.

The fixture is a scalar miniature of the measured stall rather than a mock of
the solver: one state, one drive, and a residual whose Newton basin is limited,
``r(x, d) = t + C t**3`` with ``t = x - d``. The step direction is exact and
the Jacobian is the true derivative, so the fixture has the property the H200
record shows — the linearisation is right at the incumbent and the trial it
predicts is wrong far from the root — without any code deciding when a solve
should fail. The line search is the product's own
:func:`nova.equilibrium.solve._damped_scale`, unchanged.

Only ``support_read.solve_batch`` is replaced, which is the seam this driver is
written above; the Newton step, the line search and the terminal fresh read it
drives are the product's.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

from nova.equilibrium import drive_continuation  # noqa: E402
from nova.equilibrium.solve import _damped_scale  # noqa: E402

pytestmark = pytest.mark.timeout(600)

#: Stiffness of the two lanes. The cubic term sets how far from the root the
#: linear model still holds, which is the basin the driver has to subdivide.
STIFFNESS = jnp.asarray([1.0e6, 0.9e6])

START_DRIVE = 1.0
TARGET_DRIVE = 1.01

#: Every read the fixture performs, as (drive, per-lane residual, tolerance).
READS: list[tuple[float, list[float], float]] = []


class _Profile(NamedTuple):
    """The drive a sub-solve is asked for, and the fixture's stiffness."""

    drive: float
    stiffness: object


class _Stationary(NamedTuple):
    position: object


class _Support(NamedTuple):
    stationary: _Stationary


class _ToyResult(NamedTuple):
    """What :func:`drive_continuation.advance_drive` reads off a solve result."""

    moments: tuple
    support: _Support
    residual: object
    converged: object
    steps: object
    reads: object


def _residual(state, drive, stiffness):
    offset = state - drive
    return offset + stiffness * offset**3


def _toy_solve_batch(
    operands,
    cache,
    moments,
    seeds,
    previous,
    convention,
    policy,
    profile,
    density_gradient,
    quadrature,
    axis_sign=-1.0,
    *,
    max_steps=60,
    tolerance=1e-10,
    **_options,
):
    """A staged solve with a seed read, Newton updates and a terminal read."""
    drive = profile.drive
    lanes = moments[0].shape[0]
    state = moments[0][:, 0]

    def fresh(candidate):
        values = jnp.abs(_residual(candidate, drive, profile.stiffness))
        jax.debug.callback(
            lambda lane: READS.append(
                (float(drive), [float(value) for value in lane], tolerance)
            ),
            values,
        )
        return values

    norm = fresh(state)
    reads = 1
    accepted = norm <= tolerance
    steps = jnp.zeros(lanes, jnp.int32)
    while bool(jnp.any(~accepted)) and int(jnp.max(steps)) < max_steps:
        active = ~accepted
        onset = state - drive
        correction = -_residual(state, drive, profile.stiffness) / (
            1.0 + 3.0 * profile.stiffness * onset**2
        )
        search = _damped_scale(
            lambda trial: (fresh(trial), jnp.ones(lanes, bool), trial),
            state,
            correction,
            norm,
            state,
            active=active,
            max_steps=8,
            tolerance=tolerance,
            keep_incumbent=False,
        )
        state = search.state
        norm = search.residual
        reads += int(search.evaluations)
        steps = steps + active.astype(jnp.int32)
        accepted = accepted | (norm <= tolerance)
    return _ToyResult(
        (state[:, None],),
        _Support(_Stationary(jnp.zeros((lanes, 1, 2)))),
        norm,
        accepted,
        steps,
        jnp.full(lanes, reads, jnp.int32),
    )


@pytest.fixture(name="toy")
def fixture_toy():
    """Substitute the solve seam and hand back a caller."""
    READS.clear()

    def run(**overrides):
        state = jnp.asarray([[START_DRIVE], [START_DRIVE]])
        arguments = dict(
            start_drive=START_DRIVE,
            target_drive=TARGET_DRIVE,
            profile_at=lambda drive: _Profile(drive, STIFFNESS),
            solve=_toy_solve_batch,
        )
        arguments.update(overrides)
        return drive_continuation.advance_drive(
            None,
            None,
            (state,),
            jnp.zeros((2, 1, 2)),
            jnp.zeros((2, 1, 2)),
            None,
            None,
            arguments.pop("profile_at"),
            None,
            None,
            **arguments,
        )

    return run


def test_direct_solve_fails_where_continuation_certifies(toy):
    """The whole drive change in one solve misses its budget; sub-steps do not."""
    direct = _toy_solve_batch(
        None,
        None,
        (jnp.asarray([[START_DRIVE], [START_DRIVE]]),),
        jnp.zeros((2, 1, 2)),
        jnp.zeros((2, 1, 2)),
        None,
        None,
        _Profile(TARGET_DRIVE, STIFFNESS),
        None,
        None,
        max_steps=drive_continuation.UPDATE_BUDGET,
        tolerance=drive_continuation.CERTIFICATION_TOLERANCE,
    )
    print("DIRECT", int(jnp.max(direct.steps)), float(jnp.max(direct.residual)))
    assert not bool(jnp.all(direct.converged))

    reached = toy()
    print(
        "CONTINUATION",
        reached.certified,
        len(reached.substeps),
        reached.updates,
        reached.reads,
        float(reached.result.residual.max()),
    )
    assert reached.certified
    assert reached.drive == TARGET_DRIVE
    assert float(jnp.max(reached.result.residual)) <= (
        drive_continuation.CERTIFICATION_TOLERANCE
    )
    assert reached.updates <= int(jnp.max(direct.steps))


def test_sub_step_shrinks_on_failure_and_grows_on_success(toy):
    """The accepted sub-step doubles after success and halves after a miss."""
    reached = toy(initial_relative_step=0.01)
    print(
        "SUBSTEPS",
        [
            (round(substep.drive, 6), substep.change, substep.converged)
            for substep in reached.substeps
        ],
    )
    drive = START_DRIVE
    growths = shifts = 0
    for current, following in zip(reached.substeps, reached.substeps[1:], strict=False):
        if current.converged:
            drive = current.drive
        remaining = abs(TARGET_DRIVE - drive)
        if not current.converged:
            shifts += 1
            assert following.sub_step == pytest.approx(
                current.sub_step * drive_continuation.SHRINK
            )
        else:
            if current.final:
                continue
            growths += 1
            assert following.sub_step == pytest.approx(
                min(
                    current.sub_step * drive_continuation.GROWTH,
                    abs(START_DRIVE),
                )
            )
        assert following.change == pytest.approx(min(following.sub_step, remaining))
    assert growths, "no sub-step was ever accepted short of the target"
    assert shifts, "no sub-step ever missed its budget"
    first_rejection = next(
        i for i, substep in enumerate(reached.substeps) if not substep.converged
    )
    accepted_changes = [
        substep.change
        for substep in reached.substeps[first_rejection + 1 :]
        if substep.converged
    ]
    assert any(
        following > current
        for current, following in zip(
            accepted_changes, accepted_changes[1:], strict=False
        )
    )
    assert reached.substeps[0].change == pytest.approx(0.01 * START_DRIVE)


def test_final_certificate_is_a_full_fresh_read_at_the_target(toy):
    """The certificate is the read made at the target drive, not an earlier one."""
    reached = toy()
    final = reached.substeps[-1]
    at_target = [row for row in READS if row[0] == TARGET_DRIVE]
    assert final.final
    assert final.tolerance == drive_continuation.CERTIFICATION_TOLERANCE
    assert at_target, "the driver never read the state at the target drive"
    drive, lanes, tolerance = at_target[-1]
    print("CERTIFICATE", drive, lanes, tolerance)
    assert tolerance == drive_continuation.CERTIFICATION_TOLERANCE
    assert max(lanes) <= drive_continuation.CERTIFICATION_TOLERANCE
    assert max(lanes) == pytest.approx(final.residual)
    assert final.residual < drive_continuation.INTERMEDIATE_TOLERANCE


def test_linear_stream_predicts_next_state_to_rounding():
    previous = (jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64),)
    slope = jnp.asarray([[0.25, -0.5], [1.0, 2.0]], dtype=jnp.float64)
    current = (previous[0] + 2.0 * slope,)
    predicted = drive_continuation.predict_seed(
        current,
        3.0,
        4.0,
        previous_moments=previous,
        previous_drive=1.0,
    )
    assert jnp.allclose(predicted[0], previous[0] + 3.0 * slope, rtol=0, atol=1e-15)


@pytest.mark.parametrize("history", [None, (jnp.ones((2, 1)),)])
def test_predictor_without_two_distinct_slices_keeps_warm_start(history):
    moments = (jnp.asarray([[1.0], [2.0]]),)
    predicted = drive_continuation.predict_seed(
        moments,
        1.0,
        1.001,
        previous_moments=history,
        previous_drive=None if history is None else 1.0,
    )
    assert jnp.array_equal(predicted[0], moments[0])


def test_predicted_seed_still_requires_terminal_fresh_certificate(toy):
    reached = toy(
        previous_moments=(jnp.asarray([[0.999], [0.999]]),),
        previous_drive=0.999,
    )
    assert reached.certified
    at_target = [row for row in READS if row[0] == TARGET_DRIVE]
    assert at_target
    assert at_target[-1][2] == 1e-10
    assert max(at_target[-1][1]) <= 1e-10
    assert reached.substeps[-1].residual == pytest.approx(max(at_target[-1][1]))


def test_predictor_does_not_certify_a_false_convergence_flag(toy):
    def dishonest(*args, **kwargs):
        result = _toy_solve_batch(*args, **kwargs)
        return result._replace(
            residual=jnp.full((2,), 1e-5), converged=jnp.ones(2, bool)
        )

    reached = toy(solve=dishonest, max_substeps=1)
    assert not reached.certified


def test_solve_receives_predicted_seed_and_accepted_factor(toy):
    from types import SimpleNamespace

    observed = []
    factors = [object(), object()]

    def solve(*args, **kwargs):
        observed.append((args[2], kwargs.get("factor_cache")))
        result = _toy_solve_batch(*args, **kwargs)
        return SimpleNamespace(
            **result._asdict(),
            factor=factors[len(observed) - 1],
            factor_valid=jnp.ones(2, bool),
        )

    reached = toy(
        target_drive=1.002,
        initial_relative_step=0.001,
        solve=solve,
        solve_options={"reuse_factor": True},
        previous_moments=(jnp.full((2, 1), 0.999),),
        previous_drive=0.999,
    )
    assert reached.certified and len(observed) == 2
    assert jnp.allclose(observed[0][0][0], 1.001, rtol=0, atol=1e-15)
    assert jnp.allclose(observed[1][0][0], 1.002, rtol=0, atol=1e-15)
    assert observed[0][1] is None
    assert observed[1][1] is factors[0]


def test_factor_reuse_is_disabled_by_default(toy):
    from types import SimpleNamespace

    observed = []
    factor = object()

    def solve(*args, **kwargs):
        observed.append(kwargs)
        result = _toy_solve_batch(*args, **kwargs)
        return SimpleNamespace(
            **result._asdict(), factor=factor, factor_valid=jnp.ones(2, bool)
        )

    reached = toy(
        target_drive=1.002,
        initial_relative_step=0.001,
        solve=solve,
        factor_cache=factor,
        previous_moments=(jnp.full((2, 1), 0.999),),
        previous_drive=0.999,
    )
    assert reached.certified and len(observed) == 2
    assert all(not options.get("reuse_factor", False) for options in observed)
    assert all("factor_cache" not in options for options in observed)
    assert all("factor_valid" not in options for options in observed)
