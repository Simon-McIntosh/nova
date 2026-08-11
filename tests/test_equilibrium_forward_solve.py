r"""Free-boundary solve contract of ``ForwardProfile`` on one bootstrapped machine.

The machine is assembled through Nova's canonical Green kernels so the solve
is a real free-boundary problem rather than an algebraic identity. A ring of
external conductors is fitted to hold an analytic Solov'ev seed, and the
shipped absolute source is an edge-vanishing pressure and diamagnetic
gradient driven above the marginal point, so the wall-limited branch is an
attractor of the fixed-point map.

That marginal point is itself a result worth stating. Removing the implicit
net-current rescale removes the mechanism that used to make the iteration
contract: with an absolute source the map has a second, trivial fixed point —
the vacuum field, where the plasma has left the material boundary and drives
no current — and a source driven only just hard enough to reach the limiter
sits on a fold between the two. The seed policy, not the accelerator, decides
which branch a solve lands on; both branches are pinned below.

How deep any route can drive the map is set by one read. The source is
normalised against the axis-to-boundary flux span, and the axis flux comes
from a single-precision sub-cell fit of the grid stencil, so it is only ever
known on a float32 ladder: below one step of that ladder the map cannot
resolve its own progress. Everything downstream — the residual, the plasma
current, the moments — inherits that step, so the pins that touch the
convergence floor are written against it rather than against a fixed depth.
Where a route stops inside the resulting band is decided by rounding far
beneath it and is a property of the arithmetic the host emits, not of the
equilibrium.

Pinned here: route parity between the host root find and the accelerated
ladder, the basin over multiple seeds, the batched ensemble solve under
``jit``/``vmap``, the differentiable moment map against finite differences,
the gradient of a converged functional with respect to a conductor current,
the current ledger's declared support, and the conservation receipts.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.domain import PlasmaDomain
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.observation import MomentEnforcementError, MomentTargets
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.jax.config import configure_dtypes

P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_FIELD_FUNCTION = 5.0
CONDUCTORS = 16
EVALUATIONS = 240
#: Evaluations the relaxed transient stays inside one domain labelling.
TRANSIENT = 30

#: Pre-registered solve tolerances. The fixed-point residual and the route
#: parities are numerical contracts; the two flux-function identities sit at
#: the fp64 differencing floor; the Grad-Shafranov and force residuals are
#: read with central differences on a map produced by Green operators, so
#: their floor is set by the coarser of the two discretisations rather than
#: by the solve — the analytic pin of the convention lives with the
#: manufactured equilibrium, not here.
RESIDUAL_TOLERANCE = 1.0e-6
PARITY_TOLERANCE = 1.0e-6
BASIN_TOLERANCE = 1.0e-4
DIVERGENCE_TOLERANCE = 1.0e-12
GRAD_SHAFRANOV_TOLERANCE = 0.1
FORCE_TOLERANCE = 0.1
MOMENT_JACOBIAN_TOLERANCE = 1.0e-2
FLUX_GRADIENT_TOLERANCE = 1.0e-3
CURRENT_GRADIENT_TOLERANCE = 1.0e-3
#: Relative conductor-current step the central differences are taken over.
#: The topology mask is a connectivity SELECTION and carries no gradient, so
#: the difference has to span enough cell-quantised boundary motion to average
#: it out; below about a thousandth of the conductor current that quantisation
#: dominates and the measured error grows as the step shrinks.
GRADIENT_STEP = 1.0e-3
#: Relative residual the registered floor sits at. One step of the axis-flux
#: ladder is 1.1e-7 of this machine's flux scale and the map turns a step of
#: the normalisation into a few times that in the residual, so the stalled
#: band runs to about five steps; the floor is registered a decade above one
#: step, clear of the band and still far under the transient it terminates.
QUANTISATION_FLOOR = 1.0e-6
#: Contraction ``TRANSIENT`` evaluations buy while the iterate still moves the
#: axis read, and the factor the same count of evaluations is allowed to move
#: the residual by once it no longer does. Separating the two is the statement
#: that the iteration has stalled: a contrast between two equal-length windows
#: of one run, which no single depth could express.
TRANSIENT_CONTRACTION = 1.0e3
STALL_CONTRACTION = 1.0e2
#: Ladder steps the stalled residual band is required to span. A step in the
#: axis flux shifts the whole normalised profile, and the current change that
#: produces re-enters the residual amplified by a factor of order unity to
#: ten; the band brackets that amplification over two decades rather than
#: predicting it, and separates the stall from both round-off and the
#: transient above it.
STALL_BAND = (0.5, 50.0)
#: Ladder steps the Krylov root find's absolute residual target is set at. A
#: target under one step asks for a flux difference the axis read cannot
#: express; eight clears the few steps one solve was measured to move it by.
ROOT_FIND_LADDER_STEPS = 8
#: Agreement between the host and traced relaxed steps over the transient.
#: Both accumulate the same fp64 arithmetic, but the traced loop is free to
#: reassociate it, so the two drift apart at the round-off amplification of
#: the map rather than staying bit identical.
IMPLEMENTATION_PARITY = 1.0e-9


def _terms():
    """Return the Solov'ev quartic, offset and vertical coefficients."""
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius, height):
    """Return the analytic seed flux [Wb] the conductors are fitted to."""
    alpha, offset, beta = _terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _wall_loop(points=61):
    """Return a material boundary lying on one seed flux surface."""
    alpha, offset, beta = _terms()
    wall_flux = _solovev(AXIS_RADIUS, 0.0) - SEED_SPAN
    inner, outer = np.sqrt(np.sort(np.roots([alpha, offset, -wall_flux])))
    centre, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
    angle = 2 * np.pi * np.arange(points) / points
    radius = centre + half * np.cos(angle)
    argument = np.clip((wall_flux - _solovev(radius, 0.0)) / beta, 0.0, None)
    return np.c_[radius, np.sign(np.sin(angle)) * np.sqrt(argument)], wall_flux


def _green_block(target, source, section=0.05):
    """Return the total-flux coupling [Wb/A] of one source set on one target."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


def _flat_profile(amplitude):
    """Return a constant absolute gradient."""

    def gradient(psi_norm):
        """Return the constant value at every normalised flux."""
        return jnp.full_like(jnp.asarray(psi_norm, dtype=jnp.float64), amplitude)

    return gradient


def _edge_vanishing_profile(amplitude):
    """Return an absolute gradient that falls linearly to zero at the boundary.

    A gradient that vanishes at the boundary makes the converged current
    insensitive to which edge cells the boolean core mask happens to hold, so
    the discrete fixed point is a single point rather than a lattice-quantised
    family.
    """

    def gradient(psi_norm):
        """Return the tapered value at one normalised flux."""
        return amplitude * (1.0 - jnp.clip(jnp.asarray(psi_norm), 0.0, 1.0))

    return gradient


@pytest.fixture(scope="module")
def machine():
    """Return the bootstrapped solve, its analytic seed and its vacuum field."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.6, 1.42, 25), np.linspace(-0.42, 0.42, 25))
    coordinate = lattice.coordinate
    wall, wall_flux = _wall_loop()
    seed_flux = _solovev(coordinate[:, 0], coordinate[:, 1])
    wall_seed = _solovev(wall[:, 0], wall[:, 1])
    inside = seed_flux >= wall_flux

    angle = 2 * np.pi * np.arange(CONDUCTORS) / CONDUCTORS
    conductor = np.c_[1.0 + 0.62 * np.cos(angle), 0.62 * np.sin(angle)]
    coupling = {
        "plasma_to_grid": _green_block(coordinate, coordinate),
        "plasma_to_wall": _green_block(wall, coordinate),
        "source_to_grid": _green_block(coordinate, conductor),
        "source_to_wall": _green_block(wall, conductor),
    }

    def build(core, current):
        """Return the solve for one declared source and conductor state."""
        return ForwardProfile.from_lattice(
            lattice,
            ForwardSource(core=core, boundary_field_function=BOUNDARY_FIELD_FUNCTION),
            external_current=current,
            wall_coordinate=wall,
            polarity=1,
            inside_material=inside,
            **coupling,
        )

    seed = jnp.asarray(np.r_[seed_flux, wall_seed])
    flat = build(
        DomainProfile(p_prime=_flat_profile(P_PRIME), ff_prime=_flat_profile(FF_PRIME)),
        np.zeros(CONDUCTORS),
    )
    cell_current = np.asarray(flat.operator.cell_current(seed))
    target = np.r_[
        seed_flux - coupling["plasma_to_grid"] @ cell_current,
        wall_seed - coupling["plasma_to_wall"] @ cell_current,
    ]
    weight = np.r_[inside.astype(float), np.ones(len(wall))]
    matrix = np.r_[coupling["source_to_grid"], coupling["source_to_wall"]]
    current = np.linalg.lstsq(matrix * weight[:, None], target * weight, rcond=None)[0]

    profile = build(
        DomainProfile(
            p_prime=_edge_vanishing_profile(2.0 * DRIVE * P_PRIME),
            ff_prime=_edge_vanishing_profile(2.0 * DRIVE * FF_PRIME),
        ),
        current,
    )
    return profile, seed, profile.operator.external()


@pytest.fixture(scope="module")
def converged(machine):
    """Return the reference equilibrium of the accelerated ladder."""
    profile, seed, _vacuum = machine
    return profile.solve(seed, route="anderson", evaluations=EVALUATIONS)


@pytest.fixture(scope="module")
def relaxed(machine):
    """Return the relaxed iteration driven to its stall on the full budget."""
    profile, seed, _vacuum = machine
    return profile.solve(seed, route="host", evaluations=EVALUATIONS, tolerance=0.0)


def _relative(left, right, scale):
    """Return a sup-norm difference read against one flux scale."""
    return float(jnp.max(jnp.abs(left - right))) / float(scale)


def _normalisation_quantum(equilibrium):
    """Return the flux [Wb] one step of the axis-flux read carries.

    The magnetic axis is fitted in single precision, so the axis flux the
    source is normalised against lands on a float32 ladder. The step of that
    ladder at this equilibrium's axis flux is the finest flux difference
    anything downstream of the normalisation can express.
    """
    return float(np.spacing(np.float32(float(equilibrium.topology.axis_flux))))


def test_the_accelerated_solve_reaches_its_fixed_point(converged):
    """The reference solve converges and publishes a finite receipt."""
    assert float(converged.fixed_point.residual) < RESIDUAL_TOLERANCE
    assert bool(converged.finite.passed)
    trace = np.asarray(converged.fixed_point.trace)
    assert trace.size == EVALUATIONS
    assert np.nanmin(trace) < np.nanmax(trace)
    assert abs(float(converged.moments.plasma_current)) > 1.0e5


def test_the_receipt_records_an_untouched_absolute_source(converged):
    """The normalisation ledger reports a unit amplitude and no rescale."""
    normalisation = converged.normalisation
    assert normalisation.policy_name == "absolute"
    assert float(normalisation.amplitude) == 1.0
    assert not bool(normalisation.rescaled)


def test_no_current_appears_outside_the_declared_support(converged):
    """Only the labelled core carries source current."""
    ledger = converged.ledger
    assert abs(float(ledger.core)) > 1.0e5
    assert float(ledger.common_sol) == 0.0
    assert float(ledger.private_flux) == 0.0
    assert float(ledger.excluded_material) == 0.0
    np.testing.assert_allclose(ledger.total, ledger.core, rtol=1e-12)
    np.testing.assert_allclose(
        converged.moments.plasma_current, ledger.core, rtol=1e-12
    )


def test_the_topology_read_publishes_every_domain(converged):
    """The solved map carries a core inside a scrape-off band inside material."""
    counts = np.asarray(converged.domains.cell_count())
    assert counts[PlasmaDomain.CORE] > 50
    assert counts[PlasmaDomain.COMMON_SOL] > 0
    assert counts[PlasmaDomain.EXCLUDED_MATERIAL] > 0
    assert counts.sum() == converged.domains.label.size
    assert not bool(converged.topology.diverted)
    axis = np.asarray(converged.topology.axis)
    assert 0.8 < axis[0] < 1.2
    assert abs(axis[1]) < 0.2


def test_the_conservation_receipt_meets_its_registered_tolerances(converged):
    """The solved map satisfies both field identities and the force balance."""
    ledger = converged.conservation
    assert int(ledger.checked_cells) > 20
    assert float(ledger.relative_divergence_b) < DIVERGENCE_TOLERANCE
    assert float(ledger.relative_divergence_j) < DIVERGENCE_TOLERANCE
    assert float(ledger.relative_grad_shafranov) < GRAD_SHAFRANOV_TOLERANCE
    assert float(ledger.relative_force) < FORCE_TOLERANCE


def test_the_accelerator_routes_agree_on_the_fixed_point(machine, converged):
    """Every accelerated route drives the shared map to one equilibrium."""
    profile, seed, _vacuum = machine
    scale = jnp.max(jnp.abs(converged.flux))
    for route, options in (
        ("picard", {"evaluations": 2 * EVALUATIONS}),
        ("newton_krylov", {"newton_steps": 8, "warmup": 40}),
    ):
        result = profile.solve(seed, route=route, **options)
        assert float(result.fixed_point.residual) < RESIDUAL_TOLERANCE, route
        assert _relative(result.flux, converged.flux, scale) < PARITY_TOLERANCE, route


def test_the_host_iteration_reproduces_the_traced_ladder(machine, converged):
    """The host and traced relaxed steps agree step for step."""
    profile, seed, _vacuum = machine
    host = profile.solve(seed, route="host", evaluations=TRANSIENT, tolerance=0.0)
    traced = profile.solve(seed, route="picard", evaluations=TRANSIENT)
    scale = jnp.max(jnp.abs(converged.flux))
    assert _relative(host.flux, traced.flux, scale) < IMPLEMENTATION_PARITY
    np.testing.assert_allclose(
        np.asarray(host.fixed_point.trace),
        np.asarray(traced.fixed_point.trace),
        rtol=1e-6,
    )
    assert type(host.fixed_point) is type(traced.fixed_point)
    assert type(host.conservation) is type(traced.conservation)
    assert type(host.domains) is type(traced.domains)


def test_the_relaxed_iteration_floors_at_the_normalisation_ladder(converged, relaxed):
    """The relaxed step stalls where the normalisation read is quantised.

    Once an iterate moves the magnetic axis by less than one step of the
    fit's float32 ladder, the map stops resolving its own progress: the
    relaxed step cycles between neighbouring reads of the same equilibrium
    instead of contracting, and the residual settles into a band a few
    ladder steps wide. The stall is read as a contrast between two windows
    of equal length — the transient buys orders of magnitude, the tail buys
    nothing — because the depth a run stops at inside the band is set by
    rounding below the ladder and is not a property of the equilibrium.
    """
    trace = np.asarray(relaxed.fixed_point.trace)
    scale = jnp.max(jnp.abs(converged.flux))
    axis_flux = float(relaxed.topology.axis_flux)
    # cast the narrowed value back to a Python float before comparing: a
    # float32 scalar compared against a Python float takes the narrow dtype
    # for the comparison and would report every value as representable
    assert float(np.float32(axis_flux)) == axis_flux, "the axis flux is on the ladder"
    ladder_step = _normalisation_quantum(relaxed) / float(scale)

    assert trace[0] / trace[TRANSIENT - 1] > TRANSIENT_CONTRACTION
    stalled = trace[-TRANSIENT:]
    assert 1.0 / STALL_CONTRACTION < stalled[0] / stalled[-1] < STALL_CONTRACTION
    assert STALL_BAND[0] < stalled.max() / ladder_step < STALL_BAND[1]
    assert stalled.max() < QUANTISATION_FLOOR
    assert _relative(relaxed.flux, converged.flux, scale) < QUANTISATION_FLOOR


def test_the_host_iteration_exits_early_on_the_shared_residual(
    machine, converged, relaxed
):
    """The eager route stops at the tolerance and lands on the same equilibrium.

    The exit reads the residual of the iterate it is standing on and the
    receipt is written after the relaxed update that follows it, so the two
    are one evaluation apart. The toleranced run walks the same iterates as
    the un-toleranced one, which makes the receipt that run's next trace
    entry exactly — and at the stall the map no longer contracts, so that
    entry is free to sit above the tolerance the exit tested against. What
    the tolerance bounds is the residual the loop read, not the residual the
    receipt reports.
    """
    profile, seed, _vacuum = machine
    host = profile.solve(
        seed, route="host", evaluations=EVALUATIONS, tolerance=QUANTISATION_FLOOR
    )
    trace = np.asarray(host.fixed_point.trace)
    assert np.isnan(trace).any(), "the early exit leaves the trace tail unfilled"
    tested = trace[np.isfinite(trace)]
    assert tested.size < EVALUATIONS
    assert tested[-1] < QUANTISATION_FLOOR
    ladder = np.asarray(relaxed.fixed_point.trace)
    np.testing.assert_allclose(tested, ladder[: tested.size], rtol=1e-12)
    np.testing.assert_allclose(
        float(host.fixed_point.residual), ladder[tested.size], rtol=1e-6
    )
    scale = jnp.max(jnp.abs(converged.flux))
    assert _relative(host.flux, converged.flux, scale) < BASIN_TOLERANCE
    np.testing.assert_allclose(
        host.moments.plasma_current, converged.moments.plasma_current, rtol=1e-4
    )


def test_the_host_root_find_holds_the_equilibrium_it_is_seeded_on(machine, converged):
    """The Krylov root find returns the same fixed point and the same receipt.

    Its step is globalised only by its own line search, so it is seeded on
    the confined branch; the vacuum branch it can otherwise reach is pinned
    separately below.

    The root find measures its own tolerance as an absolute sup-norm on the
    flux residual, and that residual carries the axis flux the source is
    normalised against. A target below one step of the axis read's ladder
    therefore asks for a flux difference the read cannot express: the root
    find drives the residual onto the ladder within a dozen steps and then
    spends the rest of any budget there, so the target is set from the
    ladder and not from a depth the map could not deliver.
    """
    profile, _seed, _vacuum = machine
    # an absolute target: a relative one is measured against the seed's own
    # residual, which is already on the ladder here and would leave the root
    # find chasing a tolerance that shrinks with its own starting point
    f_tol = ROOT_FIND_LADDER_STEPS * _normalisation_quantum(converged)
    host = profile.solve(converged.flux, route="host_krylov", f_tol=f_tol, maxiter=20)
    scale = jnp.max(jnp.abs(converged.flux))
    assert float(host.fixed_point.residual) < RESIDUAL_TOLERANCE
    assert _relative(host.flux, converged.flux, scale) < PARITY_TOLERANCE
    np.testing.assert_allclose(
        host.moments.plasma_current, converged.moments.plasma_current, rtol=1e-6
    )
    assert np.asarray(host.fixed_point.trace).size == profile.evaluations


def test_the_vacuum_field_is_a_second_fixed_point(machine):
    """A seed with no plasma stays there, which is what pins the seed policy.

    The absolute source drives current only where the topology read finds an
    axis-connected core, so a flux map without one reproduces itself. The
    receipt reports it honestly: an empty core, a zero ledger and no
    normalisation action taken to hide it.
    """
    profile, _seed, vacuum = machine
    result = profile.solve(vacuum, route="picard", evaluations=40)
    counts = np.asarray(result.domains.cell_count())
    assert counts[PlasmaDomain.CORE] == 0
    assert float(result.ledger.total) == 0.0
    assert float(result.moments.plasma_current) == 0.0
    assert not bool(result.normalisation.rescaled)


def test_the_solve_reaches_one_equilibrium_from_several_seeds(machine, converged):
    """Seeds spanning the plasma amplitude land on the same fixed point."""
    profile, seed, vacuum = machine
    scale = jnp.max(jnp.abs(converged.flux))
    for factor in (0.85, 1.0, 1.15):
        start = vacuum + factor * (seed - vacuum)
        result = profile.solve(start, route="anderson", evaluations=EVALUATIONS)
        assert _relative(result.flux, converged.flux, scale) < BASIN_TOLERANCE, factor
        np.testing.assert_allclose(
            result.moments.plasma_current,
            converged.moments.plasma_current,
            rtol=BASIN_TOLERANCE,
        )


def test_the_batched_ensemble_solve_matches_the_per_slice_solve(machine, converged):
    """A jitted vmap over seeds reproduces the single-slice equilibria."""
    profile, seed, vacuum = machine
    factors = (0.85, 1.0, 1.15)
    seeds = jnp.stack([vacuum + factor * (seed - vacuum) for factor in factors])
    ensemble = jax.jit(
        lambda state: profile.solve_batch(
            state, route="anderson", evaluations=EVALUATIONS
        )
    )(seeds)
    assert ensemble.flux.shape == (len(factors), seed.size)
    scale = jnp.max(jnp.abs(converged.flux))
    for index in range(len(factors)):
        single = profile.solve(seeds[index], route="anderson", evaluations=EVALUATIONS)
        assert _relative(ensemble.flux[index], single.flux, scale) < PARITY_TOLERANCE, (
            index
        )
    assert bool(jnp.all(ensemble.finite.passed))


def test_the_moment_map_is_differentiable_against_finite_differences(
    converged, machine
):
    """The published moment Jacobian reproduces a central difference."""
    profile, _seed, _vacuum = machine
    targets = MomentTargets(
        plasma_current=0.9 * float(converged.moments.plasma_current),
        poloidal_beta=0.5,
        internal_inductance=0.8,
    )
    residual = np.asarray(profile.moment_residual(converged.flux, targets))
    assert np.all(np.isfinite(residual))
    assert np.max(np.abs(residual)) > 1.0e-3

    jacobian = profile.moment_jacobian(converged.flux, targets)
    assert jacobian.shape == (3, converged.flux.size)
    rng = np.random.default_rng(11)
    direction = jnp.asarray(rng.standard_normal(converged.flux.shape))
    direction = direction / jnp.max(jnp.abs(direction))
    step = 1.0e-3
    numeric = (
        profile.moment_residual(converged.flux + step * direction, targets)
        - profile.moment_residual(converged.flux - step * direction, targets)
    ) / (2.0 * step)
    analytic = jacobian @ direction
    assert float(jnp.max(jnp.abs(analytic))) > 1.0
    error = float(jnp.max(jnp.abs(numeric - analytic))) / float(
        jnp.max(jnp.abs(analytic))
    )
    assert error < MOMENT_JACOBIAN_TOLERANCE


def test_the_solve_is_differentiable_in_the_conductor_current(machine):
    """A functional of the converged flux differentiates through the ladder."""
    profile, seed, _vacuum = machine
    conductor = profile.operator.external_current

    def linked_flux(current):
        """Return a smooth scalar of the equilibrium the conductors support."""
        state = profile.solve(
            seed, route="picard", current=current, evaluations=80, relaxation=0.7
        )
        return jnp.sum(state.flux**2)

    gradient = jax.grad(linked_flux)(conductor)
    assert np.all(np.isfinite(np.asarray(gradient)))
    probe = int(np.argmax(np.abs(np.asarray(gradient))))
    delta = GRADIENT_STEP * float(jnp.abs(conductor[probe]))
    numeric = float(
        (
            linked_flux(conductor.at[probe].add(delta))
            - linked_flux(conductor.at[probe].add(-delta))
        )
        / (2.0 * delta)
    )
    error = abs(float(gradient[probe]) - numeric) / max(abs(numeric), 1.0e-30)
    assert error < FLUX_GRADIENT_TOLERANCE


def test_the_plasma_current_is_differentiable_in_the_conductor_current(machine):
    """The observed plasma current carries a derivative through the solve."""
    profile, seed, _vacuum = machine
    conductor = profile.operator.external_current

    def observed_current(current):
        """Return the plasma current the conductors support."""
        return profile.solve(
            seed, route="picard", current=current, evaluations=80, relaxation=0.7
        ).moments.plasma_current

    gradient = jax.grad(observed_current)(conductor)
    probe = int(np.argmax(np.abs(np.asarray(gradient))))
    assert abs(float(gradient[probe])) > 0.0
    delta = GRADIENT_STEP * float(jnp.abs(conductor[probe]))
    numeric = float(
        (
            observed_current(conductor.at[probe].add(delta))
            - observed_current(conductor.at[probe].add(-delta))
        )
        / (2.0 * delta)
    )
    error = abs(float(gradient[probe]) - numeric) / max(abs(numeric), 1.0e-30)
    assert error < CURRENT_GRADIENT_TOLERANCE


def test_the_solve_refuses_to_enforce_a_moment(machine, converged):
    """Enforcement fails and the converged solve stays reproducible."""
    profile, seed, _vacuum = machine
    with pytest.raises(MomentEnforcementError, match="scalar closure"):
        profile.solve(seed, enforce=("plasma_current",))
    with pytest.raises(MomentEnforcementError, match="scalar closure"):
        profile.solve_batch(jnp.stack([seed]), enforce=("poloidal_beta",))
    repeat = profile.solve(seed, route="anderson", evaluations=EVALUATIONS)
    np.testing.assert_array_equal(np.asarray(repeat.flux), np.asarray(converged.flux))


def test_an_externally_supplied_flux_map_is_qualified_by_the_same_receipt(
    machine, converged
):
    """The observation entry point returns the solve's contract without iterating."""
    profile, _seed, _vacuum = machine
    observed = profile.observe(converged.flux)
    assert float(observed.fixed_point.residual) < RESIDUAL_TOLERANCE
    np.testing.assert_allclose(
        observed.moments.plasma_current, converged.moments.plasma_current, rtol=1e-12
    )
    np.testing.assert_array_equal(
        np.asarray(observed.domains.label), np.asarray(converged.domains.label)
    )


if __name__ == "__main__":
    pytest.main([__file__])
