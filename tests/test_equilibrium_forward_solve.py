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

How deep any route can drive the map is set by the finest flux difference the
map can express, and two mechanisms compete to set it. The source is
normalised against the axis-to-boundary flux span, and the axis flux comes
from a sub-cell fit of the grid stencil, so it is only ever known on the
ladder of that fit's dtype; everything downstream — the residual, the plasma
current, the moments — inherits one step of it. Against that stands the
round-off the plasma coupling accumulates, a dot product over the grid's
nodes carrying up to that many units in the last place of the flux scale. A
float32 fit puts its ladder decades above the sum and binds alone; a float64
fit puts one step at a single unit in the last place and the sum binds
instead. The pins that touch the convergence floor are written against the
coarser of the two rather than against a fixed depth. Where a route stops
inside the resulting band is decided by rounding far beneath it and is a
property of the arithmetic the host emits, not of the equilibrium.

Pinned here: route parity between the host root find and the accelerated
ladder, the basin over multiple seeds, the batched ensemble solve under
``jit``/``vmap``, the differentiable moment map against finite differences,
the gradient of a converged functional with respect to a conductor current,
the current ledger's declared support, and the conservation receipts.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import mu_0

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.biot.null import Null1D
    from nova.biot.target import FluxTarget
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
#: Relative residual the registered early-exit floor sits at. It has to clear
#: the band the relaxed step stalls in and stay far under the transient it
#: terminates. That band moves with the fit dtype — seven decades between a
#: float32 fit, whose ladder step is 1.1e-7 of this machine's flux scale, and
#: a float64 one, which stalls on the coupling sum's round-off near 1.3e-13 —
#: so the floor is registered against the coarser of the two and bounds the
#: stall under either.
QUANTISATION_FLOOR = 1.0e-6
#: Contraction ``TRANSIENT`` evaluations buy while the iterate still moves the
#: axis read, and the factor the same count of evaluations is allowed to move
#: the residual by once it no longer does. Separating the two is the statement
#: that the iteration has stalled: a contrast between two equal-length windows
#: of one run, which no single depth could express.
TRANSIENT_CONTRACTION = 1.0e3
STALL_CONTRACTION = 1.0e2
#: Multiples of the solve's flux resolution the stalled residual band is
#: required to span. Whichever mechanism sets that resolution re-enters the
#: residual amplified by a factor of order unity: a step of the axis ladder
#: shifts the whole normalised profile, and the coupling sum's round-off is
#: already measured in the residual's own units. The band brackets that
#: amplification over two decades rather than predicting it, and separates
#: the stall from both round-off and the transient above it.
STALL_BAND = (0.1, 10.0)
#: Multiples of the flux resolution the Krylov root find's absolute residual
#: target is set at. A target under one asks for a flux difference the map
#: cannot express; eight clears the few the solve was measured to move by.
ROOT_FIND_RESOLUTION_STEPS = 8
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


def _with_direct_samples(profile, seed):
    """Attach exact grid-free sample rows to the synthetic structured solve."""
    operator = profile.operator
    coordinate = np.asarray(operator.moment_geometry.sample_node_coordinates)
    sample = FluxTarget(
        source_target=jnp.zeros((len(coordinate), len(operator.external_current))),
        plasma_target=jnp.zeros((len(coordinate), operator.grid.node_number)),
        null=Null1D(jnp.asarray(coordinate)),
    )
    sampled_seed = _solovev(coordinate[:, 0], coordinate[:, 1])
    return (
        replace(operator, sample=sample, use_linear_moments=True),
        jnp.r_[seed, sampled_seed],
    )


def test_solve_path_current_moments_match_direct_stencil_and_clip(machine):
    """Interior rings and every LCFS crossing use their direct moment rule."""
    profile, seed, _vacuum = machine
    operator, seed = _with_direct_samples(profile, seed)
    masks, topology = operator.read(seed)
    geometry = operator.moment_geometry
    shared_flux = operator.shared_node_flux(seed)
    clipped = geometry.atomic_mesh.traced_clip(
        operator.polarity * (shared_flux - topology.boundary_flux)
    )
    sample_flux = operator.sample_node_flux(seed)
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    direct = operator.support_current_moments(
        operator.source.core,
        masks.psi_norm,
        sample_psi_norm,
        clipped,
    )
    closed_branch = masks.core | masks.common_sol
    direct = type(direct)(*(jnp.where(closed_branch, entry, 0.0) for entry in direct))
    radial_second, vertical_second, cross_second = geometry.second_moment.T
    determinant = radial_second * vertical_second - cross_second**2
    expected = (
        direct.cell_current,
        (vertical_second * direct.radial_moment - cross_second * direct.vertical_moment)
        / determinant,
        (radial_second * direct.vertical_moment - cross_second * direct.radial_moment)
        / determinant,
    )
    actual = operator.cell_current_moments(seed)
    for observed, direct in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, direct, rtol=2e-13, atol=2e-13)

    crossing_closed = np.asarray(clipped.boundary & (masks.core | masks.common_sol))
    crossing_unqualified = np.asarray(
        clipped.boundary & ~(masks.core | masks.common_sol)
    )
    assert crossing_closed.sum() > 0
    assert np.count_nonzero(np.asarray(actual.cell_current)[crossing_closed]) == int(
        crossing_closed.sum()
    )
    assert np.count_nonzero(np.asarray(actual.cell_current)[crossing_unqualified]) == 0


def test_unclipped_support_uses_the_own_node_profile_for_every_cell(machine):
    """The full support carries every cell through the own-node profile."""
    profile, seed, _vacuum = machine
    operator, seed = _with_direct_samples(profile, seed)
    masks, topology = operator.read(seed)
    complete = operator.moment_geometry.atomic_mesh.traced_clip(
        jnp.ones(len(operator.moment_geometry.atomic_mesh.node_coordinates))
    )
    sample_flux = operator.sample_node_flux(seed)
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    observed = operator.support_current_moments(
        operator.source.core,
        masks.psi_norm,
        sample_psi_norm,
        complete,
    )
    carried = np.sort(
        np.concatenate(
            [stencil.ring_centre for stencil in operator._support_moment_stencils]
        )
    )
    np.testing.assert_array_equal(carried, np.arange(operator.grid.node_number))
    for entry in observed:
        assert np.all(np.isfinite(np.asarray(entry)))


def test_linear_moment_construction_requires_direct_sample_targets(machine):
    """A linear-moment operator cannot fall back to reconstructed samples."""
    profile, _seed, _vacuum = machine
    with pytest.raises(ValueError, match="direct pre-clip sample targets"):
        replace(profile.operator, sample=None, use_linear_moments=True)


def test_receipts_and_integral_observation_share_solve_current_moments(machine):
    """Both observation entries consume the zeroth moment used by the map."""
    profile, seed, _vacuum = machine
    operator, seed = _with_direct_samples(profile, seed)
    profile = replace(profile, operator=operator)
    expected = operator.cell_current_moments(seed).cell_current
    receipt = profile.observe(seed)
    np.testing.assert_array_equal(receipt.cell_current, expected)
    np.testing.assert_array_equal(
        profile.integral_observation(seed).plasma_current,
        receipt.moments.plasma_current,
    )


def _fit_dtype(profile):
    """Return the dtype the grid locator fits its sub-cell nulls in."""
    return np.dtype(profile.operator.grid.null.fit_dtype)


def _normalisation_quantum(profile, equilibrium):
    """Return the flux [Wb] one step of the axis-flux read carries.

    The magnetic axis is fitted in the locator's own dtype, so the axis flux
    the source is normalised against lands on that dtype's ladder. The step
    of that ladder at this equilibrium's axis flux is the finest flux
    difference the normalisation itself can express.

    The step is read on the magnitude of that flux. ``np.spacing`` carries
    the sign of its argument, and the sign of an axis flux is set by the
    plasma current direction and the flux convention, not by anything about
    the resolution: a machine sitting on a negative axis flux would return a
    negative quantum, which loses every ``max`` against the coupling sum and
    inverts each comparison written against the resolution. The ladder is
    symmetric about zero, so stepping the magnitude gives the same distance.
    """
    axis_flux = abs(float(equilibrium.topology.axis_flux))
    return float(np.spacing(_fit_dtype(profile).type(axis_flux)))


def _flux_resolution(profile, equilibrium, scale):
    """Return the finest flux difference [Wb] the whole solve can express.

    Two floors compete and the coarser one binds. One is a step of the
    axis-flux ladder. The other is the round-off the plasma coupling
    accumulates: every residual entry is a dot product over the grid's nodes,
    so it carries up to that many units in the last place of the flux scale.
    A float32 fit puts its ladder decades above the sum and binds alone; a
    float64 fit puts one step at a single unit in the last place, underneath
    the sum, and the sum binds instead.
    """
    accumulation = profile.operator.grid.node_number * float(np.spacing(float(scale)))
    return max(_normalisation_quantum(profile, equilibrium), accumulation)


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


def _resolution_stub(fit_dtype, axis_flux, node_number=1):
    """Return the smallest profile/equilibrium pair the resolution read touches.

    The two floors are derived from a fit dtype, an axis flux and a node
    count alone, so the sign handling can be pinned without solving anything.
    """
    grid = SimpleNamespace(
        null=SimpleNamespace(fit_dtype=fit_dtype), node_number=node_number
    )
    profile = SimpleNamespace(operator=SimpleNamespace(grid=grid))
    equilibrium = SimpleNamespace(topology=SimpleNamespace(axis_flux=axis_flux))
    return profile, equilibrium


@pytest.mark.parametrize("fit_dtype", ["float32", "float64"])
def test_the_flux_resolution_is_a_magnitude_on_either_axis_flux_sign(fit_dtype):
    """A sign-flipped axis flux leaves both floors, and their max, unchanged.

    The fixture below carries a positive axis flux, but the sign is a
    convention: reference machines with a negative axis flux are ordinary.
    ``np.spacing`` carries the sign of its argument, so reading the ladder
    step off a signed flux would hand back a negative quantum on exactly
    those machines — it would lose the ``max`` against the coupling sum and
    invert every comparison written against the resolution.
    """
    flux = 1.5
    step = float(np.spacing(np.dtype(fit_dtype).type(flux)))
    for sign in (1.0, -1.0):
        quantum = _normalisation_quantum(*_resolution_stub(fit_dtype, sign * flux))
        assert quantum == step > 0.0, sign

    # this node count puts the coupling sum below a float32 ladder step and
    # above a float64 one, so the two fits bind on opposite floors and a
    # sign-flipped quantum would be visible in either direction
    scale = 1.0
    nodes = 4096
    accumulation = nodes * float(np.spacing(scale))
    binding = step if fit_dtype == "float32" else accumulation
    assert (step > accumulation) == (fit_dtype == "float32")
    for sign in (1.0, -1.0):
        resolution = _flux_resolution(
            *_resolution_stub(fit_dtype, sign * flux, node_number=nodes), scale
        )
        assert resolution == binding, sign


def test_the_relaxed_iteration_floors_at_the_flux_resolution(
    machine, converged, relaxed
):
    """The relaxed step stalls where the map stops resolving its own progress.

    Once an iterate moves the flux by less than the map can express — one
    step of the fit's ladder, or the round-off its coupling sum accumulates,
    whichever is coarser — the relaxed step cycles between neighbouring reads
    of the same equilibrium instead of contracting, and the residual settles
    into a band a small multiple of that resolution wide. The stall is read
    as a contrast between two windows of equal length — the transient buys
    orders of magnitude, the tail buys nothing — because the depth a run
    stops at inside the band is set by rounding below the resolution and is
    not a property of the equilibrium.
    """
    profile, _seed, _vacuum = machine
    trace = np.asarray(relaxed.fixed_point.trace)
    scale = jnp.max(jnp.abs(converged.flux))
    axis_flux = float(relaxed.topology.axis_flux)
    # cast the narrowed value back to a Python float before comparing: a
    # narrow scalar compared against a Python float takes the narrow dtype
    # for the comparison and would report every value as representable
    narrowed = float(_fit_dtype(profile).type(axis_flux))
    assert narrowed == axis_flux, "the axis flux is on the fit's ladder"
    resolution = _flux_resolution(profile, relaxed, scale) / float(scale)

    assert trace[0] / trace[TRANSIENT - 1] > TRANSIENT_CONTRACTION
    stalled = trace[-TRANSIENT:]
    assert 1.0 / STALL_CONTRACTION < stalled[0] / stalled[-1] < STALL_CONTRACTION
    assert STALL_BAND[0] < stalled.max() / resolution < STALL_BAND[1]
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

    The root find measures its own tolerance as an absolute norm on the flux
    residual, and that residual carries both the axis flux the source is
    normalised against and the round-off of the coupling sum that produced
    it. A target below the resolution those two set asks for a flux
    difference the map cannot express, and the root find exhausts its budget
    and raises rather than returning. Seeded on a converged map it needs one
    Krylov step, and where that step lands is decided by rounding beneath the
    resolution, so the target is set from the resolution with enough margin
    to hold on a host that rounds the other way.
    """
    profile, _seed, _vacuum = machine
    scale = jnp.max(jnp.abs(converged.flux))
    # an absolute target: a relative one is measured against the seed's own
    # residual, which is already on the floor here and would leave the root
    # find chasing a tolerance that shrinks with its own starting point
    f_tol = ROOT_FIND_RESOLUTION_STEPS * _flux_resolution(profile, converged, scale)
    host = profile.solve(converged.flux, route="host_krylov", f_tol=f_tol, maxiter=20)
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
