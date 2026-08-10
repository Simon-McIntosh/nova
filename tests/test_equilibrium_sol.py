r"""Contract of the bounded source continuation beyond the separatrix.

Two things are pinned here, and they are pinned separately because they fail
in different ways.

The declaration is pinned without solving anything: what a continuation must
state before it may be used, that a jump at the separatrix is refused until a
current layer owns it, that the value and first derivative it claims to match
really are the core closure's own on either branch, that the gradients and
their primitives are exactly zero and exactly constant beyond the declared
bound, and that the private-flux branch is an independent policy — varying it
moves its own current and nothing else.

The equilibrium is pinned by solving a bootstrapped machine twice, once with
the core closure alone and once with the same closure continued into the
common scrape-off layer. The machine's material boundary is a circle rather
than a flux surface, which is what gives it a scrape-off band worth continuing
into: with a wall lying on a seed surface the open region inside the material
is the thin shell between the last-closed-flux-surface cut and the separatrix,
and a continuation there would have nowhere to act.

A declared continuation is expected to MOVE the solution — it is extra current
inside the vessel — so the shift is measured and bounded rather than
suppressed. What must not move is the solve that declares nothing: with a core
closure whose gradients and slopes both vanish at the separatrix, declaring the
machinery on both open branches reproduces the undeclared solve exactly, array
for array.
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
    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.continuation import (
        ANCHOR_TOLERANCE,
        SeparatrixContinuation,
        SeparatrixJumpError,
        separatrix_derivatives,
    )
    from nova.equilibrium.convention import toroidal_current_density
    from nova.equilibrium.domain import PlasmaDomain
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.observation import current_ledger
    from nova.equilibrium.source import (
        ContinuationForm,
        DomainProfile,
        ForwardSource,
        SeparatrixContinuity,
    )
    from nova.equilibrium.topology import Topology
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes

P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_PRESSURE = 1.2e3
BOUNDARY_FIELD_FUNCTION = 5.0
CONDUCTORS = 16
EVALUATIONS = 240
#: Evaluations the relaxed transient stays inside one domain labelling.
TRANSIENT = 30
#: Radius of the circular material boundary [m]. A wall that is not a flux
#: surface limits the plasma at one point and leaves the rest of the vessel
#: open, which is the configuration a scrape-off closure is declared for.
WALL_RADIUS = 0.30
#: Gradient left at the separatrix as a fraction of the axis value. A finite
#: edge gradient is what a continuation has to match; an edge-vanishing core
#: closure is continued by zero, which is pinned separately.
EDGE_FRACTION = 0.1
#: Declared support of the scrape-off continuation, in separatrix distance.
#: A first-derivative match carries the core's own outward slope, so a support
#: much wider than the value-to-slope ratio turns the gradient through zero
#: before the taper closes it — bounded, still continuous, and no longer a
#: monotone decay.
SUPPORT = 0.25
PRIVATE_SUPPORT = 0.05
DECAY_WIDTH = 0.08

#: Pre-registered tolerances. The fixed-point residual is loose because the
#: converged depth is host dependent — the relaxed step floors where one edge
#: cell alternates in and out of the labelling — while the shifts and ledger
#: rows below it are reproducible to several digits. The two conservation
#: residuals are read on the union of the declared domains and sit at the
#: same central-difference floor the core alone reaches, which is the
#: measurement that says the continued region is an equilibrium and not just a
#: smooth profile.
RESIDUAL_TOLERANCE = 1.0e-5
DIVERGENCE_TOLERANCE = 1.0e-12
GRAD_SHAFRANOV_TOLERANCE = 0.05
FORCE_TOLERANCE = 0.05
PARITY_TOLERANCE = 1.0e-6
#: Offset in normalised flux the two sides of the separatrix are read at, and
#: the agreement demanded between them. Each derivative is differentiated
#: rather than differenced, so the residual jump is the genuine
#: :math:`O(\text{offset})` variation of the next derivative and not a
#: cancellation error.
CONTINUITY_STEP = 1.0e-7
CONTINUITY_TOLERANCE = 1.0e-4
#: Bounds on the shift a declared scrape-off continuation makes. It must be
#: resolvable — a continuation that changed nothing would not be a physical
#: source — and it must stay a perturbation of the same equilibrium rather
#: than a jump to another branch.
SHIFT_FLOOR = 1.0e-3
SHIFT_CEILING = 1.0e-1


def _terms():
    """Return the Solov'ev quartic, offset and vertical coefficients."""
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius, height):
    """Return the analytic seed flux [Wb] the conductors are fitted to."""
    alpha, offset, beta = _terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _green_block(target, source, section=0.05):
    """Return the total-flux coupling [Wb/A] of one source set on one target."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


def _pedestal(amplitude, edge=EDGE_FRACTION):
    """Return an absolute gradient with a finite value at the separatrix.

    Linear in normalised flux and written without a clip, so it carries one
    unambiguous derivative at the separatrix for a continuation to match.
    """

    def gradient(psi_norm):
        """Return the tapered value at one normalised flux."""
        return amplitude * (edge + (1.0 - edge) * (1.0 - jnp.asarray(psi_norm)))

    return gradient


def _curved(amplitude, edge=EDGE_FRACTION):
    """Return a gradient with value, slope and curvature all finite at the edge.

    Quadratic in normalised flux, so a continuation declaring the curvature
    class has a non-zero second derivative to match rather than a trivial one.
    """

    def gradient(psi_norm):
        """Return the tapered value at one normalised flux."""
        flux = jnp.asarray(psi_norm)
        return amplitude * (edge + (1.0 - edge) * (1.0 - flux) * (1.0 + 0.6 * flux))

    return gradient


def _edge_vanishing(amplitude):
    """Return a gradient whose value and slope both vanish at the separatrix."""

    def gradient(psi_norm):
        """Return the quadratically tapered value at one normalised flux."""
        return amplitude * (1.0 - jnp.asarray(psi_norm)) ** 2

    return gradient


def _core(amplitude=2.0 * DRIVE):
    """Return the core closure the continuations are anchored on."""
    return DomainProfile(
        p_prime=_pedestal(amplitude * P_PRIME),
        ff_prime=_pedestal(amplitude * FF_PRIME),
    )


def _policy(
    form=ContinuationForm.HERMITE_POLYNOMIAL,
    continuity=SeparatrixContinuity.VALUE_AND_GRADIENT,
    support=SUPPORT,
    **options,
):
    """Return one declared continuation policy."""
    return SeparatrixContinuation(
        form=form, continuity=continuity, support=support, **options
    )


@pytest.fixture(scope="module", autouse=True)
def device_precision():
    """Publish the fp64 device policy the continuation contract is read in."""
    configure_dtypes()


@pytest.fixture(scope="module")
def machine():
    """Return a solve builder for one bootstrapped machine and its seed.

    The conductor currents are fitted to hold the analytic seed against a flat
    source, exactly as the static solve contract bootstraps its machine, so the
    continuation is added to a real free-boundary problem rather than to an
    algebraic identity.
    """
    lattice = FluxLattice(np.linspace(0.6, 1.42, 25), np.linspace(-0.42, 0.42, 25))
    coordinate = lattice.coordinate
    angle = 2 * np.pi * np.arange(61) / 61
    wall = np.c_[AXIS_RADIUS + WALL_RADIUS * np.cos(angle), WALL_RADIUS * np.sin(angle)]
    seed_flux = _solovev(coordinate[:, 0], coordinate[:, 1])
    wall_seed = _solovev(wall[:, 0], wall[:, 1])
    inside = (coordinate[:, 0] - AXIS_RADIUS) ** 2 + coordinate[
        :, 1
    ] ** 2 <= WALL_RADIUS**2

    conductor_angle = 2 * np.pi * np.arange(CONDUCTORS) / CONDUCTORS
    conductor = np.c_[
        1.0 + 0.62 * np.cos(conductor_angle), 0.62 * np.sin(conductor_angle)
    ]
    coupling = {
        "plasma_to_grid": _green_block(coordinate, coordinate),
        "plasma_to_wall": _green_block(wall, coordinate),
        "source_to_grid": _green_block(coordinate, conductor),
        "source_to_wall": _green_block(wall, conductor),
    }

    def assemble(source, current):
        """Return the solve for one declared source and conductor state."""
        return ForwardProfile.from_lattice(
            lattice,
            source,
            external_current=current,
            wall_coordinate=wall,
            polarity=1,
            inside_material=inside,
            **coupling,
        )

    seed = jnp.asarray(np.r_[seed_flux, wall_seed])
    flat = assemble(
        ForwardSource(
            core=DomainProfile(
                p_prime=lambda psi: jnp.full_like(jnp.asarray(psi), P_PRIME),
                ff_prime=lambda psi: jnp.full_like(jnp.asarray(psi), FF_PRIME),
            )
        ),
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

    def build(core, **closures):
        """Return the solve for one core closure and its open continuations."""
        return assemble(
            ForwardSource(
                core=core,
                boundary_pressure=BOUNDARY_PRESSURE,
                boundary_field_function=BOUNDARY_FIELD_FUNCTION,
                **closures,
            ),
            current,
        )

    return build, seed


@pytest.fixture(scope="module")
def baseline(machine):
    """Return the equilibrium of the core closure alone."""
    build, seed = machine
    return build(_core()).solve(seed, route="anderson", evaluations=EVALUATIONS)


@pytest.fixture(scope="module")
def continued(machine):
    """Return the equilibrium of the same closure continued into the SOL."""
    build, seed = machine
    core = _core()
    profile = build(core, common_sol=_policy().extend(core, PlasmaDomain.COMMON_SOL))
    return profile.solve(seed, route="anderson", evaluations=EVALUATIONS)


@pytest.fixture(scope="module")
def diverted():
    """Return a two-ring diverted flux map, its labels and its lattice.

    A plasma ring and a weaker divertor ring below it produce a saddle between
    them, so the map carries a genuine private-flux pocket: cells around the
    divertor ring sit at normalised flux below one exactly like the core, yet
    the X-point cut separates them from the axis. Nothing is solved here — the
    map is a labelled substrate for reading the source on both open branches.
    """
    lattice = FluxLattice(np.linspace(0.55, 1.45, 45), np.linspace(-0.9, 0.5, 71))
    coordinate = lattice.coordinate
    rings = np.array([[1.0, 0.05], [1.0, -0.62]])
    current = np.array([1.0e6, 5.0e5])
    flux = _green_block(coordinate, rings, section=0.06) @ current
    angle = 2 * np.pi * np.arange(48) / 48
    wall = np.c_[1.0 + 0.42 * np.cos(angle), -0.2 + 0.66 * np.sin(angle)]
    wall_flux = _green_block(wall, rings, section=0.06) @ current
    topology = Topology(
        Null2D.from_coordinates(coordinate, hex_stencil(lattice.shape), maxsize=5),
        Null1D(jnp.asarray(wall)),
    )
    inside = jnp.asarray(
        ((coordinate[:, 0] - 1.0) / 0.42) ** 2 + ((coordinate[:, 1] + 0.2) / 0.66) ** 2
        <= 1.0
    )
    masks, state = topology.read(jnp.asarray(np.r_[flux, wall_flux]), 1, inside)
    return lattice, masks, state


def _relative(left, right, scale):
    """Return a sup-norm difference read against one flux scale."""
    return float(jnp.max(jnp.abs(left - right))) / float(scale)


def _one_sided_slope(gradient, sense, step=CONTINUITY_STEP):
    """Return the slope of a gradient approaching the separatrix from one side.

    Second-order accurate and evaluated strictly on one side, so it measures
    what a reader approaching the separatrix along that branch would find
    rather than what the callable returns at the point itself.
    """
    values = [
        float(jnp.asarray(gradient(jnp.asarray(1.0 + sense * offset * step))))
        for offset in range(3)
    ]
    return (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * sense * step)


def _derivative(gradient, psi_norm, order):
    """Return the ``order``-th derivative of a flux function at one point."""
    derivative = gradient
    for _ in range(order):
        derivative = jax.grad(derivative)
    return float(jnp.asarray(derivative(jnp.asarray(psi_norm))))


def test_a_continuation_publishes_the_class_form_and_support_it_declared():
    """The receipt states everything needed to read the continuation back."""
    core = _core()
    record = _policy().extend(core, PlasmaDomain.COMMON_SOL).continuation_record()
    assert bool(record.active)
    assert record.form_name == "hermite_polynomial"
    assert record.continuity_name == "value_and_gradient"
    assert record.domain_name == "common_sol"
    assert float(record.support) == SUPPORT
    assert np.isnan(float(record.decay_width))
    assert float(record.truncated_fraction) < 1.0e-12
    anchor = separatrix_derivatives(core.p_prime, 1)[0]
    np.testing.assert_allclose(record.separatrix_pressure_gradient, anchor, rtol=1e-12)
    assert float(record.separatrix_pressure_gradient) != 0.0

    undeclared = DomainProfile(
        p_prime=core.p_prime, ff_prime=core.ff_prime
    ).continuation_record()
    assert not bool(undeclared.active)
    assert undeclared.domain_name == "undeclared"
    assert float(undeclared.support) == 0.0


def test_a_separatrix_jump_is_refused_until_a_current_layer_owns_it():
    """A jump in the source is a sheet, and no sheet model is declared."""
    with pytest.raises(SeparatrixJumpError, match="toroidal surface current"):
        _policy(continuity=SeparatrixContinuity.VALUE_JUMP)
    with pytest.raises(SeparatrixJumpError, match="thin current layer"):
        _policy(continuity=SeparatrixContinuity.GRADIENT_JUMP)
    for continuity in (
        SeparatrixContinuity.VALUE_JUMP,
        SeparatrixContinuity.GRADIENT_JUMP,
    ):
        with pytest.raises(SeparatrixJumpError, match="value_and_gradient"):
            _policy(continuity=continuity)
    with pytest.raises(ValueError, match="how many orders"):
        _policy(continuity=SeparatrixContinuity.UNDECLARED)


def test_a_continuation_declares_a_bound_and_only_the_scale_its_form_uses():
    """An unbounded support or a width the form ignores is refused."""
    with pytest.raises(ValueError, match="positive separatrix distance"):
        _policy(support=0.0)
    with pytest.raises(ValueError, match="carries no decay width"):
        _policy(decay_width=0.1)
    with pytest.raises(ValueError, match="positive decay width"):
        _policy(form=ContinuationForm.EXPONENTIAL_DECAY)
    with pytest.raises(ValueError, match="functional form"):
        _policy(form=ContinuationForm.UNDECLARED)
    with pytest.raises(ValueError, match="not an open domain"):
        _policy().extend(_core(), PlasmaDomain.CORE)


def test_an_open_domain_refuses_a_closure_that_declared_nothing():
    """A bare profile on an open argument carries no continuation contract."""
    core = _core()
    for name in ("common_sol", "private_flux"):
        with pytest.raises(NotImplementedError, match="continuity class"):
            ForwardSource(core=core, **{name: core})
    with pytest.raises(ValueError, match="each open domain needs its own"):
        ForwardSource(
            core=core, common_sol=_policy().extend(core, PlasmaDomain.PRIVATE_FLUX)
        )


def test_a_continuation_anchored_off_the_core_is_a_jump_in_disguise():
    """An independent private-flux policy is not an independent anchor."""
    core = _core()
    colder = DomainProfile(
        p_prime=_pedestal(0.5 * 2.0 * DRIVE * P_PRIME),
        ff_prime=_pedestal(2.0 * DRIVE * FF_PRIME),
    )
    with pytest.raises(SeparatrixJumpError, match="current sheet no model here owns"):
        ForwardSource(
            core=core,
            private_flux=_policy(support=PRIVATE_SUPPORT).extend(
                colder, PlasmaDomain.PRIVATE_FLUX
            ),
        )


def test_a_kinked_core_gradient_has_no_separatrix_derivative_to_match():
    """A clipped profile carries two slopes at the separatrix, so it is refused.

    Clipping the argument is the ordinary way to keep a profile finite outside
    the core, and it leaves the callable differentiable everywhere except at
    the point a continuation anchors on. Automatic differentiation resolves the
    tie there by averaging the two sides, so the continuation would meet its
    class against a slope half the core's.
    """

    def clipped(psi_norm):
        """Return a gradient tapered with a clipped argument."""
        return 2.0 * DRIVE * P_PRIME * (1.0 - jnp.clip(jnp.asarray(psi_norm), 0.0, 1.0))

    inside = _one_sided_slope(clipped, -1.0, step=1.0e-3)
    differentiated = float(jax.grad(clipped)(jnp.asarray(1.0)))
    assert abs(differentiated - inside) > 0.1 * abs(inside)
    with pytest.raises(ValueError, match="no single derivative at the separatrix"):
        _policy().extend(
            DomainProfile(p_prime=clipped, ff_prime=_pedestal(FF_PRIME)),
            PlasmaDomain.COMMON_SOL,
        )


@pytest.mark.parametrize("domain", [PlasmaDomain.COMMON_SOL, PlasmaDomain.PRIVATE_FLUX])
@pytest.mark.parametrize(
    ("form", "options"),
    [
        (ContinuationForm.HERMITE_POLYNOMIAL, {}),
        (ContinuationForm.EXPONENTIAL_DECAY, {"decay_width": DECAY_WIDTH}),
    ],
)
@pytest.mark.parametrize(
    "continuity",
    [
        SeparatrixContinuity.VALUE_AND_GRADIENT,
        SeparatrixContinuity.VALUE_GRADIENT_AND_CURVATURE,
    ],
)
def test_the_continuation_meets_the_core_at_the_separatrix(
    domain, form, options, continuity
):
    """Every declared order is continuous in normalised flux across the cut.

    The two sides are read a fixed offset inside the core and inside the open
    domain, so what is measured is the discontinuity a reader crossing the
    separatrix would see rather than whatever either callable returns at the
    point itself. Each order is scaled by the natural size of that derivative,
    the separatrix value over the shortest declared length, because a matched
    derivative may legitimately be zero and cannot scale itself.
    """
    core = DomainProfile(
        p_prime=_curved(2.0 * DRIVE * P_PRIME),
        ff_prime=_curved(2.0 * DRIVE * FF_PRIME),
    )
    sense = 1.0 if domain is PlasmaDomain.COMMON_SOL else -1.0
    continuation = _policy(form=form, continuity=continuity, **options).extend(
        core, domain
    )
    length = min(SUPPORT, options.get("decay_width", SUPPORT))
    for inner, outer in (
        (core.p_prime, continuation.p_prime),
        (core.ff_prime, continuation.ff_prime),
    ):
        value = abs(_derivative(inner, 1.0, 0))
        assert value > 0.0
        for order in range(continuity.matched_orders):
            scale = value / length**order
            inside = _derivative(inner, 1.0 - sense * CONTINUITY_STEP, order)
            outside = _derivative(outer, 1.0 + sense * CONTINUITY_STEP, order)
            assert abs(outside - inside) < CONTINUITY_TOLERANCE * scale, order
            if order < 2:
                assert abs(inside) > 1.0e-3 * scale, order


@pytest.mark.parametrize("domain", [PlasmaDomain.COMMON_SOL, PlasmaDomain.PRIVATE_FLUX])
def test_the_continued_source_is_exactly_zero_beyond_its_support(domain):
    """Past the declared bound the gradients vanish and the primitives freeze."""
    core = _core()
    sense = 1.0 if domain is PlasmaDomain.COMMON_SOL else -1.0
    continuation = _policy().extend(core, domain)
    beyond = jnp.asarray(1.0 + sense * (SUPPORT + np.array([1.0e-9, 0.05, 0.5, 4.0])))
    radius = jnp.full(beyond.shape, AXIS_RADIUS)
    span = jnp.asarray(-1.0)
    for gradient in (continuation.p_prime, continuation.ff_prime):
        assert np.all(np.asarray(gradient(beyond)) == 0.0)
    assert np.all(np.asarray(continuation.current_density(radius, beyond)) == 0.0)
    pressure = np.asarray(
        continuation.pressure(radius, beyond, BOUNDARY_PRESSURE, span)
    )
    squared = np.asarray(
        continuation.field_function_squared(beyond, BOUNDARY_FIELD_FUNCTION, span)
    )
    for frozen in (pressure, squared):
        np.testing.assert_array_equal(frozen, np.full(frozen.shape, frozen[0]))
    assert abs(pressure[0] - BOUNDARY_PRESSURE) > 0.0
    assert abs(squared[0] - BOUNDARY_FIELD_FUNCTION**2) > 0.0


@pytest.mark.parametrize("domain", [PlasmaDomain.COMMON_SOL, PlasmaDomain.PRIVATE_FLUX])
def test_the_open_primitives_integrate_outward_from_the_boundary(domain):
    """Pressure and the field function follow their own gradients exactly.

    Both families integrate in closed form, so the primitive is checked
    against a quadrature of the published gradient rather than against a
    second closed form: an error in the sign of the branch, in the flux span
    or in the convention shows up as a first-order disagreement.
    """
    core = _core()
    sense = 1.0 if domain is PlasmaDomain.COMMON_SOL else -1.0
    continuation = _policy().extend(core, domain)
    span = -0.78
    distance = np.linspace(0.0, 1.2 * SUPPORT, 20001)
    psi_norm = jnp.asarray(1.0 + sense * distance)
    radius = jnp.full(psi_norm.shape, AXIS_RADIUS)

    def outward_integral(gradient):
        """Return the published gradient integrated away from the separatrix."""
        sampled = np.asarray(gradient(psi_norm))
        return np.concatenate(
            [[0.0], np.cumsum(0.5 * (sampled[1:] + sampled[:-1]) * np.diff(distance))]
        )

    def agreement(achieved, expected, boundary):
        """Return the primitive's error against the swing it carries."""
        return np.max(np.abs(achieved - expected)) / np.max(np.abs(expected - boundary))

    pressure = np.asarray(
        continuation.pressure(radius, psi_norm, BOUNDARY_PRESSURE, span)
    )
    expected = BOUNDARY_PRESSURE - span * sense * outward_integral(continuation.p_prime)
    assert agreement(pressure, expected, BOUNDARY_PRESSURE) < 1.0e-7

    squared = np.asarray(
        continuation.field_function_squared(psi_norm, BOUNDARY_FIELD_FUNCTION, span)
    )
    expected_squared = BOUNDARY_FIELD_FUNCTION**2 - 2.0 * span * sense * (
        outward_integral(continuation.ff_prime)
    )
    assert agreement(squared, expected_squared, BOUNDARY_FIELD_FUNCTION**2) < 1.0e-7

    # pressure falls away from the boundary into the scrape-off layer and rises
    # into the private-flux region, because that branch continues the core's own
    # trend to lower normalised flux; its low-pressure policy is therefore
    # bought with a short support rather than with a sign
    change = float(pressure[-1]) - BOUNDARY_PRESSURE
    assert change < 0.0 if domain is PlasmaDomain.COMMON_SOL else change > 0.0
    short = _policy(support=PRIVATE_SUPPORT).extend(core, domain)
    swing = abs(
        float(short.pressure(radius, psi_norm, BOUNDARY_PRESSURE, span)[-1])
        - BOUNDARY_PRESSURE
    )
    peak = float(jnp.max(jnp.abs(short.p_prime(psi_norm))))
    assert swing < 0.5 * abs(change)
    assert swing <= abs(span) * PRIVATE_SUPPORT * peak


def test_the_plasma_reaches_its_boundary_before_any_open_label_starts(diverted):
    """The core mask extends to the boundary; the open branch starts past it.

    A cell inside the boundary curve is plasma, so the closed test cuts at the
    boundary flux itself and not a fraction inside it. The check bites on this
    lattice: fourteen cells sit in the outermost thousandth of the flux span, a
    band a cut short of the boundary would have handed to the scrape-off label
    although every one of them lies within the plasma boundary curve.
    """
    _lattice, masks, _state = diverted
    psi_norm = np.asarray(masks.psi_norm)
    label = np.asarray(masks.label)
    inside = label != int(PlasmaDomain.EXCLUDED_MATERIAL)

    edge = inside & (psi_norm > 0.999) & (psi_norm <= 1.0)
    assert edge.sum() > 0, "the lattice must resolve the outermost flux band"
    assert not np.any(label[edge] == int(PlasmaDomain.COMMON_SOL))
    assert np.all(np.asarray(masks.psi_norm)[np.asarray(masks.common_sol)] > 1.0)
    assert np.max(psi_norm[np.asarray(masks.core)]) > 0.999

    # the private-flux branch is removed from the plasma by CONNECTIVITY, not
    # by its flux value: it sits below one exactly like the core it is cut from
    private = np.asarray(masks.private_flux)
    assert np.all(psi_norm[private] <= 1.0)
    assert np.max(psi_norm[private]) > 0.999
    assert not np.any(private & np.asarray(masks.core))


def test_the_private_flux_branch_is_labelled_and_reached_by_its_own_closure(diverted):
    """The map carries a private-flux pocket and the closure drives it."""
    lattice, masks, state = diverted
    counts = np.asarray(masks.cell_count())
    assert bool(state.diverted)
    assert counts[PlasmaDomain.CORE] > 50
    assert counts[PlasmaDomain.PRIVATE_FLUX] > 5
    assert counts[PlasmaDomain.COMMON_SOL] > 5

    core = _core()
    radius = jnp.asarray(lattice.node_radius)
    area = jnp.asarray(lattice.cell_area)
    source = ForwardSource(
        core=core,
        boundary_pressure=BOUNDARY_PRESSURE,
        boundary_field_function=BOUNDARY_FIELD_FUNCTION,
        private_flux=_policy(support=PRIVATE_SUPPORT).extend(
            core, PlasmaDomain.PRIVATE_FLUX
        ),
    )
    ledger = current_ledger(source.cell_current(radius, area, masks), masks)
    assert abs(float(ledger.private_flux)) > 0.0
    assert float(ledger.common_sol) == 0.0
    assert float(ledger.excluded_material) == 0.0


def test_varying_the_private_flux_policy_moves_only_its_own_current(diverted):
    """The two open branches are independent declarations, not one policy.

    The core and common-scrape-off rows are compared for EXACT equality: an
    independently declared private-flux continuation may not perturb them at
    all, not even at round-off, because it reaches a disjoint set of cells.
    """
    lattice, masks, _state = diverted
    core = _core()
    radius = jnp.asarray(lattice.node_radius)
    area = jnp.asarray(lattice.cell_area)

    def ledger_of(**closures):
        """Return the domain-split ledger of one declared source."""
        source = ForwardSource(
            core=core,
            boundary_pressure=BOUNDARY_PRESSURE,
            boundary_field_function=BOUNDARY_FIELD_FUNCTION,
            common_sol=_policy().extend(core, PlasmaDomain.COMMON_SOL),
            **closures,
        )
        return current_ledger(source.cell_current(radius, area, masks), masks)

    bare = ledger_of()
    narrow = ledger_of(
        private_flux=_policy(support=PRIVATE_SUPPORT).extend(
            core, PlasmaDomain.PRIVATE_FLUX
        )
    )
    wide = ledger_of(
        private_flux=_policy(support=4.0 * PRIVATE_SUPPORT).extend(
            core, PlasmaDomain.PRIVATE_FLUX
        )
    )
    decayed = ledger_of(
        private_flux=_policy(
            form=ContinuationForm.EXPONENTIAL_DECAY,
            support=PRIVATE_SUPPORT,
            decay_width=0.5 * PRIVATE_SUPPORT,
        ).extend(core, PlasmaDomain.PRIVATE_FLUX)
    )
    for varied in (narrow, wide, decayed):
        assert float(varied.core) == float(bare.core)
        assert float(varied.common_sol) == float(bare.common_sol)
        assert float(varied.excluded_material) == 0.0
    assert float(bare.private_flux) == 0.0
    for varied in (narrow, wide, decayed):
        assert abs(float(varied.private_flux)) > 0.0
    # shrinking the support is what makes the private region cold, and the
    # functional form is a second, independent lever on the same bound: the
    # polynomial is forced back to zero at it while the truncated exponential
    # still carries amplitude there, so the two differ on identical support
    assert abs(float(narrow.private_flux)) < abs(float(wide.private_flux))
    assert abs(float(decayed.private_flux) - float(narrow.private_flux)) > 0.1 * abs(
        float(narrow.private_flux)
    )
    assert abs(float(wide.private_flux)) < abs(float(bare.core))


def test_no_current_reaches_a_cell_beyond_the_support_or_outside_material(diverted):
    """Bounded support and the material mask are enforced on the labelled map."""
    lattice, masks, _state = diverted
    core = _core()
    radius = jnp.asarray(lattice.node_radius)
    area = jnp.asarray(lattice.cell_area)
    source = ForwardSource(
        core=core,
        boundary_field_function=BOUNDARY_FIELD_FUNCTION,
        common_sol=_policy().extend(core, PlasmaDomain.COMMON_SOL),
        private_flux=_policy(support=PRIVATE_SUPPORT).extend(
            core, PlasmaDomain.PRIVATE_FLUX
        ),
    )
    cell_current = np.asarray(source.cell_current(radius, area, masks))
    psi_norm = np.asarray(masks.psi_norm)
    for selection, bound in (
        (np.asarray(masks.common_sol), SUPPORT),
        (np.asarray(masks.private_flux), PRIVATE_SUPPORT),
    ):
        sense = 1.0 if bound == SUPPORT else -1.0
        beyond = selection & (sense * (psi_norm - 1.0) > bound)
        assert beyond.sum() > 0
        assert np.all(cell_current[beyond] == 0.0)
    assert np.all(cell_current[np.asarray(masks.excluded_material)] == 0.0)
    ledger = current_ledger(jnp.asarray(cell_current), masks)
    assert float(ledger.excluded_material) == 0.0
    np.testing.assert_allclose(
        ledger.total,
        float(ledger.core) + float(ledger.common_sol) + float(ledger.private_flux),
        rtol=1e-12,
    )


def test_the_continued_solve_reaches_its_fixed_point(continued):
    """The continued map converges under the shared ladder to a finite receipt."""
    assert float(continued.fixed_point.residual) < RESIDUAL_TOLERANCE
    assert bool(continued.finite.passed)
    counts = np.asarray(continued.domains.cell_count())
    assert counts[PlasmaDomain.CORE] > 100
    assert counts[PlasmaDomain.COMMON_SOL] > 50
    assert counts[PlasmaDomain.PRIVATE_FLUX] == 0
    # the continuation acts on cells the partition puts beyond the boundary,
    # so its domain starts exactly where the plasma stops
    psi_norm = np.asarray(continued.domains.psi_norm)
    assert np.all(psi_norm[np.asarray(continued.domains.common_sol)] > 1.0)
    assert np.all(psi_norm[np.asarray(continued.domains.core)] <= 1.0)
    record = continued.continuation.common_sol
    assert bool(continued.continuation.active)
    assert record.form_name == "hermite_polynomial"
    assert float(record.support) == SUPPORT
    assert not bool(continued.continuation.private_flux.active)
    assert continued.normalisation.policy_name == "absolute"
    assert not bool(continued.normalisation.rescaled)
    assert float(continued.normalisation.amplitude) == 1.0


def test_the_scrape_off_current_is_published_in_its_own_ledger_row(baseline, continued):
    """The continuation's current appears where a reader can separate it."""
    assert float(baseline.ledger.common_sol) == 0.0
    scrape_off = float(continued.ledger.common_sol)
    assert abs(scrape_off) > 0.0
    assert np.sign(scrape_off) == np.sign(float(continued.ledger.core))
    assert abs(scrape_off) < 0.05 * abs(float(continued.ledger.core))
    assert float(continued.ledger.private_flux) == 0.0
    assert float(continued.ledger.excluded_material) == 0.0
    np.testing.assert_allclose(
        continued.ledger.total,
        float(continued.ledger.core) + scrape_off,
        rtol=1e-9,
    )
    # the integral observations stay the confined plasma's, so a scrape-off
    # policy cannot move the denominator beta_p and l_i are read against
    np.testing.assert_allclose(
        continued.moments.plasma_current, continued.ledger.core, rtol=1e-12
    )


def test_the_continuation_moves_the_solution_by_a_resolvable_bounded_amount(
    baseline, continued
):
    """A declared continuation changes the equilibrium, and by how much is read.

    Every shift below is a shipped-value change a caller should expect from
    declaring the policy, which is why they are bounded from both sides: too
    small would mean the declaration did nothing, too large would mean the
    solve moved to another branch of the free-boundary map.
    """
    scale = jnp.max(jnp.abs(baseline.flux))
    shift = _relative(continued.flux, baseline.flux, scale)
    current = (
        float(continued.moments.plasma_current) / float(baseline.moments.plasma_current)
        - 1.0
    )
    axis = float(jnp.linalg.norm(continued.topology.axis - baseline.topology.axis))
    assert SHIFT_FLOOR < shift < SHIFT_CEILING
    assert 0.0 < abs(current) < 0.05
    assert 0.0 < axis < 0.05
    for name in ("poloidal_beta", "internal_inductance"):
        moved = abs(
            float(getattr(continued.moments, name))
            / float(getattr(baseline.moments, name))
            - 1.0
        )
        assert 0.0 < moved < 0.1, name
    assert not bool(baseline.topology.diverted)
    assert not bool(continued.topology.diverted)


def test_the_continued_equilibrium_meets_its_conservation_tolerances(
    baseline, continued
):
    """Force balance holds across the seam, not only inside the core.

    The residuals are read on the union of the declared domains, so the cells
    the continuation drives are checked too. Both physical residuals stay at
    the central-difference floor the core alone reaches, which is what
    distinguishes a self-consistent continuation from a merely smooth profile.
    """
    ledger = continued.conservation
    assert int(ledger.checked_cells) > int(baseline.conservation.checked_cells)
    assert float(ledger.relative_divergence_b) < DIVERGENCE_TOLERANCE
    assert float(ledger.relative_divergence_j) < DIVERGENCE_TOLERANCE
    assert float(ledger.relative_grad_shafranov) < GRAD_SHAFRANOV_TOLERANCE
    assert float(ledger.relative_force) < FORCE_TOLERANCE
    assert float(ledger.relative_grad_shafranov) < 2.0 * float(
        baseline.conservation.relative_grad_shafranov
    )


def test_no_current_appears_beyond_the_support_in_the_solved_map(continued):
    """The solved map carries open cells past the bound, and they are empty."""
    psi_norm = np.asarray(continued.domains.psi_norm)
    open_region = np.asarray(continued.domains.common_sol)
    beyond = open_region & (psi_norm - 1.0 > SUPPORT)
    assert beyond.sum() > 10
    cell_current = np.asarray(continued.cell_current)
    assert np.all(cell_current[beyond] == 0.0)
    assert np.all(cell_current[np.asarray(continued.domains.excluded_material)] == 0.0)
    inside_support = open_region & (psi_norm - 1.0 < SUPPORT)
    assert np.any(cell_current[inside_support] != 0.0)


def test_declaring_a_continuation_on_an_empty_domain_changes_nothing(
    machine, continued
):
    """A limited plasma has no private-flux cells, so its policy is inert."""
    build, seed = machine
    core = _core()
    profile = build(
        core,
        common_sol=_policy().extend(core, PlasmaDomain.COMMON_SOL),
        private_flux=_policy(support=PRIVATE_SUPPORT).extend(
            core, PlasmaDomain.PRIVATE_FLUX
        ),
    )
    result = profile.solve(seed, route="anderson", evaluations=EVALUATIONS)
    assert np.asarray(result.domains.cell_count())[PlasmaDomain.PRIVATE_FLUX] == 0
    assert float(result.ledger.private_flux) == 0.0
    assert bool(result.continuation.private_flux.active)
    np.testing.assert_array_equal(np.asarray(result.flux), np.asarray(continued.flux))


def test_the_static_path_is_unchanged_when_no_continuation_is_declared(machine):
    """Declaring the machinery on a null anchor reproduces the solve exactly.

    A core closure whose gradient and slope both vanish at the separatrix is
    continued by identically zero, so the declared solve and the undeclared one
    are the same physical problem. Their flux maps, current images and
    observations agree array for array, which is what says the continuation
    seam adds no arithmetic to a source that does not use it.

    The conservation ledger is the one receipt that legitimately differs: its
    checked set follows the DECLARED support, not the amplitude, so declaring
    a null continuation widens the region the residuals are read over while
    leaving the residuals themselves where they were.
    """
    build, seed = machine
    core = DomainProfile(
        p_prime=_edge_vanishing(3.0 * DRIVE * P_PRIME),
        ff_prime=_edge_vanishing(3.0 * DRIVE * FF_PRIME),
    )
    undeclared = build(core).solve(seed, route="anderson", evaluations=EVALUATIONS)
    declared = build(
        core,
        common_sol=_policy().extend(core, PlasmaDomain.COMMON_SOL),
        private_flux=_policy(support=PRIVATE_SUPPORT).extend(
            core, PlasmaDomain.PRIVATE_FLUX
        ),
    ).solve(seed, route="anderson", evaluations=EVALUATIONS)

    assert float(undeclared.fixed_point.residual) < RESIDUAL_TOLERANCE
    np.testing.assert_array_equal(
        np.asarray(declared.flux), np.asarray(undeclared.flux)
    )
    np.testing.assert_array_equal(
        np.asarray(declared.cell_current), np.asarray(undeclared.cell_current)
    )
    np.testing.assert_array_equal(
        np.asarray(declared.domains.label), np.asarray(undeclared.domains.label)
    )
    np.testing.assert_array_equal(
        np.asarray(declared.moments.stack()), np.asarray(undeclared.moments.stack())
    )
    assert float(declared.ledger.common_sol) == 0.0
    assert float(declared.ledger.private_flux) == 0.0
    assert not bool(undeclared.continuation.active)
    assert bool(declared.continuation.active)
    assert int(declared.conservation.checked_cells) > int(
        undeclared.conservation.checked_cells
    )
    assert float(declared.conservation.relative_grad_shafranov) == float(
        undeclared.conservation.relative_grad_shafranov
    )


def test_the_source_evaluation_is_the_declared_gradients_unscaled(machine):
    """The domain-selected image is the same expression it always was."""
    build, seed = machine
    core = _core()
    profile = build(core, common_sol=_policy().extend(core, PlasmaDomain.COMMON_SOL))
    masks = profile.operator.read(seed)[0]
    radius = jnp.asarray(profile.lattice.node_radius)
    area = jnp.asarray(profile.lattice.cell_area)
    density = core.current_density(radius, jnp.where(masks.core, masks.psi_norm, 0.0))
    open_density = toroidal_current_density(
        radius,
        profile.source.common_sol.p_prime(masks.psi_norm),
        profile.source.common_sol.ff_prime(masks.psi_norm),
    )
    expected = jnp.where(masks.core, density, 0.0) + jnp.where(
        masks.common_sol, open_density, 0.0
    )
    np.testing.assert_array_equal(
        np.asarray(profile.source.cell_current(radius, area, masks)),
        np.asarray(expected * area),
    )


def test_the_host_and_traced_routes_agree_on_the_continued_map(machine, continued):
    """The eager and accelerated routes drive one continued source identically."""
    build, seed = machine
    core = _core()
    profile = build(core, common_sol=_policy().extend(core, PlasmaDomain.COMMON_SOL))
    host = profile.solve(seed, route="host", evaluations=TRANSIENT, tolerance=0.0)
    traced = profile.solve(seed, route="picard", evaluations=TRANSIENT)
    scale = jnp.max(jnp.abs(continued.flux))
    assert _relative(host.flux, traced.flux, scale) < PARITY_TOLERANCE
    np.testing.assert_allclose(
        np.asarray(host.fixed_point.trace),
        np.asarray(traced.fixed_point.trace),
        rtol=1e-6,
    )
    relaxed = profile.solve(seed, route="picard", evaluations=2 * EVALUATIONS)
    assert float(relaxed.fixed_point.residual) < RESIDUAL_TOLERANCE
    assert _relative(relaxed.flux, continued.flux, scale) < PARITY_TOLERANCE
    np.testing.assert_allclose(
        relaxed.ledger.common_sol, continued.ledger.common_sol, rtol=1e-4
    )


def test_the_batched_continued_solve_matches_the_single_slice(machine, continued):
    """A jitted vmap over seeds carries the continuation and its receipt."""
    build, seed = machine
    core = _core()
    profile = build(core, common_sol=_policy().extend(core, PlasmaDomain.COMMON_SOL))
    seeds = jnp.stack([seed, 0.98 * seed])
    ensemble = jax.jit(
        lambda state: profile.solve_batch(
            state, route="anderson", evaluations=EVALUATIONS
        )
    )(seeds)
    assert ensemble.flux.shape == (2, seed.size)
    assert bool(jnp.all(ensemble.finite.passed))
    scale = jnp.max(jnp.abs(continued.flux))
    assert _relative(ensemble.flux[0], continued.flux, scale) < PARITY_TOLERANCE
    np.testing.assert_allclose(
        np.asarray(ensemble.ledger.common_sol),
        float(continued.ledger.common_sol),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(ensemble.continuation.common_sol.support), SUPPORT
    )


def test_the_exponential_family_publishes_the_amplitude_it_truncates(machine):
    """A truncated continuation reports the step it takes to zero."""
    build, seed = machine
    core = _core()
    policy = _policy(form=ContinuationForm.EXPONENTIAL_DECAY, decay_width=DECAY_WIDTH)
    profile = build(core, common_sol=policy.extend(core, PlasmaDomain.COMMON_SOL))
    result = profile.solve(seed, route="anderson", evaluations=EVALUATIONS)
    record = result.continuation.common_sol
    assert record.form_name == "exponential_decay"
    assert float(record.decay_width) == DECAY_WIDTH
    assert 1.0e-3 < float(record.truncated_fraction) < 0.5
    assert float(result.fixed_point.residual) < RESIDUAL_TOLERANCE
    assert abs(float(result.ledger.common_sol)) > 0.0
    assert float(result.conservation.relative_grad_shafranov) < (
        GRAD_SHAFRANOV_TOLERANCE
    )
    # the discarded amplitude is a property of the declaration, so shortening
    # the width against the same bound removes the step entirely
    tight = _policy(
        form=ContinuationForm.EXPONENTIAL_DECAY, decay_width=0.2 * DECAY_WIDTH
    ).extend(core, PlasmaDomain.COMMON_SOL)
    assert tight.truncated_fraction < 0.1 * float(record.truncated_fraction)


def test_the_anchor_check_tolerance_is_far_above_its_own_truncation():
    """The kink check cannot fire on a smooth profile's differencing error."""
    curved = _curved(2.0 * DRIVE * P_PRIME)
    for gradient in (_core().p_prime, curved):
        value, slope = separatrix_derivatives(gradient, 2)
        measured = _one_sided_slope(gradient, -1.0, step=1.0e-3)
        assert abs(float(slope) - measured) < ANCHOR_TOLERANCE * abs(float(slope))
        assert float(value) != 0.0


def test_the_package_publishes_the_continuation_and_geometry_names():
    """Every name the equilibrium package advertises resolves on demand.

    The exports are loaded from their module on first attribute access, so an
    entry naming the wrong module fails only when someone reaches for it. This
    reaches for all of them.
    """
    import nova.equilibrium as equilibrium

    assert not [name for name in equilibrium.__all__ if not hasattr(equilibrium, name)]
    for name in (
        "ContinuationForm",
        "ContinuationLedger",
        "ContinuationRecord",
        "ContinuedDomainProfile",
        "SeparatrixContinuation",
        "SeparatrixContinuity",
        "SeparatrixJumpError",
        "FluxSurfaceGeometry",
        "GridMotion",
        "SurfaceGeometryError",
        "source_field_function",
    ):
        assert name in equilibrium.__all__, name
    assert equilibrium.SeparatrixContinuation is SeparatrixContinuation
    assert equilibrium.ContinuationForm is ContinuationForm
    assert (
        equilibrium.FluxSurfaceGeometry.__module__
        == "nova.equilibrium.flux_surface_geometry"
    )


if __name__ == "__main__":
    pytest.main([__file__])
