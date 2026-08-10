r"""Toroidal-rotation closure of the forward equilibrium solve.

The analytic target is the banked Maschke-Perrin/rotating-Solov'ev family in
:mod:`tests.rotating_equilibrium_references`, which is exact at every Mach
number and imports nothing from the package under test. Two conversions stand
between it and Nova and both are done explicitly here rather than absorbed
into a tolerance.

Flux convention
    The references carry the poloidal flux per radian :math:`\psi`, maximal
    on the axis and zero on the plasma boundary; Nova carries the total
    poloidal flux :math:`\Phi = 2 \pi \psi` and a normalised flux that runs
    from zero on the axis to one on the boundary. The pair

    .. math::
        \Phi = 2 \pi \psi, \qquad
        \psi_N = 1 - \psi / \psi_a, \qquad
        \Phi_b - \Phi_a = -2 \pi \psi_a

    fixes everything else: a reference gradient with respect to :math:`\psi`
    becomes a Nova gradient with respect to the negated total flux by
    dividing by :math:`-2 \pi`. Under that map the two toroidal current
    densities are the SAME number with the SAME sign, so the reproduction
    tests below are read at a floating-point tolerance rather than a physical
    one — any factor of :math:`2\pi` or sign slip is a failure of several
    orders, not a near miss.

Species convention
    The references define the mean particle mass by
    :math:`\rho = \bar{m} p / T` and use a pure deuterium plasma. Nova takes
    the same mass as a declared field, so it is passed through explicitly and
    a test pins what changing it does: a quasineutral reading halves it and
    halves the centrifugal exponent.

The reference module's own tests own the properties of the analytic family
(the exactness of its Grad-Shafranov residual, the linear growth of the
Shafranov shift in the rotation parameter, the Mach-independence of the axis
and of :math:`q` on axis). Those are not repeated here. What is tested here
is that Nova's source reproduces that family, that its force balance closes
with the centrifugal term, and that a free-boundary solve driven by it lands
back on the analytic member with those invariants intact.
"""

from __future__ import annotations

import dataclasses
import math

import matplotlib.path
import numpy as np
import pytest

from nova.utilities.importmanager import skip_import
from tests.rotating_equilibrium_references import (
    RotatingEquilibrium,
    delta_star_finite_difference,
    interior_sample,
    reference_cases,
    rotating_equilibrium,
)

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.convention import delta_star_from_current_density
    from nova.equilibrium.domain import DomainMasks, PlasmaDomain
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.rotation import IsothermalRotation, RotatingDomainProfile
    from nova.equilibrium.source import DomainProfile, ForwardSource, RotationClosure
    from nova.jax.config import configure_dtypes

#: Wb of total poloidal flux per Wb/rad of flux per radian.
TOTAL_FLUX = 2.0 * np.pi

CASE_NAMES = tuple(reference_cases())

#: Agreement demanded where Nova and the references evaluate the same closed
#: form through different algebra: a few round-off amplifications of an
#: exponential, and nothing physical in between.
SOURCE_TOLERANCE = 1.0e-12
#: Agreement of the source with the elliptic operator of the analytic flux,
#: read through a fourth-order stencil at a fixed fraction of the minor
#: radius, so the floor is the stencil truncation rather than the closure.
OPERATOR_TOLERANCE = 1.0e-7
#: Central-difference checks of the closure's own derivatives.
DERIVATIVE_TOLERANCE = 1.0e-7
#: Cancellation demanded of the uniform-Mach branch, where the two terms of
#: the exponent gradient are equal and opposite by construction.
CANCELLATION_TOLERANCE = 1.0e-14

# ----------------------------------------------------------------------
# the analytic family as a Nova source
# ----------------------------------------------------------------------


def reference_flux(case: RotatingEquilibrium, psi_norm):
    """Return the flux per radian [Wb/rad] at one Nova normalised flux."""
    return case.axis_flux * (1.0 - np.asarray(psi_norm, dtype=float))


def normalised_flux(case: RotatingEquilibrium, flux):
    """Return the Nova normalised flux of one reference flux per radian."""
    return 1.0 - np.asarray(flux, dtype=float) / case.axis_flux


def flux_span(case: RotatingEquilibrium) -> float:
    """Return ``Phi_b - Phi_a`` [Wb]: the boundary of the family is zero flux."""
    return -TOTAL_FLUX * case.axis_flux


def rotation_closure(case: RotatingEquilibrium) -> IsothermalRotation:
    """Return the Nova isothermal closure of one reference member.

    The member's linear temperature and its square-root angular frequency are
    rewritten in normalised flux and their flux gradients are taken in Nova's
    negated-total-flux sense. They are written out in ``jax.numpy`` rather
    than delegated to the reference methods so the closure traces, which the
    reference module — deliberately numpy and scipy only — does not.
    """
    axis, edge = case.axis_temperature, case.boundary_temperature
    parameter, mass = case.rotation_parameter, case.mean_particle_mass
    # -dT/dPhi of a temperature linear in the flux, over the flux span
    temperature_slope = -(axis - edge) / (TOTAL_FLUX * case.axis_flux)

    def temperature(psi_norm):
        """Return the surface temperature [J], linear in the flux label.

        Held at the boundary value outside the plasma. The closure is declared
        on the core and the solve discards it elsewhere, but a temperature
        continued linearly would pass through zero a little beyond the
        separatrix and take the angular frequency imaginary with it.
        """
        return edge + (axis - edge) * jnp.clip(1.0 - jnp.asarray(psi_norm), 0.0, 1.0)

    def angular_frequency(psi_norm):
        """Return ``Omega = sqrt(2 theta T / mbar)`` [rad/s]."""
        return jnp.sqrt(2.0 * parameter * temperature(psi_norm) / mass)

    def temperature_gradient(psi_norm):
        """Return the constant ``-dT/dPhi`` [J/Wb]."""
        return jnp.full_like(
            jnp.asarray(psi_norm, dtype=jnp.float64), temperature_slope
        )

    def angular_frequency_gradient(psi_norm):
        """Return ``-dOmega/dPhi``, the chain rule through the temperature."""
        frequency = angular_frequency(psi_norm)
        safe = jnp.where(frequency > 0.0, frequency, 1.0)
        return jnp.where(
            frequency > 0.0,
            parameter * temperature_gradient(psi_norm) / (mass * safe),
            0.0,
        )

    return IsothermalRotation(
        temperature=temperature,
        angular_frequency=angular_frequency,
        temperature_gradient=temperature_gradient,
        angular_frequency_gradient=angular_frequency_gradient,
        mean_particle_mass=mass,
        reference_radius=case.major_radius,
    )


def flux_gradients(case: RotatingEquilibrium):
    """Return the member's ``(p_prime, ff_prime)`` in Nova's flux sense.

    Both are constant across this family, and both carry the sign of the
    conversion: a positive reference gradient with respect to the flux per
    radian is a negative Nova gradient with respect to the negated total
    flux, which is what drives a positive toroidal current.
    """
    pressure_gradient = -case.pressure_flux_gradient / TOTAL_FLUX
    field_gradient = -case.f_f_prime / TOTAL_FLUX

    def p_prime(psi_norm):
        """Return the constant reference-radius pressure gradient [Pa/Wb]."""
        return jnp.full_like(
            jnp.asarray(psi_norm, dtype=jnp.float64), pressure_gradient
        )

    def ff_prime(psi_norm):
        """Return the constant diamagnetic gradient [T^2 m^2/Wb]."""
        return jnp.full_like(jnp.asarray(psi_norm, dtype=jnp.float64), field_gradient)

    return p_prime, ff_prime


def rotating_profile(case: RotatingEquilibrium) -> RotatingDomainProfile:
    """Return the Nova rotating core profile of one reference member."""
    p_prime, ff_prime = flux_gradients(case)
    axis_pressure = case.axis_pressure

    def reference_pressure(psi_norm):
        """Return ``p_0(psi_N)``, linear and vanishing at the boundary."""
        return axis_pressure * (1.0 - jnp.asarray(psi_norm))

    return RotatingDomainProfile(
        p_prime=p_prime,
        ff_prime=ff_prime,
        reference_pressure=reference_pressure,
        rotation=rotation_closure(case),
    )


@pytest.fixture(scope="module", autouse=True)
def device_precision():
    """Publish the fp64 device policy the rotation contract is read in."""
    configure_dtypes()


def sample(case: RotatingEquilibrium):
    """Return interior points and their Nova normalised flux."""
    radius, height = interior_sample(case)
    return radius, height, normalised_flux(case, case.flux(radius, height))


# ----------------------------------------------------------------------
# the source reproduces the analytic family
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", CASE_NAMES)
def test_the_rotating_source_reproduces_the_reference_current_density(name):
    """Nova's rotating source is the reference source, at every Mach number."""
    case = reference_cases()[name]
    radius, height, psi_norm = sample(case)
    density = np.asarray(
        rotating_profile(case).current_density(
            jnp.asarray(radius), jnp.asarray(psi_norm)
        )
    )
    expected = case.toroidal_current_density(radius, height)
    assert np.all(expected > 0.0)
    np.testing.assert_allclose(density, expected, rtol=SOURCE_TOLERANCE)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_the_rotating_source_matches_the_operator_of_the_analytic_flux(name):
    """The source reproduces the elliptic operator the analytic flux carries.

    Only the flux map is sampled on the right-hand side, so agreement pins
    the whole chain — the total-flux factor, both signs, the centrifugal
    factor and the toroidal-field term — against a quantity that never sees
    Nova's algebra.
    """
    case = reference_cases()[name]
    radius, height, psi_norm = sample(case)
    density = rotating_profile(case).current_density(
        jnp.asarray(radius), jnp.asarray(psi_norm)
    )
    elliptic = np.asarray(delta_star_from_current_density(radius, density))
    spacing = case.minor_radius / 40.0
    expected = TOTAL_FLUX * delta_star_finite_difference(case, radius, height, spacing)
    error = np.max(np.abs(elliptic - expected)) / np.max(np.abs(expected))
    assert error < OPERATOR_TOLERANCE


@pytest.mark.parametrize("name", CASE_NAMES)
def test_the_closure_reproduces_the_reference_thermodynamics(name):
    """The temperature and angular frequency handed to Nova are the member's."""
    case = reference_cases()[name]
    closure = rotation_closure(case)
    psi_norm = np.linspace(0.0, 1.0, 11)
    flux = reference_flux(case, psi_norm)
    np.testing.assert_allclose(
        np.asarray(closure.temperature(jnp.asarray(psi_norm))),
        case.temperature(flux),
        rtol=SOURCE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(closure.angular_frequency(jnp.asarray(psi_norm))),
        case.angular_frequency(flux),
        rtol=SOURCE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(closure.centrifugal_exponent(jnp.asarray(psi_norm))),
        case.rotation_parameter,
        rtol=SOURCE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(closure.thermal_mach_number(jnp.asarray(psi_norm))),
        case.thermal_mach_number,
        rtol=SOURCE_TOLERANCE,
    )


@pytest.mark.parametrize("name", CASE_NAMES)
def test_a_uniform_mach_closure_cancels_its_exponent_gradient(name):
    """An angular frequency tracking the square root of the temperature has none.

    Both terms of the exponent gradient are large and individually non-zero —
    the temperature falls by two orders of magnitude across these members —
    so the cancellation is a statement about the chain rule and not about
    small numbers.
    """
    case = reference_cases()[name]
    closure = rotation_closure(case)
    psi_norm = jnp.asarray(np.linspace(0.0, 1.0, 11))
    gradient = np.asarray(closure.centrifugal_exponent_gradient(psi_norm))
    scale = np.max(
        np.abs(
            np.asarray(closure.centrifugal_exponent(psi_norm))
            * np.asarray(closure.temperature_gradient(psi_norm))
            / np.asarray(closure.temperature(psi_norm))
        )
    )
    assert scale > 0.0
    assert np.max(np.abs(gradient)) < CANCELLATION_TOLERANCE * scale


# ----------------------------------------------------------------------
# the isothermal closure itself
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", CASE_NAMES)
def test_the_isothermal_factor_pins_the_radial_pressure_variation(name):
    """``d ln p / dR`` at fixed flux is ``2 theta R`` and nothing else."""
    case = reference_cases()[name]
    core = rotating_profile(case)
    span = flux_span(case)
    psi_norm = jnp.asarray([0.05, 0.4, 0.9])
    radius = jnp.asarray([0.85, 1.0, 1.15]) * case.major_radius
    step = 1.0e-5 * case.major_radius

    def pressure(shift):
        """Return the closure pressure at a shifted major radius."""
        return core.pressure(radius + shift, psi_norm, 0.0, span)

    numeric = np.asarray(
        (jnp.log(pressure(step)) - jnp.log(pressure(-step))) / (2.0 * step)
    )
    analytic = np.asarray(core.rotation.log_pressure_radial_gradient(radius, psi_norm))
    np.testing.assert_allclose(
        analytic, 2.0 * case.rotation_parameter * np.asarray(radius), rtol=1e-14
    )
    np.testing.assert_allclose(numeric, analytic, rtol=DERIVATIVE_TOLERANCE)
    np.testing.assert_allclose(
        np.asarray(pressure(0.0)),
        case.pressure_at(reference_flux(case, psi_norm), np.asarray(radius)),
        rtol=SOURCE_TOLERANCE,
    )


@pytest.mark.parametrize("name", CASE_NAMES)
def test_the_body_force_is_the_radial_balance_of_the_closure(name):
    """The declared body force is exactly the pressure gradient it balances.

    Radial balance at fixed flux reads ``dp/dR = rho R Omega^2``, so the
    force receipt closes only if the body force the profile publishes equals
    a difference of its own pressure. Without the centrifugal term the
    residual would be the whole outboard pile-up.
    """
    case = reference_cases()[name]
    core = rotating_profile(case)
    span = flux_span(case)
    psi_norm = jnp.asarray([0.05, 0.4, 0.9])
    radius = jnp.asarray([0.85, 1.0, 1.15]) * case.major_radius
    step = 1.0e-5 * case.major_radius
    numeric = np.asarray(
        (
            core.pressure(radius + step, psi_norm, 0.0, span)
            - core.pressure(radius - step, psi_norm, 0.0, span)
        )
        / (2.0 * step)
    )
    pressure = core.pressure(radius, psi_norm, 0.0, span)
    body_force = np.asarray(core.radial_body_force(radius, psi_norm, pressure))
    assert np.all(body_force > 0.0)
    np.testing.assert_allclose(numeric, body_force, rtol=DERIVATIVE_TOLERANCE)
    # the same force written from the reference mass density and frequency
    np.testing.assert_allclose(
        body_force,
        case.mass_density_at(reference_flux(case, psi_norm), np.asarray(radius))
        * np.asarray(radius)
        * case.angular_frequency(reference_flux(case, psi_norm)) ** 2,
        rtol=SOURCE_TOLERANCE,
    )


def test_the_mean_particle_mass_convention_is_declared_not_assumed():
    """A quasineutral reading of the same plasma halves the exponent."""
    case = reference_cases()["moderate-rotation-conventional"]
    closure = rotation_closure(case)
    quasineutral = dataclasses.replace(
        closure, mean_particle_mass=0.5 * closure.mean_particle_mass
    )
    psi_norm = jnp.asarray(np.linspace(0.0, 1.0, 5))
    radius = jnp.asarray(1.1 * case.major_radius)
    np.testing.assert_allclose(
        np.asarray(quasineutral.centrifugal_exponent(psi_norm)),
        0.5 * np.asarray(closure.centrifugal_exponent(psi_norm)),
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(quasineutral.centrifugal_factor(radius, psi_norm)),
        np.sqrt(np.asarray(closure.centrifugal_factor(radius, psi_norm))),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(quasineutral.thermal_mach_number(psi_norm)),
        np.asarray(closure.thermal_mach_number(psi_norm)) / math.sqrt(2.0),
        rtol=1e-14,
    )


def test_the_exponent_gradient_term_reaches_the_source():
    """A rotation that does not track the temperature adds a second term.

    The exactly solvable branch has a uniform thermal Mach number and no
    exponent gradient at all, so this member is manufactured instead: a
    constant angular frequency over a falling temperature makes ``theta`` a
    genuine flux function. The source is then checked against a difference of
    the closure's own pressure at fixed major radius, which is the definition
    the Grad-Shafranov source is written from.
    """
    case = reference_cases()["moderate-rotation-conventional"]
    span = flux_span(case)
    closure = rotation_closure(case)
    frequency = float(closure.angular_frequency(jnp.asarray(0.0)))

    def angular_frequency(psi_norm):
        """Return a rigid rotation, uniform across the flux surfaces."""
        return jnp.full_like(jnp.asarray(psi_norm, dtype=jnp.float64), frequency)

    def angular_frequency_gradient(psi_norm):
        """Return the vanishing flux gradient of a rigid rotation."""
        return jnp.zeros_like(jnp.asarray(psi_norm, dtype=jnp.float64))

    rigid = dataclasses.replace(
        closure,
        angular_frequency=angular_frequency,
        angular_frequency_gradient=angular_frequency_gradient,
    )
    p_prime, ff_prime = flux_gradients(case)
    axis_pressure = case.axis_pressure

    def reference_pressure(psi_norm):
        """Return ``p_0(psi_N)``, linear and vanishing at the boundary."""
        return axis_pressure * (1.0 - jnp.asarray(psi_norm))

    core = RotatingDomainProfile(
        p_prime=p_prime,
        ff_prime=ff_prime,
        reference_pressure=reference_pressure,
        rotation=rigid,
    )
    psi_norm = jnp.asarray([0.15, 0.4, 0.65])
    radius = jnp.asarray([0.9, 1.0, 1.12]) * case.major_radius
    step = 1.0e-6
    numeric = -np.asarray(
        (
            core.pressure(radius, psi_norm + step, 0.0, span)
            - core.pressure(radius, psi_norm - step, 0.0, span)
        )
        / (2.0 * step * span)
    )
    analytic = np.asarray(core.pressure_gradient(radius, psi_norm))
    np.testing.assert_allclose(numeric, analytic, rtol=1.0e-6)

    # the term is not a rounding correction: dropping it is a per-cent-scale
    # error in the drive, growing with distance from the reference radius
    without = np.asarray(
        core.p_prime(psi_norm) * core.rotation.centrifugal_factor(radius, psi_norm)
    )
    assert np.max(np.abs(analytic - without) / np.abs(analytic)) > 0.01


# ----------------------------------------------------------------------
# the static limit
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", CASE_NAMES)
def test_the_static_limit_recovers_the_static_source_exactly(name):
    """A closure with no rotation reproduces the static source bit for bit.

    The exponent and its gradient are exactly zero, so the two extra factors
    are an exact one and an exact zero and the declared gradient reaches the
    current density unchanged — an equality, not a tolerance.
    """
    case = reference_cases()[name].static_limit()
    p_prime, ff_prime = flux_gradients(case)
    rotating = rotating_profile(case)
    static = DomainProfile(p_prime=p_prime, ff_prime=ff_prime)
    radius, height, psi_norm = sample(case)
    radius, psi_norm = jnp.asarray(radius), jnp.asarray(psi_norm)

    np.testing.assert_array_equal(
        np.asarray(rotating.current_density(radius, psi_norm)),
        np.asarray(static.current_density(radius, psi_norm)),
    )
    np.testing.assert_allclose(
        np.asarray(rotating.current_density(radius, psi_norm)),
        case.toroidal_current_density(np.asarray(radius), height),
        rtol=SOURCE_TOLERANCE,
    )
    np.testing.assert_array_equal(
        np.asarray(
            rotating.radial_body_force(
                radius, psi_norm, rotating.pressure(radius, psi_norm, 0.0, 1.0)
            )
        ),
        np.zeros(radius.size),
    )
    # the pressure primitive the closure carries and the one the static
    # profile integrates back from its gradient are the same function
    span = flux_span(case)
    np.testing.assert_allclose(
        np.asarray(rotating.pressure(radius, psi_norm, 0.0, span)),
        np.asarray(static.pressure(radius, psi_norm, 0.0, span)),
        rtol=1.0e-12,
    )


def test_a_static_source_publishes_a_static_rotation_receipt():
    """A profile that declared no rotation says so, and declares no conventions."""
    case = reference_cases()["weak-rotation-reactor"]
    p_prime, ff_prime = flux_gradients(case)
    source = ForwardSource(core=DomainProfile(p_prime=p_prime, ff_prime=ff_prime))
    masks = _probe_masks(np.linspace(0.0, 1.0, 5))
    record = source.rotation_record(jnp.full(5, case.major_radius), masks)
    assert record.closure_name == "static"
    assert not bool(record.active)
    assert not np.isfinite(float(record.reference_radius))
    assert not np.isfinite(float(record.mean_particle_mass))
    assert float(record.axis_mach_number) == 0.0
    assert float(record.minimum_centrifugal_factor) == 1.0
    assert float(record.maximum_centrifugal_factor) == 1.0


def _probe_masks(psi_norm):
    """Return an all-core domain label set over a normalised flux probe."""
    return DomainMasks(
        label=jnp.full(len(psi_norm), int(PlasmaDomain.CORE), dtype=jnp.int8),
        psi_norm=jnp.asarray(psi_norm),
    )


# ----------------------------------------------------------------------
# typed validation of the closure
# ----------------------------------------------------------------------


def test_the_closure_refuses_a_sampled_image_in_place_of_a_flux_function():
    """A temperature profile is a callable of flux, not a measurement array."""
    closure = rotation_closure(reference_cases()["moderate-rotation-conventional"])
    with pytest.raises(TypeError, match="callable flux function"):
        dataclasses.replace(closure, temperature=np.linspace(1.0, 2.0, 8))
    with pytest.raises(TypeError, match="callable flux function"):
        dataclasses.replace(closure, angular_frequency=[1.0, 2.0])


def test_the_closure_refuses_an_undeclared_species_convention():
    """A mean particle mass and a reference radius have no defensible default."""
    closure = rotation_closure(reference_cases()["moderate-rotation-conventional"])
    with pytest.raises(ValueError, match="mean_particle_mass must be positive"):
        dataclasses.replace(closure, mean_particle_mass=0.0)
    with pytest.raises(ValueError, match="reference_radius must be positive"):
        dataclasses.replace(closure, reference_radius=-1.0)


def test_the_rotating_profile_refuses_a_closure_it_cannot_read():
    """The rotation field is a typed closure, not an arbitrary namespace."""
    case = reference_cases()["moderate-rotation-conventional"]
    p_prime, ff_prime = flux_gradients(case)
    with pytest.raises(TypeError, match="IsothermalRotation"):
        RotatingDomainProfile(
            p_prime=p_prime,
            ff_prime=ff_prime,
            reference_pressure=lambda psi_norm: psi_norm,
            rotation=object(),
        )
    with pytest.raises(TypeError, match="callable flux function"):
        RotatingDomainProfile(
            p_prime=p_prime,
            ff_prime=ff_prime,
            reference_pressure=np.linspace(0.0, 1.0, 4),
            rotation=rotation_closure(case),
        )


def test_the_source_refuses_a_boundary_pressure_the_closure_contradicts():
    """The declared boundary primitive is the one at the reference radius."""
    case = reference_cases()["moderate-rotation-conventional"]
    core = rotating_profile(case)
    with pytest.raises(ValueError, match="reference-radius pressure"):
        ForwardSource(core=core, boundary_pressure=1.0e4)
    source = ForwardSource(core=core, boundary_pressure=0.0)
    record = source.rotation_record(
        jnp.full(5, case.major_radius), _probe_masks(np.linspace(0.0, 1.0, 5))
    )
    assert record.closure_name == "isothermal_surface"
    assert bool(record.active)
    assert int(record.closure) == int(RotationClosure.ISOTHERMAL_SURFACE)
    assert float(record.reference_radius) == case.major_radius
    assert float(record.mean_particle_mass) == case.mean_particle_mass
    np.testing.assert_allclose(
        float(record.axis_mach_number), case.thermal_mach_number, rtol=1e-14
    )
    # every probe sits at the reference radius, where the factor is one
    assert float(record.minimum_centrifugal_factor) == pytest.approx(1.0, rel=1e-14)


# ----------------------------------------------------------------------
# tracing, batching and derivatives
# ----------------------------------------------------------------------


def test_the_rotating_source_traces_and_batches_unchanged():
    """The eager evaluation, the jitted one and a vmapped batch agree."""
    case = reference_cases()["strong-rotation-compact"]
    core = rotating_profile(case)
    radius, _height, psi_norm = sample(case)
    radius, psi_norm = jnp.asarray(radius), jnp.asarray(psi_norm)
    eager = np.asarray(core.current_density(radius, psi_norm))
    traced = np.asarray(jax.jit(core.current_density)(radius, psi_norm))
    np.testing.assert_allclose(traced, eager, rtol=1e-15)

    batch = jnp.stack([psi_norm, 0.5 * psi_norm])
    mapped = jax.vmap(core.current_density, in_axes=(None, 0))(radius, batch)
    assert mapped.shape == batch.shape
    np.testing.assert_allclose(np.asarray(mapped[0]), eager, rtol=1e-15)
    np.testing.assert_allclose(
        np.asarray(mapped[1]),
        np.asarray(core.current_density(radius, 0.5 * psi_norm)),
        rtol=1e-15,
    )


def test_the_source_carries_a_derivative_in_the_rotation():
    """The drive differentiates through the closure to the Mach number.

    The rotation reaches the source only through the angular frequency, so a
    derivative with respect to a thermal Mach number exercises the exponent,
    its gradient and the exponential together.
    """
    case = reference_cases()["moderate-rotation-conventional"]
    radius, _height, psi_norm = sample(case)
    radius, psi_norm = jnp.asarray(radius), jnp.asarray(psi_norm)
    base = rotation_closure(case)
    profile = rotating_profile(case)
    mass = case.mean_particle_mass

    def drive(mach):
        """Return the integrated current density at one thermal Mach number."""
        parameter = mach**2 / case.major_radius**2

        def angular_frequency(label):
            """Return the frequency a traced Mach number implies."""
            return jnp.sqrt(2.0 * parameter * base.temperature(label) / mass)

        def angular_frequency_gradient(label):
            """Return its flux gradient through the temperature."""
            return (
                parameter
                * base.temperature_gradient(label)
                / (mass * angular_frequency(label))
            )

        core = dataclasses.replace(
            profile,
            rotation=dataclasses.replace(
                base,
                angular_frequency=angular_frequency,
                angular_frequency_gradient=angular_frequency_gradient,
            ),
        )
        return jnp.sum(core.current_density(radius, psi_norm))

    mach = case.thermal_mach_number
    gradient = float(jax.grad(drive)(mach))
    step = 1.0e-5 * mach
    numeric = float((drive(mach + step) - drive(mach - step)) / (2.0 * step))
    assert abs(gradient) > 0.0
    assert abs(gradient - numeric) / abs(numeric) < 1.0e-6


# ----------------------------------------------------------------------
# the free-boundary rotating solve
# ----------------------------------------------------------------------

#: Thermal Mach numbers the solve ladder is run at, spanning the banked
#: reference range with the static member as its first rung.
MACH_LADDER = (0.0, 0.10, 0.35, 0.90)
CONDUCTORS = 16
EVALUATIONS = 240
NODES = 29

#: Pre-registered solve tolerances. The fixed-point residual is a numerical
#: contract; the Grad-Shafranov and force residuals are read with central
#: differences on a map produced by Green operators, so their floor is the
#: coarser of the two discretisations — the analytic pins of the closure live
#: with the source tests above, not here.
RESIDUAL_TOLERANCE = 1.0e-6
GRAD_SHAFRANOV_TOLERANCE = 0.1
FORCE_TOLERANCE = 0.1
DIVERGENCE_TOLERANCE = 1.0e-12
#: Displacement of the solved axis from the analytic one, as a fraction of
#: the node spacing: the topology read locates the axis by a stencil fit on
#: the same lattice the current image is quantised on.
AXIS_TOLERANCE = 0.5
#: Agreement of the solved boundary displacement with the analytic one, as a
#: fraction of the node spacing.
BOUNDARY_TOLERANCE = 0.5
#: Displacement below which a boundary motion read from an interpolated
#: midplane crossing is not resolved, as a fraction of the node spacing.
RESOLUTION_FLOOR = 0.1


def solve_member(mach: float) -> RotatingEquilibrium:
    """Return the analytic member one rung of the solve ladder is built on.

    The banked reference cases carry machine scales chosen for the analytic
    checks; this member keeps the same closed form at a size and aspect ratio
    a lattice of a few hundred nodes resolves, and the rung changes the
    rotation parameter alone so the pressure and diamagnetic drives, the axis
    flux and therefore the whole comparison are held fixed.
    """
    base = rotating_equilibrium(
        name="solve-ladder",
        major_radius=1.0,
        outboard_radius=1.25,
        half_height=0.30,
        vacuum_field=2.0,
        axis_pressure=1.0e5,
        thermal_mach_number=0.0,
        axis_temperature=2.0e3 * 1.602176634e-19,
        boundary_temperature=5.0e1 * 1.602176634e-19,
    )
    return base.with_rotation_parameter(mach**2 / base.major_radius**2)


def _green_block(target, source, section):
    """Return the total-flux coupling [Wb/A] of one source set on one target."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


def _limiter_loop(case: RotatingEquilibrium, points=61, clearance=0.06):
    """Return a material boundary that touches the analytic plasma at one point.

    The half-height of the analytic boundary at one radius follows from the
    midplane flux alone, because the vertical dependence is a single
    quadratic: ``k_z Z_b^2 = psi(R, 0)``. Laying the wall ON that contour
    would make it useless as a limiter, though — the wall-point read fits a
    quadratic for the flux extremum ALONG the wall, and a wall of constant
    flux has no extremum to find. The contour is therefore held off the
    plasma by a clearance that closes smoothly to zero at the outboard
    midplane, so the flux along the wall has one clean maximum, the limiter
    contact sits there, and the last closed surface it selects is the
    analytic boundary itself.
    """
    inboard, outboard = case.boundary_midplane_radii()
    centre, half = 0.5 * (inboard + outboard), 0.5 * (outboard - inboard)
    angle = 2 * np.pi * (np.arange(points) + 0.5) / points
    radius = centre - half * np.cos(angle)
    height = np.sign(np.sin(angle)) * np.sqrt(
        np.clip(case.flux(radius, 0.0) / case.field_coefficient, 0.0, None)
    )
    offset = 1.0 + clearance * 0.5 * (1.0 + np.cos(angle))
    return np.c_[
        case.major_radius + offset * (radius - case.major_radius),
        offset * height,
    ]


@pytest.fixture(scope="module")
def ladder():
    """Return one bootstrapped free-boundary solve per rung of the ladder.

    Each rung fits a ring of external conductors to hold its own analytic
    member: the wall limits the plasma on that member's boundary contour, the
    source is the member's own rotating closure, and the conductor currents
    are the least-squares state that reproduces the analytic flux on the
    plasma and on the wall once the plasma's own current image is subtracted.
    The solve is then a genuine free-boundary fixed point whose answer is
    known.
    """
    configure_dtypes()
    lattice = FluxLattice(
        np.linspace(0.58, 1.34, NODES), np.linspace(-0.38, 0.38, NODES)
    )
    coordinate = lattice.coordinate
    section = lattice.radial_step
    angle = 2 * np.pi * np.arange(CONDUCTORS) / CONDUCTORS
    conductor = np.c_[0.96 + 0.60 * np.cos(angle), 0.60 * np.sin(angle)]
    plasma_to_grid = _green_block(coordinate, coordinate, section)
    source_to_grid = _green_block(coordinate, conductor, 0.05)

    rungs = {}
    for mach in MACH_LADDER:
        case = solve_member(mach)
        wall = _limiter_loop(case)
        seed_flux = TOTAL_FLUX * case.flux(coordinate[:, 0], coordinate[:, 1])
        wall_flux = TOTAL_FLUX * case.flux(wall[:, 0], wall[:, 1])
        inside = matplotlib.path.Path(wall).contains_points(coordinate)
        coupling = {
            "plasma_to_grid": plasma_to_grid,
            "source_to_grid": source_to_grid,
            "plasma_to_wall": _green_block(wall, coordinate, section),
            "source_to_wall": _green_block(wall, conductor, 0.05),
        }
        source = ForwardSource(
            core=rotating_profile(case),
            boundary_pressure=0.0,
            boundary_field_function=case.boundary_f,
        )

        def build(current, coupling=coupling, wall=wall, inside=inside, source=source):
            """Return the solve for one conductor state."""
            return ForwardProfile.from_lattice(
                lattice,
                source,
                external_current=current,
                wall_coordinate=wall,
                polarity=1,
                inside_material=jnp.asarray(inside),
                **coupling,
            )

        seed = jnp.asarray(np.r_[seed_flux, wall_flux])
        cell_current = np.asarray(
            build(np.zeros(CONDUCTORS)).operator.cell_current(seed)
        )
        target = np.r_[
            seed_flux - coupling["plasma_to_grid"] @ cell_current,
            wall_flux - coupling["plasma_to_wall"] @ cell_current,
        ]
        weight = np.r_[inside.astype(float), np.ones(len(wall))]
        matrix = np.r_[coupling["source_to_grid"], coupling["source_to_wall"]]
        current = np.linalg.lstsq(
            matrix * weight[:, None], target * weight, rcond=None
        )[0]
        rungs[mach] = (build(current), seed, case)
    return lattice, rungs


@pytest.fixture(scope="module")
def converged(ladder):
    """Return the converged equilibrium of every rung of the ladder."""
    _lattice, rungs = ladder
    return {
        mach: profile.solve(seed, route="anderson", evaluations=EVALUATIONS)
        for mach, (profile, seed, _case) in rungs.items()
    }


def _midplane_crossing(lattice, flux, level, axis_radius):
    """Return the inboard and outboard radii where the flux reaches a level.

    Linear interpolation between the two nodes that bracket the crossing on
    the midplane row, which resolves a boundary displacement well below the
    node spacing the domain labels are quantised on.
    """
    row = int(np.argmin(np.abs(lattice.height)))
    radius = lattice.radius
    profile = np.asarray(flux[: lattice.node_count]).reshape(lattice.shape)[:, row]
    inboard = outboard = np.nan
    for index in range(radius.size - 1):
        first, second = profile[index], profile[index + 1]
        if (first - level) * (second - level) > 0.0 or first == second:
            continue
        fraction = (level - first) / (second - first)
        crossing = radius[index] + fraction * (radius[index + 1] - radius[index])
        if crossing < axis_radius:
            inboard = crossing
        else:
            outboard = crossing
    assert np.isfinite(inboard) and np.isfinite(outboard)
    return inboard, outboard


@pytest.mark.parametrize("mach", MACH_LADDER)
def test_the_rotating_solve_reaches_its_fixed_point(converged, mach):
    """Every rung of the ladder converges under the shared accelerated ladder."""
    result = converged[mach]
    assert float(result.fixed_point.residual) < RESIDUAL_TOLERANCE
    assert bool(result.finite.passed)
    assert abs(float(result.moments.plasma_current)) > 1.0e5
    assert float(result.ledger.common_sol) == 0.0
    assert float(result.ledger.private_flux) == 0.0


@pytest.mark.parametrize("mach", MACH_LADDER)
def test_the_rotating_solve_publishes_its_closure_and_conventions(
    ladder, converged, mach
):
    """The receipt states the closure it ran under and what it produced.

    A rung at zero Mach number still ran UNDER the isothermal closure — the
    receipt names the closure the source declared, not whether the numbers it
    produced happened to be trivial — and its centrifugal factors are then
    exactly one.
    """
    lattice, _rungs = ladder
    record = converged[mach].rotation
    case = solve_member(mach)
    assert record.closure_name == "isothermal_surface"
    assert bool(record.active)
    assert float(record.reference_radius) == case.major_radius
    assert float(record.mean_particle_mass) == case.mean_particle_mass
    np.testing.assert_allclose(
        float(record.axis_mach_number), case.thermal_mach_number, rtol=1e-12
    )
    minimum = float(record.minimum_centrifugal_factor)
    maximum = float(record.maximum_centrifugal_factor)
    if mach == 0.0:
        assert (minimum, maximum) == (1.0, 1.0)
        return
    # the labelled core reaches at most one node beyond the analytic boundary
    inboard, outboard = case.boundary_midplane_radii()
    spacing = lattice.radial_step
    assert minimum < 1.0 < maximum
    assert maximum <= math.exp(
        case.rotation_parameter * ((outboard + spacing) ** 2 - case.major_radius**2)
    )
    assert minimum >= math.exp(
        case.rotation_parameter
        * (max(inboard - spacing, 0.0) ** 2 - case.major_radius**2)
    )


@pytest.mark.parametrize("mach", MACH_LADDER)
def test_the_rotating_solve_meets_its_conservation_tolerances(converged, mach):
    """Force balance closes with the centrifugal term at every Mach number."""
    ledger = converged[mach].conservation
    assert int(ledger.checked_cells) > 20
    assert float(ledger.relative_divergence_b) < DIVERGENCE_TOLERANCE
    assert float(ledger.relative_divergence_j) < DIVERGENCE_TOLERANCE
    assert float(ledger.relative_grad_shafranov) < GRAD_SHAFRANOV_TOLERANCE
    assert float(ledger.relative_force) < FORCE_TOLERANCE


def test_the_solved_axis_is_the_analytic_one_at_every_mach_number(ladder, converged):
    """The axis sits at the reference radius and does not move with rotation.

    The analytic family puts its magnetic axis at ``R_0`` for every Mach
    number, so a solve that reproduces the family reproduces that invariance
    — and any drift is a property of the lattice, not of the rotation.
    """
    lattice, _rungs = ladder
    spacing = lattice.radial_step
    radii = []
    for mach in MACH_LADDER:
        axis = np.asarray(converged[mach].topology.axis)
        radii.append(float(axis[0]))
        assert abs(axis[1]) < AXIS_TOLERANCE * lattice.vertical_step
        assert abs(axis[0] - solve_member(mach).major_radius) < AXIS_TOLERANCE * spacing
    assert max(radii) - min(radii) < AXIS_TOLERANCE * spacing


def test_rotation_pulls_the_boundary_in_by_the_analytic_shift(ladder, converged):
    """The solved boundary moves the way and the distance the family pins.

    Rotation pulls the outboard midplane boundary in harder than the inboard
    one, so the geometric centre moves inward while the axis stays put and the
    outward Shafranov shift grows. Both crossings are read off the solved flux
    by interpolation and compared against the closed-form midplane radii of the
    member the rung was built on. The reference radius stands in for the axis
    because its Mach-independence is pinned separately, which leaves this a
    statement about the boundary alone.

    Displacements are read against the static rung so the systematic offset
    between an interpolated crossing and a true contour crossing cancels. What
    is left is a resolution floor, and the weakest rung sits under it: its
    analytic shift growth is a hundredth of a node spacing. The ladder is
    therefore read for agreement with the exact analytic growth at every rung,
    and for resolved monotone growth only where the effect clears the floor.
    """
    lattice, _rungs = ladder
    spacing = lattice.radial_step
    static = solve_member(0.0)
    reference = static.boundary_midplane_radii()
    solved = _midplane_crossing(
        lattice,
        converged[0.0].flux,
        float(converged[0.0].topology.boundary_flux),
        static.major_radius,
    )
    static_shift = static.major_radius - 0.5 * sum(solved)
    resolved = []
    for mach in MACH_LADDER[1:]:
        case = solve_member(mach)
        analytic = case.boundary_midplane_radii()
        crossing = _midplane_crossing(
            lattice,
            converged[mach].flux,
            float(converged[mach].topology.boundary_flux),
            case.major_radius,
        )
        for side in range(2):
            assert (
                abs(
                    (crossing[side] - solved[side]) - (analytic[side] - reference[side])
                )
                < BOUNDARY_TOLERANCE * spacing
            ), (mach, side)
        growth = case.major_radius - 0.5 * sum(crossing) - static_shift
        analytic_growth = case.shafranov_shift - static.shafranov_shift
        assert analytic_growth > 0.0
        assert abs(growth - analytic_growth) < (
            RESOLUTION_FLOOR * spacing + 0.25 * analytic_growth
        ), mach
        if analytic_growth > RESOLUTION_FLOOR * spacing:
            resolved.append(growth)
    assert len(resolved) >= 2
    assert resolved == sorted(resolved)
    assert resolved[0] > 0.0


def test_the_rotating_solve_is_differentiable_in_the_conductor_current(ladder):
    """A functional of the converged rotating flux carries a gradient."""
    _lattice, rungs = ladder
    profile, seed, _case = rungs[0.35]
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
    delta = 1.0e-3 * float(jnp.abs(conductor[probe]))
    numeric = float(
        (
            linked_flux(conductor.at[probe].add(delta))
            - linked_flux(conductor.at[probe].add(-delta))
        )
        / (2.0 * delta)
    )
    error = abs(float(gradient[probe]) - numeric) / max(abs(numeric), 1.0e-30)
    assert error < 1.0e-3


def test_the_host_and_traced_routes_agree_on_the_rotating_map(ladder):
    """The eager and accelerated routes drive the rotating source identically."""
    _lattice, rungs = ladder
    profile, seed, _case = rungs[0.90]
    host = profile.solve(seed, route="host", evaluations=30, tolerance=0.0)
    traced = profile.solve(seed, route="picard", evaluations=30)
    scale = float(jnp.max(jnp.abs(traced.flux)))
    assert float(jnp.max(jnp.abs(host.flux - traced.flux))) / scale < 1.0e-9
    np.testing.assert_allclose(
        np.asarray(host.fixed_point.trace),
        np.asarray(traced.fixed_point.trace),
        rtol=1e-6,
    )


if __name__ == "__main__":
    pytest.main([__file__])
