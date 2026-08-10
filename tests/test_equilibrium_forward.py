r"""Contract of the prescribed-source forward equilibrium solve.

Pinned here without solving anything: the authority boundary (no measurement
or least-squares type reaches the forward modules), the immutable absolute
source state and its refusal of a sampled image, the absence of any place to
put a net-current rescale, the topology-qualified domain partition, and the
convention chain — sign, total-flux factor, pressure drive and force
residual — read off an analytic Solov'ev equilibrium.

The analytic equilibrium is the constant-gradient member of the Solov'ev
family written in Nova's total flux,

.. math::
    \Phi = \alpha R^4 + \sigma R^2 + \beta Z^2, \qquad
    \Delta^\star \Phi = 8 \alpha R^2 + 2 \beta,

so matching it to :math:`\Delta^\star \Phi = 4 \pi^2 (\mu_0 R^2 p' + FF')`
fixes :math:`\alpha = \pi^2 \mu_0 p' / 2` and :math:`\beta = 2 \pi^2 FF'`
with :math:`\sigma` free. Choosing :math:`\sigma = -2 \alpha R_{ax}^2` puts
the magnetic axis at :math:`R_{ax}`. Every factor in that chain is therefore
under test: a missing :math:`2\pi`, a flipped COCOS sign or a flux-per-radian
gradient shows up as a residual four decades above the differencing floor.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import mu_0

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium.conservation import (
        FluxLattice,
        conservation_ledger,
        delta_star,
        poloidal_field,
    )
    from nova.equilibrium.convention import (
        TOTAL_FLUX_FACTOR,
        toroidal_current_density,
    )
    from nova.equilibrium.domain import DomainMasks, PlasmaDomain
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.observation import (
        MomentEnforcementError,
        core_field_function_squared,
        core_pressure,
        current_ledger,
        observe_moments,
        reject_unsupported_enforcement,
    )
    from nova.equilibrium.source import (
        DomainProfile,
        ForwardSource,
        NormalisationPolicy,
    )
    from nova.equilibrium.topology import Topology
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes
    from nova.linalg.interpolant import Linear

ROOT = Path(__file__).parents[1]

#: Modules a caller reaches when solving a prescribed-source equilibrium.
FORWARD_MODULES = (
    "nova/equilibrium/forward.py",
    "nova/equilibrium/forward_operator.py",
    "nova/equilibrium/source.py",
    "nova/equilibrium/domain.py",
    "nova/equilibrium/observation.py",
    "nova/equilibrium/conservation.py",
    "nova/equilibrium/convention.py",
)

#: Modules and symbols that carry magnetic measurements or fit coefficients.
FITTING_MODULES = frozenset(
    {"nova.equilibrium.measurement", "nova.equilibrium.profile"}
)
FITTING_SYMBOLS = frozenset(
    {"Magnetics", "ProfilePrior", "ReconstructProfile", "lstsq", "least_squares"}
)
MEASUREMENT_TOKENS = ("measure", "sensor", "whiten", "magnetics", "prior")

P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
FLUX_SPAN = -0.5
BOUNDARY_PRESSURE = 1.2e3
BOUNDARY_FIELD_FUNCTION = 5.0

#: Pre-registered conservation tolerances of the analytic equilibrium, read
#: against the drive each residual is normalised by. The two flux-function
#: identities sit at the fp64 differencing floor; the two physical residuals
#: are second-order in the node spacing and are registered per lattice.
DIVERGENCE_TOLERANCE = 1.0e-12
GRAD_SHAFRANOV_TOLERANCE = {41: 1.0e-4, 81: 3.0e-5}
FORCE_TOLERANCE = {41: 3.0e-4, 81: 1.0e-4}
CONVERGENCE_ORDER_FLOOR = 3.4


@pytest.fixture(scope="module", autouse=True)
def device_precision():
    """Publish the fp64 device policy the equilibrium contract is read in."""
    configure_dtypes()


def _terms():
    """Return the Solov'ev quartic, offset and vertical coefficients."""
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def solovev_flux(radius, height):
    """Return the analytic total poloidal flux [Wb] of the test equilibrium."""
    alpha, offset, beta = _terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def constant_source():
    """Return the absolute source the analytic equilibrium is built from."""
    nodes = jnp.linspace(0.0, 1.0, 9)
    return ForwardSource(
        core=DomainProfile(
            p_prime=Linear(nodes, jnp.full(9, P_PRIME)),
            ff_prime=Linear(nodes, jnp.full(9, FF_PRIME)),
        ),
        boundary_pressure=BOUNDARY_PRESSURE,
        boundary_field_function=BOUNDARY_FIELD_FUNCTION,
    )


def analytic_case(nodes: int):
    """Return the lattice, flux map, domain labels and span of the equilibrium."""
    lattice = FluxLattice(np.linspace(0.6, 1.4, nodes), np.linspace(-0.45, 0.45, nodes))
    coordinate = lattice.coordinate
    flux = solovev_flux(coordinate[:, 0], coordinate[:, 1])
    axis_flux = solovev_flux(AXIS_RADIUS, 0.0)
    psi_norm = (flux - axis_flux) / FLUX_SPAN
    masks = DomainMasks(
        label=jnp.asarray(
            np.where(psi_norm <= 1.0, PlasmaDomain.CORE, PlasmaDomain.COMMON_SOL),
            dtype=jnp.int8,
        ),
        psi_norm=jnp.asarray(psi_norm),
    )
    return lattice, jnp.asarray(flux), masks, jnp.asarray(FLUX_SPAN)


@pytest.fixture(scope="module")
def analytic():
    """Return the analytic equilibrium at the registered coarse lattice."""
    return analytic_case(41)


@pytest.fixture(scope="module")
def diverted():
    """Return a two-ring diverted flux map, its topology read and its labels.

    A plasma ring and a weaker divertor ring below it produce a saddle
    between them, so the flux map carries a genuine private-flux pocket:
    cells around the divertor ring sit at normalised flux below one exactly
    like the core, yet the X-point cut separates them from the axis.
    """
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.55, 1.45, 45), np.linspace(-0.9, 0.5, 71))
    coordinate = lattice.coordinate
    rings = np.array([[1.0, 0.05], [1.0, -0.62]])
    current = np.array([1.0e6, 5.0e5])
    columns = np.stack(
        [
            hybrid_greens(coordinate[:, 0], coordinate[:, 1], a, z, 0.06, 0.06)[0]
            for a, z in rings
        ],
        axis=1,
    )
    flux = columns @ current
    angle = 2 * np.pi * np.arange(48) / 48
    wall = np.c_[1.0 + 0.42 * np.cos(angle), -0.2 + 0.66 * np.sin(angle)]
    wall_flux = (
        np.stack(
            [
                hybrid_greens(wall[:, 0], wall[:, 1], a, z, 0.06, 0.06)[0]
                for a, z in rings
            ],
            axis=1,
        )
        @ current
    )
    topology = Topology(
        Null2D.from_coordinates(coordinate, hex_stencil(lattice.shape), maxsize=5),
        Null1D(jnp.asarray(wall)),
    )
    inside = jnp.asarray(
        ((coordinate[:, 0] - 1.0) / 0.42) ** 2 + ((coordinate[:, 1] + 0.2) / 0.66) ** 2
        <= 1.0
    )
    masks, state = topology.read(jnp.asarray(np.r_[flux, wall_flux]), 1, inside)
    return masks, state, inside


def _parse(relative: str) -> ast.Module:
    """Return the parsed source of one public forward module."""
    return ast.parse((ROOT / relative).read_text())


@pytest.mark.parametrize("relative", FORWARD_MODULES)
def test_forward_modules_import_no_measurement_or_fitting_type(relative):
    """The forward solve cannot reach a measurement or a coefficient fit."""
    for node in ast.walk(_parse(relative)):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in FITTING_MODULES, relative
            assert not FITTING_SYMBOLS.intersection(
                alias.name for alias in node.names
            ), relative
        if isinstance(node, ast.Import):
            assert not FITTING_MODULES.intersection(
                alias.name for alias in node.names
            ), relative


@pytest.mark.parametrize("relative", FORWARD_MODULES)
def test_forward_signatures_accept_no_measurement_argument(relative):
    """No public forward entry point names an argument a measurement fits."""
    for node in ast.walk(_parse(relative)):
        if not isinstance(node, ast.FunctionDef):
            continue
        names = [argument.arg for argument in (*node.args.args, *node.args.kwonlyargs)]
        for name in names:
            assert not any(token in name.lower() for token in MEASUREMENT_TOKENS), (
                f"{relative}:{node.name} accepts {name}"
            )


def test_source_refuses_a_sampled_image_in_place_of_a_flux_function():
    """A measurement or cell-current image is not a force-balanced source."""
    with pytest.raises(TypeError, match="callable flux function"):
        DomainProfile(p_prime=np.linspace(0.0, 1.0, 8), ff_prime=lambda u: u)
    with pytest.raises(TypeError, match="callable flux function"):
        DomainProfile(p_prime=lambda u: u, ff_prime=[1.0, 2.0, 3.0])


def test_source_refuses_an_undeclared_normalisation_policy():
    """The shipped closure preserves the supplied amplitude exactly."""
    with pytest.raises(NotImplementedError, match="preserves the supplied source"):
        ForwardSource(
            core=constant_source().core,
            normalisation=NormalisationPolicy.DECLARED_SCALAR_CURRENT,
        )


@pytest.mark.parametrize("domain", ["common_sol", "private_flux"])
def test_source_refuses_an_open_region_closure(domain):
    """An open-region continuation needs its own declared physical contract."""
    core = constant_source().core
    with pytest.raises(NotImplementedError, match="continuity class"):
        ForwardSource(core=core, **{domain: core})


def test_the_operator_carries_no_net_plasma_current():
    """There is no field a net-current rescale could be applied through."""
    names = {field.name for field in dataclasses.fields(ForwardFluxOperator)}
    assert not [name for name in names if "current" in name and "external" not in name]
    assert "external_current" in names


def test_the_cell_current_image_is_the_supplied_source_unscaled(analytic):
    """The evaluated source reaches the current image with no amplitude change."""
    lattice, flux, masks, _ = analytic
    source = constant_source()
    radius = lattice.node_radius
    area = lattice.cell_area
    expected = np.where(
        np.asarray(masks.core),
        toroidal_current_density(radius, P_PRIME, FF_PRIME) * area,
        0.0,
    )
    current = np.asarray(
        source.cell_current(jnp.asarray(radius), jnp.asarray(area), masks)
    )
    np.testing.assert_allclose(current, expected, rtol=1e-12, atol=0.0)
    assert abs(current.sum()) > 1.0e5


def test_absolute_profiles_are_untouched_by_a_source_evaluation(analytic):
    """The supplied gradients are byte-identical before and after evaluation."""
    lattice, flux, masks, span = analytic
    source = constant_source()
    before = (
        np.asarray(source.core.p_prime.data).tobytes(),
        np.asarray(source.core.ff_prime.data).tobytes(),
    )
    source.cell_current(
        jnp.asarray(lattice.node_radius), jnp.asarray(lattice.cell_area), masks
    )
    core_pressure(source, masks, jnp.asarray(lattice.node_radius), span)
    after = (
        np.asarray(source.core.p_prime.data).tobytes(),
        np.asarray(source.core.ff_prime.data).tobytes(),
    )
    assert before == after


def test_manufactured_source_matches_the_current_density_read_from_the_field(analytic):
    """The pinned current density reproduces the field's own toroidal current.

    Ampere's law fixes the toroidal current density of a flux map with no
    freedom left, so agreement with the flux functions pins the total-flux
    factor and both COCOS signs at once.
    """
    lattice, flux, _masks, _span = analytic
    radius = lattice.node_radius
    from_field = -np.asarray(delta_star(lattice, flux)) / (
        TOTAL_FLUX_FACTOR * mu_0 * radius
    )
    from_source = toroidal_current_density(radius, P_PRIME, FF_PRIME)
    interior = np.asarray(lattice.interior())
    error = np.max(np.abs(from_field - from_source)[interior])
    assert error / np.max(np.abs(from_source)[interior]) < 1.0e-4
    assert np.all(from_source > 0.0)


def test_poloidal_field_follows_the_pinned_flux_derivatives(analytic):
    """The field reads the flux map with the declared sign and total-flux factor."""
    lattice, flux, _masks, _span = analytic
    alpha, offset, beta = _terms()
    radius, height = lattice.node_radius, lattice.coordinate[:, 1]
    radial, vertical = poloidal_field(lattice, flux)
    interior = np.asarray(lattice.interior())
    expected_radial = -2.0 * beta * height / (TOTAL_FLUX_FACTOR * radius)
    expected_vertical = (4.0 * alpha * radius**3 + 2.0 * offset * radius) / (
        TOTAL_FLUX_FACTOR * radius
    )
    field_scale = np.max(np.abs(expected_vertical[interior]))
    np.testing.assert_allclose(
        np.asarray(radial)[interior],
        expected_radial[interior],
        rtol=1e-10,
        atol=1e-12 * field_scale,
    )
    # the vertical field passes through zero at the magnetic axis, so its
    # second-order truncation is read against the field scale rather than
    # against a value that vanishes there
    np.testing.assert_allclose(
        np.asarray(vertical)[interior],
        expected_vertical[interior],
        rtol=0.0,
        atol=1e-3 * field_scale,
    )


def test_profile_primitives_integrate_inward_from_the_boundary(analytic):
    """Pressure and the toroidal-field function recover their boundary values."""
    _lattice, _flux, _masks, span = analytic
    source = constant_source()
    probe = DomainMasks(
        label=jnp.full(3, int(PlasmaDomain.CORE), dtype=jnp.int8),
        psi_norm=jnp.asarray([0.0, 0.5, 1.0]),
    )
    radius = jnp.full(3, AXIS_RADIUS)
    pressure = np.asarray(core_pressure(source, probe, radius, span))
    squared = np.asarray(core_field_function_squared(source, probe, span))
    np.testing.assert_allclose(pressure[2], BOUNDARY_PRESSURE, rtol=1e-12)
    np.testing.assert_allclose(
        pressure[0], BOUNDARY_PRESSURE + float(span) * P_PRIME, rtol=1e-10
    )
    np.testing.assert_allclose(
        pressure[1], 0.5 * (pressure[0] + pressure[2]), rtol=1e-10
    )
    np.testing.assert_allclose(squared[2], BOUNDARY_FIELD_FUNCTION**2, rtol=1e-12)
    np.testing.assert_allclose(
        squared[0],
        BOUNDARY_FIELD_FUNCTION**2 + 2.0 * float(span) * FF_PRIME,
        rtol=1e-10,
    )
    assert pressure[0] > pressure[2]


@pytest.mark.parametrize("nodes", [41, 81])
def test_analytic_equilibrium_meets_the_registered_conservation_tolerances(nodes):
    """Every conservation residual of the analytic solution is inside budget."""
    lattice, flux, masks, span = analytic_case(nodes)
    ledger = conservation_ledger(lattice, flux, constant_source(), masks, span)
    assert int(ledger.checked_cells) > 100
    assert float(ledger.relative_divergence_b) < DIVERGENCE_TOLERANCE
    assert float(ledger.relative_divergence_j) < DIVERGENCE_TOLERANCE
    assert float(ledger.relative_grad_shafranov) < GRAD_SHAFRANOV_TOLERANCE[nodes]
    assert float(ledger.relative_force) < FORCE_TOLERANCE[nodes]


def test_conservation_residuals_fall_at_second_order():
    """Halving the node spacing quarters the two physical residuals."""
    ledgers = {}
    for nodes in (41, 81):
        lattice, flux, masks, span = analytic_case(nodes)
        ledgers[nodes] = conservation_ledger(
            lattice, flux, constant_source(), masks, span
        )
    for name in ("relative_grad_shafranov", "relative_force"):
        coarse = float(getattr(ledgers[41], name))
        fine = float(getattr(ledgers[81], name))
        assert coarse / fine > CONVERGENCE_ORDER_FLOOR, name


def test_integral_observations_reproduce_their_pinned_definitions(analytic):
    """The moments match volume integrals assembled independently in the test."""
    lattice, flux, masks, span = analytic
    source = constant_source()
    radius = lattice.node_radius
    area = lattice.cell_area
    core = np.asarray(masks.core)
    cell_current = np.asarray(
        source.cell_current(jnp.asarray(radius), jnp.asarray(area), masks)
    )
    radial, vertical = poloidal_field(lattice, flux)
    field_squared = np.asarray(radial) ** 2 + np.asarray(vertical) ** 2
    observation = observe_moments(
        source,
        masks,
        jnp.asarray(radius),
        jnp.asarray(area),
        jnp.asarray(cell_current),
        jnp.asarray(field_squared),
        span,
    )

    volume_element = np.where(core, 2.0 * np.pi * radius * area, 0.0)
    volume = volume_element.sum()
    plasma_current = cell_current[core].sum()
    major_radius = (radius * volume_element).sum() / volume
    pressure = BOUNDARY_PRESSURE + float(span) * P_PRIME * (
        1.0 - np.asarray(masks.psi_norm)
    )
    reference = mu_0 * major_radius * plasma_current**2
    np.testing.assert_allclose(observation.volume, volume, rtol=1e-12)
    np.testing.assert_allclose(observation.plasma_current, plasma_current, rtol=1e-12)
    np.testing.assert_allclose(observation.major_radius, major_radius, rtol=1e-12)
    np.testing.assert_allclose(
        observation.poloidal_beta,
        4.0 * (pressure * volume_element).sum() / reference,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        observation.internal_inductance,
        2.0 * (field_squared * volume_element).sum() / (mu_0 * reference),
        rtol=1e-12,
    )
    assert 0.0 < float(observation.poloidal_beta) < 10.0
    assert 0.0 < float(observation.internal_inductance) < 10.0


def test_private_flux_is_labelled_apart_from_the_core(diverted):
    """A closed-surface pocket the X-point cut isolates is not core."""
    masks, state, _inside = diverted
    counts = np.asarray(masks.cell_count())
    assert bool(state.diverted)
    assert counts[PlasmaDomain.CORE] > 50
    assert counts[PlasmaDomain.PRIVATE_FLUX] > 5
    private = np.asarray(masks.private_flux)
    psi_norm = np.asarray(masks.psi_norm)
    assert np.all(psi_norm[private] <= 1.0)
    assert not np.any(private & np.asarray(masks.core))


def test_open_flux_alone_does_not_select_the_open_field_cells(diverted):
    """Normalised flux above one misses the private branch and leaks material."""
    masks, _state, inside = diverted
    psi_norm = np.asarray(masks.psi_norm)
    material = np.asarray(inside)
    naive = psi_norm > 1.0
    private = np.asarray(masks.private_flux)
    common = np.asarray(masks.common_sol)

    assert np.any(private & ~naive), "the private branch sits below one"
    assert np.any(naive & ~material), "the range label leaks outside the material"
    # inside the material the open label and the flux range agree exactly: the
    # closed test cuts at the boundary itself, so no within-boundary cell can
    # carry an open label and no open cell can sit below one
    np.testing.assert_array_equal(common, naive & material)
    assert np.all(psi_norm[common] > 1.0)


def test_no_current_appears_outside_the_declared_support(diverted):
    """Only the core carries source current; every other domain integrates zero."""
    masks, _state, _inside = diverted
    source = constant_source()
    radius = jnp.asarray(np.full(masks.psi_norm.shape, 1.0))
    area = jnp.asarray(np.full(masks.psi_norm.shape, 1.0e-4))
    ledger = current_ledger(source.cell_current(radius, area, masks), masks)
    assert abs(float(ledger.core)) > 0.0
    assert float(ledger.common_sol) == 0.0
    assert float(ledger.private_flux) == 0.0
    assert float(ledger.excluded_material) == 0.0
    np.testing.assert_allclose(ledger.total, ledger.core, rtol=1e-12)


def test_enforcing_a_moment_fails_against_an_absolute_closure():
    """An absolute source carries no scalar freedom to close a moment with."""
    source = constant_source()
    assert source.closure_degrees == 0
    with pytest.raises(MomentEnforcementError, match="scalar closure"):
        reject_unsupported_enforcement(("plasma_current",), source.closure_degrees)
    with pytest.raises(MomentEnforcementError, match="unknown integral"):
        reject_unsupported_enforcement(("triangularity",), 3)
    with pytest.raises(MomentEnforcementError, match="enforced twice"):
        reject_unsupported_enforcement(("plasma_current", "plasma_current"), 3)
    assert reject_unsupported_enforcement((), 0) == ()


def test_the_solve_refuses_enforcement_before_reading_the_profiles():
    """The enforcement guard fires before a single profile value is read."""
    lattice = FluxLattice(np.linspace(0.8, 1.2, 9), np.linspace(-0.2, 0.2, 9))
    nodes = lattice.node_count
    angle = 2 * np.pi * np.arange(12) / 12
    source = constant_source()
    profile = ForwardProfile.from_lattice(
        lattice,
        source,
        external_current=np.zeros(1),
        source_to_grid=np.zeros((nodes, 1)),
        plasma_to_grid=np.zeros((nodes, nodes)),
        source_to_wall=np.zeros((len(angle), 1)),
        plasma_to_wall=np.zeros((len(angle), nodes)),
        wall_coordinate=np.c_[1.0 + 0.15 * np.cos(angle), 0.15 * np.sin(angle)],
    )
    before = np.asarray(source.core.p_prime.data).tobytes()
    with pytest.raises(MomentEnforcementError):
        profile.solve(np.zeros(nodes + len(angle)), enforce=("poloidal_beta",))
    with pytest.raises(TypeError):
        profile.solve(np.zeros(nodes + len(angle)), measured=np.zeros(4))
    assert np.asarray(source.core.p_prime.data).tobytes() == before


if __name__ == "__main__":
    pytest.main([__file__])
