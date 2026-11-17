"""The cold seed is one traced array construction, not a host island.

The oracle is the host construction the portfolio carried before the seed
became an array kernel: the limiter extent, the uniform disc and the saddle
normalisation evaluated with the host construction primitives.  The kernels are
compared against it slice by slice, and the comparison is tight enough that a
relative perturbation of the oracle at 1e-9 is refused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import cold_seed
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward import (
    ColdSeedConstruction,
    ForwardProfile,
    SaddleSeedGeometry,
)
from nova.equilibrium.moment import (
    CurrentCells,
    MomentConfig,
    MomentOrder,
    ReconstructMoment,
    limiter_radial_extent,
)
from nova.equilibrium.solve_request import ColdSeedPortfolio, ForwardSolveRequest
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes

RELATIVE_TOLERANCE = 1.0e-12
CENTROIDS = ((1.0, 0.0), (0.97, 0.06), (1.04, -0.05))
CURRENTS = (12_000.0, 9_500.0, 15_250.0)
AXIS = (1.0, 0.0)
SADDLE = (1.18, 0.24)


def _zero_profile(value):
    return jnp.zeros_like(value)


def _lattice_profile() -> ForwardProfile:
    """Build a polygon-backed operator whose moments carry their geometry."""

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    lattice = FluxLattice(np.linspace(0.6, 1.4, 9), np.linspace(-0.4, 0.4, 9))
    node_count = lattice.node_count
    angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    wall_count = angle.size
    return ForwardProfile.from_lattice(
        lattice,
        ForwardSource(
            core=DomainProfile(p_prime=_zero_profile, ff_prime=_zero_profile)
        ),
        external_current=np.zeros(1),
        source_to_grid=np.zeros((node_count, 1)),
        plasma_to_grid=np.eye(node_count),
        source_to_wall=np.zeros((wall_count, 1)),
        plasma_to_wall=np.zeros((wall_count, node_count)),
        wall_coordinate=np.c_[1.0 + 0.5 * np.cos(angle), 0.5 * np.sin(angle)],
        cubic_cell_average=False,
    )


def _host_disc_current(profile: ForwardProfile, current: float, centroid) -> tuple:
    """Return the pre-kernel host disc current, its radius and support count."""

    operator = profile.operator
    centre = np.asarray(centroid, dtype=np.float64)
    coordinate = np.asarray(operator.grid.coordinate, dtype=np.float64)
    wall = np.asarray(operator.wall.coordinate, dtype=np.float64)
    config = MomentConfig(order=MomentOrder.CENTROID)
    inboard, outboard = limiter_radial_extent(wall[:, 0], wall[:, 1], centre[1])
    supported_distance = min(centre[0] - inboard, outboard - centre[0])
    radius = config.seed_radius_fraction * supported_distance
    reconstruction = ReconstructMoment(
        CurrentCells(
            coordinate[:, 0],
            coordinate[:, 1],
            candidate=np.asarray(operator.inside_material, dtype=np.float64),
        ),
        major_radius=float(centre[0]),
        config=config,
    )
    cell_current = reconstruction.uniform_disc(
        float(centre[0]), float(centre[1]), radius, float(current)
    )
    return cell_current, radius, int(np.count_nonzero(cell_current))


def _host_neutral_flux(profile: ForwardProfile, current: float, centroid) -> np.ndarray:
    """Return the host neutral limited seed: external field plus moment image."""

    operator = profile.operator
    cell_current, _radius, _count = _host_disc_current(profile, current, centroid)
    uniform = jnp.asarray(cell_current, dtype=jnp.float64)
    zero = jnp.zeros_like(uniform)
    coefficients = operator.coupling_current_moments(
        CellCurrentMoments(uniform, zero, zero)
    )
    neutral = operator.external(None) + operator.current_moment_image(coefficients)
    return np.asarray(neutral, dtype=np.float64)


def _host_saddle_seed(
    profile: ForwardProfile,
    neutral: np.ndarray,
    geometry: SaddleSeedGeometry,
) -> np.ndarray:
    """Return the pre-kernel host saddle-geometry seed, evaluated with NumPy."""

    operator = profile.operator
    axis = np.asarray(geometry.axis, dtype=np.float64)
    saddle = np.asarray(geometry.saddle, dtype=np.float64)
    displacement = saddle - axis
    distance = float(np.linalg.norm(displacement))
    along = displacement / distance
    across = np.array((-along[1], along[0]))
    parts = [
        np.asarray(operator.grid.coordinate, dtype=np.float64),
        np.asarray(operator.wall.coordinate, dtype=np.float64),
    ]
    if operator.sample is not None:
        parts.append(np.asarray(operator.sample.coordinate, dtype=np.float64))
    coordinates = np.vstack(parts)
    grid = parts[0]
    axis_index = int(np.argmin(np.linalg.norm(grid - axis, axis=1)))
    saddle_index = np.argmin(np.linalg.norm(grid - saddle, axis=1))
    polarity = float(operator.polarity)
    flux_span = polarity * (neutral[axis_index] - neutral[saddle_index])
    resolution = np.finfo(np.float64).eps * max(np.max(np.abs(neutral)), 1.0)
    flux_span = max(float(flux_span), float(resolution))
    local = coordinates - axis
    axial = local @ along / distance
    transverse = local @ across / distance
    potential = -0.5 * axial**2 + axial**3 / 3.0 - 0.5 * transverse**2
    normalized = 6.0 * (potential + 1.0 / 6.0)
    return neutral[saddle_index] + polarity * flux_span * normalized


def _host_branch(profile: ForwardProfile, current: float, centroid, index: int):
    """Return the host seed of one selected branch, as the portfolio receipt did."""

    neutral = _host_neutral_flux(profile, current, centroid)
    if index == int(ColdSeedConstruction.CURRENT_CENTROID_DISC):
        return neutral
    geometry = SaddleSeedGeometry(AXIS, SADDLE)
    return _host_saddle_seed(profile, neutral, geometry)


def _relative_difference(observed, reference) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    return float(np.max(np.abs(observed - reference))) / scale


def _assert_seed_parity(observed, reference) -> None:
    """Require a traced seed to reproduce its host oracle to 1e-12 relative."""

    difference = _relative_difference(observed, reference)
    assert difference <= RELATIVE_TOLERANCE, (
        f"traced cold seed differs from the host oracle by {difference:.3e} relative"
    )


@pytest.mark.parametrize("current, centroid", tuple(zip(CURRENTS, CENTROIDS)))
def test_traced_seed_matches_the_host_construction(current, centroid) -> None:
    """Both branches reproduce the host seed on every cold-seed case."""

    profile = _lattice_profile()
    geometry = SaddleSeedGeometry(AXIS, SADDLE)
    portfolio = profile.cold_seed_portfolio(
        current, centroid, diverted_geometry=geometry
    )

    for index in (0, 1):
        _assert_seed_parity(
            np.asarray(portfolio.branches.flux[index]),
            _host_branch(profile, current, centroid, index),
        )


def test_seed_parity_refuses_a_perturbed_host_reference() -> None:
    """The 1e-12 comparison has teeth: a 1e-9 perturbed oracle is refused."""

    profile = _lattice_profile()
    current, centroid = CURRENTS[0], CENTROIDS[0]
    traced = np.asarray(profile.cold_seed_portfolio(current, centroid).branches.flux[0])
    reference = _host_branch(profile, current, centroid, 0)

    _assert_seed_parity(traced, reference)
    with pytest.raises(AssertionError, match="differs from the host oracle"):
        _assert_seed_parity(traced, reference * (1.0 + 1.0e-9))


def test_jit_vmap_cold_seed_kernel_runs_over_three_distinct_slices() -> None:
    """The seed kernel traces over three slices with no host array call."""

    profile = _lattice_profile()
    geometry = SaddleSeedGeometry(AXIS, SADDLE)
    centroids = jnp.asarray(CENTROIDS, dtype=jnp.float64)
    currents = jnp.asarray(CURRENTS, dtype=jnp.float64)

    def build(current, centroid):
        kernel = cold_seed.current_centroid_seed(
            profile.operator,
            centroid,
            current,
            radius_fraction=MomentConfig(
                order=MomentOrder.CENTROID
            ).seed_radius_fraction,
        )
        return kernel.flux, cold_seed.saddle_geometry_seed(
            profile.operator, kernel.flux, geometry
        )

    traced = jax.jit(jax.vmap(build))(currents, centroids)
    limited, diverted = (np.asarray(branch) for branch in traced)

    assert limited.shape == (len(CURRENTS), profile.operator.node_number)
    for slice_index, (current, centroid) in enumerate(zip(CURRENTS, CENTROIDS)):
        _assert_seed_parity(
            limited[slice_index], _host_branch(profile, current, centroid, 0)
        )
        _assert_seed_parity(
            diverted[slice_index], _host_branch(profile, current, centroid, 1)
        )

    for left in range(len(CURRENTS)):
        for right in range(left + 1, len(CURRENTS)):
            assert not np.array_equal(limited[left], limited[right])

    primitives = {
        equation.primitive.name
        for equation in jax.make_jaxpr(jax.vmap(build))(currents, centroids).jaxpr.eqns
    }
    assert not primitives & {"host_callback", "pure_callback", "io_callback"}


def test_request_cold_seed_builds_inside_jit() -> None:
    """The production request's seed construction traces and keeps its provenance."""

    profile = _lattice_profile()
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="traced-cold-seed-carrier",
        source_profile=profile.source,
        seed_policy=ColdSeedPortfolio(
            plasma_current=CURRENTS[1],
            centroid=CENTROIDS[1],
            requested_class="diverted",
        ),
    )
    policy = request.seed_policy
    assert isinstance(policy, ColdSeedPortfolio)

    seed = jax.jit(lambda _: request.seed_policy.resolve(profile))(
        jnp.zeros(())
    ).block_until_ready()

    index = 0 if policy.requested_class == "limited" else 1
    host = np.asarray(
        profile.cold_seed_portfolio(
            policy.plasma_current, policy.centroid
        ).branches.flux[index]
    )
    _assert_seed_parity(seed, host)
    assert policy.provenance().to_dict() == {
        "kind": "cold_seed_portfolio",
        "requested_class": "diverted",
        "plasma_current": CURRENTS[1],
        "centroid": CENTROIDS[1],
    }
