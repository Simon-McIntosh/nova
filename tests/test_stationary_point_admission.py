"""Connected, contained stationary-point admission on the solve's read path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp
    from scipy.interpolate import griddata

    from nova.biot.null import Null1D, Null2D
    from nova.biot.target import FluxTarget
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.connectivity_boundary import _points_inside_polygon
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.equilibrium.topology import Topology
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes


BANK_OPERANDS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/"
    "bank-regeneration-raw-20260902/current-operands.npz"
)


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Match production topology precision."""
    configure_dtypes()


def _material_fixture():
    """Structured hex grid with a bounding-box wall, as in the qualification suite."""
    radius = np.linspace(0.0, 2.0, 7)
    height = np.linspace(-2.0, 2.0, 7)
    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    coordinate = np.c_[radial.ravel(), vertical.ravel()]
    grid = Null2D.from_coordinates(coordinate, hex_stencil((7, 7)), maxsize=5)
    wall = Null1D(
        jnp.asarray(
            [[0.0, -2.0], [2.0, -2.0], [2.0, 2.0], [0.0, 2.0]],
            dtype=jnp.float64,
        )
    )
    return Topology(grid, wall), coordinate


def test_isolated_deep_well_loses_to_broad_connected_extremum():
    """A materially disconnected but far deeper well must not outrank the axis.

    The isolated candidate's raw flux is twice as extreme as the connected
    candidate's, so a pure flux-value ranking over unfiltered candidates would
    pick it. Admission must instead select whichever candidate floods a
    material-connected component, regardless of relative flux magnitude.
    """
    topology, coordinate = _material_fixture()
    connected = np.array([1.0, 0.0, 5.0, 0.0])
    isolated = np.array([1.0, 2.0, 10.0, 0.0])
    vmap_o = jnp.asarray(
        np.vstack((connected, isolated, np.full((3, 4), np.nan))),
        dtype=jnp.float64,
    )
    vmap_x = jnp.full((5, 4), np.nan, dtype=jnp.float64)
    data_w = jnp.asarray([2.0, 0.0, -1.0, 0.0], dtype=jnp.float64)

    central = (np.abs(coordinate[:, 0] - 1.0) <= 0.7) & (
        np.abs(coordinate[:, 1]) <= 0.7
    )
    flux = np.where(central, 1.0, -1.0)
    connected_owner = int(np.argmin(np.sum((coordinate - connected[:2]) ** 2, axis=1)))
    isolated_owner = int(np.argmin(np.sum((coordinate - isolated[:2]) ** 2, axis=1)))
    flux[connected_owner] = connected[2]
    flux[isolated_owner] = isolated[2]
    material = central.copy()
    material[connected_owner] = False
    material[isolated_owner] = False

    qualified = topology.qualified_o_candidates(
        vmap_o, vmap_x, data_w, 1, jnp.asarray(flux), jnp.asarray(material)
    )
    np.testing.assert_array_equal(
        np.asarray(qualified), [True, False, False, False, False]
    )
    selected = topology.o_point_data(vmap_o, 1, qualified)
    np.testing.assert_array_equal(np.asarray(selected), connected)

    unfiltered_index = int(jnp.argmax(vmap_o[:, 2]))
    assert unfiltered_index != int(
        jnp.argmax(jnp.where(qualified, vmap_o[:, 2], -jnp.inf))
    )


def test_out_of_vessel_saddle_loses_to_in_vessel_x_point():
    """A saddle outside the wall polygon must not outrank an in-vessel one.

    The excluded candidate scores higher on raw ``polarity * (x_psi - o_psi)``
    than the true separatrix, so a ranking with no containment screen would
    select it instead.
    """
    radius = np.linspace(0.4, 1.6, 13)
    height = np.linspace(-1.0, 1.0, 11)
    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    coordinate = np.c_[radial.ravel(), vertical.ravel()]
    grid = Null2D.from_coordinates(coordinate, hex_stencil((13, 11)), maxsize=5)
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    wall_coordinate = np.c_[1.0 + 0.3 * np.cos(wall_angle), 0.45 * np.sin(wall_angle)]
    wall = Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64))
    topology = Topology(grid, wall)

    o_psi = jnp.asarray(1.0)
    in_vessel = jnp.asarray([1.0, -0.3, 0.6, 0.0])
    out_of_vessel = jnp.asarray([1.55, 0.0, 0.9, 0.0])
    vmap_x = jnp.asarray(
        np.vstack(
            (
                np.asarray(in_vessel),
                np.asarray(out_of_vessel),
                np.full((3, 4), np.nan),
            )
        ),
        dtype=jnp.float64,
    )

    index = topology.x_point_index(vmap_x, 1.0, o_psi)
    np.testing.assert_allclose(np.asarray(vmap_x[index]), np.asarray(in_vessel))

    unscreened_score = 1.0 * (np.asarray(vmap_x)[:, 2] - float(o_psi))
    assert int(np.nanargmax(unscreened_score)) == 1
    assert int(index) != 1


def _bank_arm_topology(arm: str):
    """Rebuild a Topology and inside-material mask for one banked MAST arm."""
    with np.load(BANK_OPERANDS, allow_pickle=False) as stored:
        radius = stored[f"arm_{arm}_radius"]
        height = stored[f"arm_{arm}_height"]
        flux2d = stored[f"arm_{arm}_flux"]
        wall = stored[f"arm_{arm}_wall"]

    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    coordinate = np.c_[radial.ravel(), vertical.ravel()]
    shape = (radius.size, height.size)
    grid = Null2D.from_coordinates(coordinate, hex_stencil(shape), maxsize=30)
    wall_null = Null1D(jnp.asarray(wall, dtype=jnp.float64))
    topology = Topology(grid, wall_null)

    flat = flux2d.T.reshape(-1)
    inside_material = _points_inside_polygon(
        jnp.asarray(coordinate[:, 0]),
        jnp.asarray(coordinate[:, 1]),
        jnp.asarray(wall[:, 0]),
        jnp.asarray(wall[:, 1]),
    )
    wall_flux = griddata(coordinate, flat, wall, method="linear")
    cell_pitch = float(np.max(np.diff(radius)))
    return topology, flat, wall_flux, inside_material, cell_pitch


@pytest.mark.parametrize("arm", ["04", "05"], ids=["21985-51-pure", "21985-51-mixed"])
@pytest.mark.skipif(
    not BANK_OPERANDS.exists(), reason="banked MAST regeneration operands unavailable"
)
def test_disjoint_axis_arms_admit_the_confined_cluster(arm):
    """Both known-disjoint 21985/51 arms must admit the R~0.90 m cluster.

    Prior to the connectivity fix these arms each admitted an axis in a
    private, materially disconnected well far from the confined region every
    other arm on the same machine geometry agrees on.
    """
    topology, flat, wall_flux, inside_material, cell_pitch = _bank_arm_topology(arm)
    combined = jnp.concatenate(
        (
            jnp.asarray(flat, dtype=jnp.float64),
            jnp.asarray(wall_flux, dtype=jnp.float64),
        )
    )
    qualification = topology.read_qualification(combined, 1, inside_material)

    axis = np.asarray(qualification.state.axis)
    assert bool(qualification.axis_admitted)
    assert abs(axis[0] - 0.90) <= cell_pitch


def _single_axis_operator(*, inside_material=None) -> ForwardFluxOperator:
    """A minimal forward operator whose flux map admits without any rescue."""
    radius = np.linspace(1.05, 2.35, 17)
    height = np.linspace(-0.72, 0.72, 19)
    lattice = FluxLattice(radius, height)
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    wall_coordinate = np.c_[1.7 + 0.64 * np.cos(wall_angle), 0.68 * np.sin(wall_angle)]
    node_count = lattice.node_count

    def _zero_profile(psi_norm):
        return jnp.zeros_like(psi_norm)

    return ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.zeros((node_count, 1)),
            plasma_target=jnp.zeros((node_count, 1)),
            null=Null2D.from_coordinates(
                lattice.coordinate, hex_stencil(lattice.shape), maxsize=5
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.zeros((len(wall_coordinate), 1)),
            plasma_target=jnp.zeros((len(wall_coordinate), 1)),
            null=Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64)),
        ),
        source=ForwardSource(
            core=DomainProfile(p_prime=_zero_profile, ff_prime=_zero_profile)
        ),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        inside_material=inside_material,
        use_linear_moments=False,
    )


def test_production_reader_retains_the_full_coarse_grid_candidate_census():
    """Forward reads must not inherit the five-slot exploratory locator table."""
    operator = _single_axis_operator()

    assert operator.grid.null.maxsize == 5
    assert operator._fixed_design_topology.grid.locator.maxsize == 30


def test_empty_o_qualification_has_a_sentinel_index_and_nan_data():
    """No candidate must never masquerade as whichever entry occupies row zero."""
    topology, _coordinate = _material_fixture()
    candidates = jnp.asarray(
        np.vstack((np.array([1.0, 0.0, 2.0, 0.0]), np.full((4, 4), np.nan)))
    )
    qualified = jnp.zeros(5, dtype=bool)

    assert int(topology.o_point_index(candidates, 1, qualified)) == -1
    selection = topology.o_point_qualification(candidates, 1, qualified)
    assert not bool(selection.admitted)
    assert np.all(np.isnan(np.asarray(selection.data)))


def test_candidate_table_reports_more_extrema_than_its_capacity():
    """A fixed table publishes truncation instead of silently hiding extrema."""
    topology, coordinate = _material_fixture()
    parity = np.indices((7, 7)).sum(axis=0) % 2
    flux = np.where(parity, 1.0, -1.0).reshape(coordinate.shape[0])

    status = topology.grid.candidate_table_status(jnp.asarray(flux))

    assert int(status["candidate_count"][0]) > topology.grid.maxsize
    assert bool(status["truncated"][0])
    assert int(status["capacity"][0]) == topology.grid.maxsize


def test_independent_second_pass_rescues_an_empty_first_qualification():
    """An independently seeded second pass may recover an empty first pass."""
    initial = SimpleNamespace(
        axis_admitted=jnp.asarray(False),
        state=SimpleNamespace(axis=jnp.full(2, jnp.nan)),
    )
    recovered_state = SimpleNamespace(axis=jnp.asarray([1.7, 0.0]))
    recovered = SimpleNamespace(
        axis_admitted=jnp.asarray(True),
        state=recovered_state,
        masks="masks",
        connected="connected",
    )

    class _QualificationSequence:
        def __init__(self):
            self.calls = 0

        def read_qualification(self, *_args):
            result = initial if self.calls == 0 else recovered
            self.calls += 1
            return result

        @staticmethod
        def split_flux_map(physical):
            return physical, jnp.empty(0)

        @staticmethod
        def grid(_grid_flux):
            return jnp.asarray([[1.7, 0.0, 1.0, 1.0]]), jnp.empty((0, 4))

    operator = object.__new__(ForwardFluxOperator)
    operator._fixed_design_topology = _QualificationSequence()
    operator.polarity = 1
    operator.inside_material = jnp.asarray([True])
    operator._material_centroid = jnp.asarray([1.7, 0.0])
    operator.grid = SimpleNamespace(coordinate=jnp.asarray([[1.7, 0.0]]))

    masks, state, connected, admitted = operator._fixed_design_read(jnp.asarray([0.0]))

    assert masks == "masks"
    assert connected == "connected"
    assert bool(admitted)
    np.testing.assert_array_equal(np.asarray(state.axis), [1.7, 0.0])


def test_second_pass_never_changes_an_already_qualified_axis():
    """Changing the independent rescue seed cannot admit a wrong-polarity null."""
    operator = _single_axis_operator()
    grid_coordinate = np.asarray(operator.grid.coordinate)
    wall_coordinate = np.asarray(operator.wall.coordinate)

    def _limited_flux(radius, height):
        return (radius - 1.7) ** 2 + height**2

    physical = jnp.asarray(
        np.r_[
            _limited_flux(grid_coordinate[:, 0], grid_coordinate[:, 1]),
            _limited_flux(wall_coordinate[:, 0], wall_coordinate[:, 1]),
        ]
    )

    for rescue_seed in ([1.05, -0.72], [2.35, 0.72]):
        operator._material_centroid = jnp.asarray(rescue_seed, dtype=jnp.float64)
        _masks, state, _connected, admitted = operator._fixed_design_read(physical)
        assert not bool(admitted)
        assert np.all(np.isnan(np.asarray(state.axis)))
