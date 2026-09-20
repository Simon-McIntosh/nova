"""Containment, polish, and telemetry pins for the forward null census."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.null import Null1D, Null2D
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward_operator import _FixedDesignNull2D
from nova.equilibrium.flux_surface_connectivity import fit_tensor_spline
from nova.equilibrium.topology import Topology
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Exercise the precision used by production forward maps."""
    configure_dtypes()


def _locator(
    radial: np.ndarray, vertical: np.ndarray, *, capacity: int = 30
) -> _FixedDesignNull2D:
    """Build the production census wrapper on one tensor-product lattice."""
    lattice = FluxLattice(radial, vertical)
    locator = Null2D.from_coordinates(
        lattice.coordinate,
        hex_stencil(lattice.shape),
        maxsize=capacity,
    )
    return _FixedDesignNull2D.from_locator(locator)


def _flat_field(values: np.ndarray) -> jax.Array:
    """Flatten ``values[vertical, radial]`` in the forward-grid ordering."""
    return jnp.asarray(values.T.reshape(-1), dtype=jnp.float64)


def _quadratic_saddle(
    radial: np.ndarray,
    vertical: np.ndarray,
    root: tuple[float, float],
) -> np.ndarray:
    """Return one exact saddle sampled as ``values[vertical, radial]``."""
    mesh_r, mesh_z = np.meshgrid(radial, vertical)
    return (mesh_r - root[0]) ** 2 - 0.8 * (mesh_z - root[1]) ** 2


def _bank_path() -> Path:
    """Resolve the externally banked terminal maps supplied to this gate."""
    value = os.environ.get("NOVA_FORWARD_CENSUS_BANK")
    if value is None:
        pytest.skip("NOVA_FORWARD_CENSUS_BANK does not name the banked operands")
    path = Path(value)
    if not path.is_file():
        pytest.skip("the configured forward-census bank is unavailable")
    return path


def _bank_rows(archive) -> list[tuple[int, dict]]:
    """Return the bank rows in their authored order."""
    metadata = json.loads(str(archive["metadata"]))
    return list(enumerate(metadata["rows"]))


def _bank_census(archive, index: int):
    """Rebuild the fixed census and its spline values for one bank row."""
    prefix = f"arm_{index:02d}"
    radial = np.asarray(archive[f"{prefix}_radius"], dtype=np.float64)
    vertical = np.asarray(archive[f"{prefix}_height"], dtype=np.float64)
    values = np.asarray(archive[f"{prefix}_flux"], dtype=np.float64)
    fixed = _locator(radial, vertical)
    field = _flat_field(values)
    return fixed, values, fixed.candidate_table_status(field)


def bank_census_receipt(path: Path, revision: str) -> dict:
    """Return the quantitative twelve-map detector comparison receipt."""
    rows = []
    with np.load(path, allow_pickle=True) as archive:
        for index, row in _bank_rows(archive):
            prefix = f"arm_{index:02d}"
            fixed, values, census = _bank_census(archive, index)
            candidates = fixed(_flat_field(values))
            wall = np.asarray(archive[f"{prefix}_wall"], dtype=np.float64)
            topology = Topology(fixed, Null1D(jnp.asarray(wall)))
            spline = fit_tensor_spline(
                fixed.spline_radial,
                fixed.spline_vertical,
                jnp.asarray(values),
            )
            current_axis = np.asarray(archive[f"{prefix}_axis"], dtype=np.float64)
            current_saddle = np.asarray(
                archive[f"{prefix}_selected_saddle"], dtype=np.float64
            )
            efit_axis = np.asarray(archive[f"{prefix}_efit_axis"], dtype=np.float64)
            efit_saddles = np.asarray(
                archive[f"{prefix}_efit_x_points"], dtype=np.float64
            )
            binding_flux = float(archive[f"{prefix}_binding_flux"])
            current_axis_flux = float(spline(current_axis[0], current_axis[1]))
            polarity = 1 if current_axis_flux >= binding_flux else -1
            selected_axis = np.asarray(
                topology.o_point_data(candidates[0], polarity), dtype=np.float64
            )
            selected_saddle = np.asarray(
                topology.x_point_data(candidates[1], polarity, selected_axis[2]),
                dtype=np.float64,
            )
            valid = np.asarray(census["retained_valid"])
            gradient = np.asarray(census["retained_spline_gradient_norm"])
            determinant = np.asarray(census["retained_spline_hessian_determinant"])
            displacement_rejected = np.asarray(census["polish_rejected"])
            requested_displacement = np.asarray(
                census["requested_displacement"], dtype=np.float64
            )
            counts = {
                key: np.asarray(census[key], dtype=int).tolist()
                for key in (
                    "raw_ring_count",
                    "polished_count",
                    "typed_count",
                    "same_root_count",
                    "retained_count",
                )
            }
            rows.append(
                {
                    "identity": row["identity"],
                    "arm": row["arm"],
                    "counts_o_x": counts,
                    "capacity_o_x": np.asarray(census["capacity"], dtype=int).tolist(),
                    "overflow_o_x": np.asarray(census["overflow"], dtype=bool).tolist(),
                    "maximum_multiplicity_o_x": [
                        int(np.max(np.asarray(census["retained_multiplicity"])[kind]))
                        for kind in range(2)
                    ],
                    "rejected_cell_displacement_o_x": np.sum(
                        displacement_rejected, axis=1, dtype=int
                    ).tolist(),
                    "maximum_requested_displacement_m": float(
                        np.nanmax(requested_displacement)
                    ),
                    "maximum_retained_gradient_wb_per_m": float(
                        np.max(gradient[valid])
                    ),
                    "minimum_absolute_retained_hessian_determinant": float(
                        np.min(np.abs(determinant[valid]))
                    ),
                    "selected_axis_rz_m": selected_axis[:2].tolist(),
                    "current_production_axis_rz_m": current_axis.tolist(),
                    "efit_axis_rz_m": efit_axis.tolist(),
                    "selected_axis_to_current_production_mm": float(
                        1.0e3 * np.linalg.norm(selected_axis[:2] - current_axis)
                    ),
                    "selected_axis_to_efit_mm": float(
                        1.0e3 * np.linalg.norm(selected_axis[:2] - efit_axis)
                    ),
                    "selected_saddle_rz_m": selected_saddle[:2].tolist(),
                    "current_production_saddle_rz_m": current_saddle.tolist(),
                    "efit_saddles_rz_m": efit_saddles.tolist(),
                    "selected_saddle_to_current_production_mm": float(
                        1.0e3 * np.linalg.norm(selected_saddle[:2] - current_saddle)
                    ),
                    "selected_saddle_to_nearest_efit_mm": float(
                        1.0e3
                        * np.min(
                            np.linalg.norm(efit_saddles - selected_saddle[:2], axis=1)
                        )
                    ),
                }
            )
    return {
        "schema": "forward-census",
        "source_revision": revision,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "count_order": ["extremum", "saddle"],
        "rows": rows,
    }


def test_forward_census_requires_spline_ring_sign_count():
    """A neighbouring quadratic fit with two ring changes is not a saddle."""
    radial = np.linspace(0.2, 2.2, 17)
    vertical = np.linspace(-1.0, 1.0, 17)
    fixed = _locator(radial, vertical)
    values = _quadratic_saddle(radial, vertical, (1.2, 0.0))
    field = _flat_field(values)

    _local_candidates, local_masks = fixed._local_fit_census(field)
    census = fixed.candidate_census(field)
    false_local_saddle = local_masks[1] & (census["ring_crossing_count"] == 2)

    assert bool(jnp.any(false_local_saddle))
    assert not bool(jnp.any(census["representative_mask"][1] & false_local_saddle))
    assert int(census["same_root_count"][1]) == 1


def test_forward_census_drops_incomplete_boundary_rings():
    """Outer nodes are absent while the first complete ring remains visible."""
    radial = np.linspace(0.2, 1.8, 9)
    vertical = np.linspace(-0.8, 0.8, 9)
    fixed = _locator(radial, vertical)
    source = np.asarray(fixed.locator.stencil[:, 0])
    source_index = np.asarray(np.unravel_index(source, (9, 9))).T

    assert np.all((source_index > 0) & (source_index < 8))
    np.testing.assert_array_equal(source_index[0], [1, 1])

    root = tuple(np.asarray(fixed.locator.physical_origin[0]))
    census = fixed.candidate_census(
        _flat_field(_quadratic_saddle(radial, vertical, root))
    )
    assert bool(census["ring_admitted_mask"][1, 0])
    assert int(census["source_origin_index"][0]) == int(source[0])


def test_forward_census_uses_one_tensor_spline_for_position_value_and_hessian():
    """Retained value, gradient, and Hessian are one spline evaluation."""
    radial = np.linspace(0.2, 2.2, 17)
    vertical = np.linspace(-1.0, 1.0, 17)
    mesh_r, mesh_z = np.meshgrid(radial, vertical)
    values = (mesh_r - 1.2) ** 2 + 1.3 * (mesh_z + 0.25) ** 2
    fixed = _locator(radial, vertical)
    census = fixed.candidate_table_status(_flat_field(values))
    index = int(np.flatnonzero(np.asarray(census["retained_valid"][0]))[0])
    position = census["retained_candidate"][0, index, :2]
    spline = fit_tensor_spline(
        jnp.asarray(radial), jnp.asarray(vertical), jnp.asarray(values)
    )
    direct = spline.evaluate(position[0], position[1])
    determinant = (
        direct.radial_second_derivative * direct.vertical_second_derivative
        - direct.mixed_derivative**2
    )

    np.testing.assert_allclose(
        np.asarray(census["retained_candidate"][0, index, 2]),
        np.asarray(direct.value),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(census["retained_spline_gradient"][0, index]),
        [direct.radial_derivative, direct.vertical_derivative],
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(census["retained_spline_hessian_determinant"][0, index]),
        np.asarray(determinant),
        rtol=0.0,
        atol=2.0e-14,
    )


def test_forward_census_deduplicates_polished_same_root():
    """Numerically coincident polished roots retain one origin and multiplicity."""
    position = jnp.asarray(
        ((1.0, -0.2), (1.0 + 8.0e-14, -0.2 - 4.0e-14), (1.5, 0.4)),
        dtype=jnp.float64,
    )
    valid = jnp.asarray((True, True, False))
    uncertainty = jnp.asarray((6.0e-14, 6.0e-14, 6.0e-14))

    representative, multiplicity, parent = _FixedDesignNull2D._deduplicate_type(
        position, valid, uncertainty
    )

    np.testing.assert_array_equal(representative, [True, False, False])
    np.testing.assert_array_equal(multiplicity, [2, 0, 0])
    np.testing.assert_array_equal(parent, [0, 0, -1])


def test_forward_census_preserves_nearby_distinct_saddles():
    """Sub-pitch roots outside numerical uncertainty remain separate."""
    cell_pitch = 0.12
    separation = 0.08
    assert separation < 1.5 * cell_pitch
    position = jnp.asarray(((1.0, 0.0), (1.0 + separation, 0.0)))
    valid = jnp.asarray((True, True))
    uncertainty = jnp.asarray((2.0e-12, 3.0e-12))

    representative, multiplicity, parent = _FixedDesignNull2D._deduplicate_type(
        position, valid, uncertainty
    )

    np.testing.assert_array_equal(representative, [True, True])
    np.testing.assert_array_equal(multiplicity, [1, 1])
    np.testing.assert_array_equal(parent, [0, 1])


def test_forward_census_has_eager_jit_vmap_parity():
    """Scalar, compiled, and one-compilation batched censuses agree exactly."""
    radial = np.linspace(0.2, 2.2, 17)
    vertical = np.linspace(-1.0, 1.0, 17)
    fixed = _locator(radial, vertical)
    first = _flat_field(_quadratic_saddle(radial, vertical, (1.18, -0.04)))
    second = _flat_field(_quadratic_saddle(radial, vertical, (1.24, 0.03)))
    fields = jnp.stack((first, second))
    keys = (
        "raw_ring_count",
        "same_root_count",
        "retained_candidate",
        "retained_valid",
        "retained_representative_origin_index",
        "retained_multiplicity",
        "retained_spline_gradient",
        "retained_spline_hessian_determinant",
    )

    eager = [fixed.candidate_table_status(field) for field in fields]
    compiled_reader = jax.jit(fixed.candidate_table_status)
    compiled = [compiled_reader(field) for field in fields]
    batch_reader = jax.jit(jax.vmap(fixed.candidate_table_status))
    before = batch_reader._cache_size()
    batched = batch_reader(fields)
    after = batch_reader._cache_size()
    repeated = batch_reader(fields)

    assert after - before == 1
    assert batch_reader._cache_size() == after
    for key in keys:
        expected = np.stack([np.asarray(item[key]) for item in eager])
        np.testing.assert_array_equal(
            np.stack([np.asarray(item[key]) for item in compiled]), expected
        )
        np.testing.assert_array_equal(np.asarray(batched[key]), expected)
        np.testing.assert_array_equal(np.asarray(repeated[key]), expected)


def test_forward_census_bank_rows_do_not_overflow():
    """The bank retains the measured physical root census without truncation."""
    receipt = bank_census_receipt(_bank_path(), "test")
    for row in receipt["rows"]:
        counts = row["counts_o_x"]
        assert 8 <= counts["same_root_count"][0] <= 9
        assert 5 <= counts["same_root_count"][1] <= 7
        assert not any(row["overflow_o_x"])
        if row["identity"] == "21985/51":
            assert counts["raw_ring_count"][0] == 10
            assert counts["typed_count"][0] == 9

    assert len(receipt["rows"]) == 12


def _diverted_oracle_read(requested_cells: int):
    """Return the diverted-oracle carrier mesh, its read, and its census.

    ``requested_cells`` is the carrier resolution argument: a negative value
    refines the hex carrier, so the realised node count grows with its
    magnitude. The diverted source declares no closed zero-flux surface of its
    own -- its field coefficient is negative, so the surface-node fallback of
    the separatrix sampler is non-finite and collapses to a single vertex --
    and the analytic reference is the object carrying the core boundary, which
    the production route passes.
    """
    from benchmarks.solovev_certificate import _case, _exact_state
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture

    carrier_case, source_case, exact = _case("diverted-jump-bearing")
    machine = oracle_fixture.cached_machine(
        carrier_case,
        requested_cells,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = _exact_state("diverted-jump-bearing", exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state, analytic=exact
    )
    coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(empty_operator, coefficients)
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    _masks, state = operator.read(jnp.asarray(oracle_state))
    grid_flux = jnp.asarray(oracle_state[: len(machine.node)])
    census = operator._fixed_design_topology.grid.candidate_table_status(grid_flux)
    return carrier_case, machine, operator, state, census


def test_forward_census_exact_diverted_oracle():
    """The certificate ladder resolves only nulls contained by its nodal rings."""
    from benchmarks.solovev_certificate import AXIS_M, X_POINT_M

    _carrier, coarse_machine, coarse_operator, coarse_state, coarse_census = (
        _diverted_oracle_read(-110)
    )
    coarse_pitch = float(np.sqrt(np.median(np.asarray(coarse_machine.area))))
    coarse_grid = coarse_operator._fixed_design_topology.grid
    nearest_saddle_cell = int(
        np.argmin(
            np.linalg.norm(
                np.asarray(coarse_grid.locator.physical_origin) - X_POINT_M,
                axis=1,
            )
        )
    )

    assert len(coarse_machine.node) == 136
    assert not coarse_grid.structured
    assert not bool(coarse_census["spline_authored"])
    np.testing.assert_allclose(np.asarray(coarse_state.axis), AXIS_M, atol=coarse_pitch)
    assert np.all(np.isnan(np.asarray(coarse_state.x_point)))
    assert not bool(coarse_state.diverted)
    # raw_ring_count measures the two ring classes that admit a candidate:
    # a stencil ring with no crossing holds an extremum, one with four holds a
    # saddle.  The machine mesh is the carrier's, so its rings straddle the
    # diverted separatrix almost everywhere (54 of 55 rings cross twice) and
    # exactly one ring lies wholly inside it, at the axis.  No ring has four
    # crossings at either rung, so the census resolves the axis and no saddle,
    # which is the ladder's own rule: a null is retained only where a nodal
    # ring contains it.
    np.testing.assert_array_equal(coarse_census["raw_ring_count"], [1, 0])
    np.testing.assert_array_equal(coarse_census["candidate_count"], [1, 0])
    assert int(coarse_census["ring_crossing_count"][nearest_saddle_cell]) == 2
    assert bool(coarse_census["ring_resolution_limited"][nearest_saddle_cell])
    assert not np.any(np.asarray(coarse_census["overflow"]))

    _carrier, fine_machine, fine_operator, fine_state, fine_census = (
        _diverted_oracle_read(-300)
    )
    fine_pitch = float(np.sqrt(np.median(np.asarray(fine_machine.area))))

    assert len(fine_machine.node) == 342
    assert not fine_operator._fixed_design_topology.grid.structured
    assert not bool(fine_census["spline_authored"])
    np.testing.assert_allclose(np.asarray(fine_state.axis), AXIS_M, atol=fine_pitch)
    # Refining the rung moves the axis and does not reach the saddle: the
    # divertor X-point sits outside every stencil ring at this pitch too, so
    # the read stays limited with a non-finite x_point and reports no
    # diversion.  The ladder therefore has no rung here whose mesh contains
    # the saddle, and the positive control for the containment rule is the
    # axis pair above rather than a resolved saddle.
    assert np.all(np.isnan(np.asarray(fine_state.x_point)))
    assert not bool(fine_state.diverted)
    np.testing.assert_array_equal(fine_census["raw_ring_count"], [1, 0])
    np.testing.assert_array_equal(fine_census["candidate_count"], [1, 0])
    assert not np.any(np.asarray(fine_census["overflow"]))


# The containment rule admits a saddle only where a nodal ring *contains* it,
# which for a six-neighbour ring on a hex carrier needs a ring centre within
# about a quarter of a cell pitch of the saddle -- further out the ring sees two
# sign changes rather than four, and the read stays limited.
ADMISSION_CENTRE_DISTANCE_IN_PITCH = 0.25
DIVERTED_ORACLE_RUNG_CELLS = (110, 300)


def _diverted_oracle_ring_geometry(requested_cells: int):
    """Return the carrier mesh, its pitch, and its ring centres.

    The flux read is deliberately not taken: the containment rule's admission
    window is a question about where a ring centre can sit, and the carrier's
    ring centres are fixed by its mesh alone.
    """
    from benchmarks.solovev_certificate import _case
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture

    carrier_case, source_case, _exact = _case("diverted-jump-bearing")
    machine = oracle_fixture.cached_machine(
        carrier_case,
        -requested_cells,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    operator = oracle_fixture.forward_operator(source_case, machine)
    centres = np.asarray(
        operator._fixed_design_topology.grid.locator.physical_origin,
        dtype=np.float64,
    )
    pitch = float(np.sqrt(np.median(np.asarray(machine.area))))
    return machine, pitch, centres


def test_forward_census_diverted_oracle_pitch_ladder_admits_no_saddle():
    """No carrier rung places a ring centre near the diverted saddle.

    The diverted oracle fixture censuses the analytic diverted flux on the
    carrier's hex mesh while the carrier's wall stays put. The wall is not the
    diverted separatrix, and the analytic X-point lies outside it at both rungs,
    so it lies outside the meshed domain: a ring centre is a mesh cell centre,
    and refinement does not move the wall that excludes the saddle. The
    measured ring-centre distances are what settle the question -- at 136 cells
    the nearest of 55 ring centres is 2.51 cell pitches from the saddle, and at
    342 cells the nearest of 203 is 1.91, both far outside the quarter-pitch
    admission window. Refinement is not a route to that window either: the wall
    does not refine while the pitch shrinks, so the saddle's distance from the
    wall, held in metres, grows without limit when counted in cell pitches. This fixture
    therefore cannot give the containment rule a positive control that is a
    saddle, and its control stays the axis pair. The control for a saddle
    belongs on the fixture whose mesh is built from the diverted separatrix,
    benchmarks/topology_read_resolution_ladder.py.
    """
    from benchmarks.solovev_certificate import X_POINT_M
    from shapely.geometry import Point, Polygon

    saddle = np.asarray(X_POINT_M, dtype=np.float64)
    rows = []
    for requested_cells in DIVERTED_ORACLE_RUNG_CELLS:
        machine, pitch, centres = _diverted_oracle_ring_geometry(requested_cells)
        distance = np.linalg.norm(centres - saddle, axis=1)
        nearest = int(np.argmin(distance))
        wall = np.asarray(machine.wall_node, dtype=np.float64)
        rows.append(
            {
                "requested_cells": requested_cells,
                "realised_cells": len(machine.node),
                "wall_nodes": len(wall),
                "ring_count": int(centres.shape[0]),
                "pitch_m": pitch,
                "nearest_ring_centre_to_saddle_m": float(distance[nearest]),
                "nearest_ring_centre_to_saddle_pitch": float(distance[nearest] / pitch),
                "saddle_inside_carrier_wall": bool(
                    Polygon(wall).contains(Point(float(saddle[0]), float(saddle[1])))
                ),
            }
        )
    print(
        "DIVERTED_ORACLE_PITCH_LADDER "
        + json.dumps(
            {
                "saddle_rz_m": saddle.tolist(),
                "admission_centre_distance_in_pitch": (
                    ADMISSION_CENTRE_DISTANCE_IN_PITCH
                ),
                "rungs": rows,
            },
            sort_keys=True,
        )
    )

    assert len(rows) == len(DIVERTED_ORACLE_RUNG_CELLS)
    # The carrier wall does not move with the rung: only the mesh refines.
    assert len({row["wall_nodes"] for row in rows}) == 1
    realised = [row["realised_cells"] for row in rows]
    assert realised == sorted(realised)
    assert realised[-1] > realised[0]
    for row in rows:
        assert not row["saddle_inside_carrier_wall"], row
        assert row["ring_count"] > 0, row
        assert (
            row["nearest_ring_centre_to_saddle_pitch"]
            > ADMISSION_CENTRE_DISTANCE_IN_PITCH
        ), row


@pytest.mark.parametrize("requested_cells", (110, 300, 500))
def test_forward_census_spline_wedge_current_oracle(requested_cells):
    """The production spline carrier preserves the analytic saddle-cell current."""
    from benchmarks import topology_read_resolution_ladder as topology_ladder
    from benchmarks import xpoint_cell_wedge_oracle as wedge_oracle
    from nova.equilibrium.separatrix_clip import AtomicCellMesh

    machine, _operator, _analytic = topology_ladder._machine_and_field(requested_cells)
    exact = topology_ladder.ANALYTIC
    x_point = np.asarray(exact.x_point, dtype=np.float64)
    magnetic_axis = np.asarray(exact.magnetic_axis, dtype=np.float64)
    boundary_flux = float(exact.flux(x_point[None, :])[0])
    axis_flux = float(exact.flux(magnetic_axis[None, :])[0])
    polarity = np.copysign(1.0, axis_flux - boundary_flux)
    cell, _candidates = wedge_oracle._xpoint_cell(machine, x_point)
    polygon = np.asarray(machine.cell_polygons[cell], dtype=np.float64)
    centre = np.asarray(machine.node[cell], dtype=np.float64)
    mesh = AtomicCellMesh.from_cells([polygon], centroids=centre[None, :])

    def analytic_level(points):
        shape = jax.ShapeDtypeStruct(points.shape[:-1], points.dtype)
        return jax.pure_callback(
            lambda value: np.asarray(
                polarity * (exact.flux(value) - boundary_flux),
                dtype=np.asarray(value).dtype,
            ),
            shape,
            points,
        )

    signed_flux = jnp.asarray(
        polarity * (exact.flux(mesh.node_coordinates) - boundary_flux)
    )
    wedges = mesh.traced_saddle_wedges(
        signed_flux,
        saddle_vertex=jnp.asarray(x_point),
        core_reference=jnp.asarray(magnetic_axis),
        curve_evaluator=analytic_level,
    )
    profile = wedge_oracle._AnalyticCurrentProfile(
        source_parameter=float(exact.source_parameter),
        flux_scale=float(exact.flux_scale_per_radian_wb),
        major_radius=float(exact.major_radius),
    )
    zero = wedge_oracle._ZeroCurrentProfile()
    profiles = (profile, zero, zero, zero)
    measured = wedge_oracle.saddle_wedge_current_moments(
        wedges, wedge_oracle._FluxPlaceholder(), profiles
    )
    actual = np.stack(
        (
            np.asarray(measured.cell_current)[0],
            np.asarray(measured.radial_moment)[0],
            np.asarray(measured.vertical_moment)[0],
        ),
        axis=1,
    )
    expected = np.asarray(
        [
            wedge_oracle._analytic_polygon_moments(
                wedge_oracle._support_vertices(wedges, slot), centre, item
            )
            for slot, item in enumerate(profiles)
        ]
    )
    relative_error = np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0e-12)
    counts = np.asarray(wedges.vertex_count)[0]
    vertices = np.asarray(wedges.support_vertices)[0]

    print(
        "XPOINT_WEDGE_ORACLE "
        + json.dumps(
            {
                "requested_cells": requested_cells,
                "realised_cells": len(machine.node),
                "xpoint_cell": cell,
                "core_current_relative_error": float(relative_error[0, 0]),
                "private_flux_current_a": float(actual[1, 0]),
                "common_sol_current_a": actual[2:, 0].tolist(),
                "vertex_counts": counts.tolist(),
            },
            sort_keys=True,
        )
    )

    assert bool(wedges.saddle[0])
    assert relative_error[0, 0] <= 1.0e-6
    np.testing.assert_array_equal(actual[1:], 0.0)
    for wedge, count in zip(vertices, counts, strict=True):
        np.testing.assert_array_equal(wedge[0], x_point)
        np.testing.assert_array_equal(wedge[count:], 0.0)


def test_forward_census_refuses_polish_outside_detected_cell():
    """A sign seed one cell from its root is reported but never retained."""
    with np.load(_bank_path(), allow_pickle=True) as archive:
        index = next(
            index
            for index, row in _bank_rows(archive)
            if row["identity"] == "21985/51" and row["arm"] == "pure"
        )
        _fixed, _values, census = _bank_census(archive, index)

    rejected = np.asarray(census["polish_rejected"][1])
    displacement = np.asarray(census["requested_displacement"])
    cell_width = np.asarray(census["cell_width"])
    representative = np.asarray(census["representative_mask"][1])
    outside = rejected & (displacement >= cell_width)

    assert np.count_nonzero(outside) == 1
    assert not np.any(representative[outside])
    assert int(census["raw_ring_count"][1]) == 8
    assert int(census["polished_count"][1]) == 7
