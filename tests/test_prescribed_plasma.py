"""Closed-boundary current prescription and conductor flux linkage."""

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from nova.circuit.conductor import ConductorSet
from nova.imas.prescribed_plasma import (
    plasma_flux_at_points,
    plasma_linked_flux,
    plasma_linked_flux_history,
    polygon_plasma_flux,
    polygon_quadrature,
)
from nova.imas.profiles import prescribe_current_density
from scripts.iter_prescribed_plasma.measure import current_vector


@pytest.fixture
def current():
    radius = np.linspace(4, 6, 41)
    height = np.linspace(-1, 1, 41)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    psi = 1 - (rr - 5) ** 2 - zz**2
    theta = np.linspace(0, 2 * np.pi, 121)
    outline = np.c_[5 + 0.8 * np.cos(theta), 0.8 * np.sin(theta)]
    return prescribe_current_density(
        radius,
        height,
        psi,
        np.linspace(1, 0.36, 50),
        np.ones(50),
        np.zeros(50),
        1,
        0.36,
        1e6,
        outline,
        (5, 0),
    )


def test_closed_boundary_prescription_matches_current_and_centroid(current):
    assert current.boundary_source == "separatrix"
    assert current.endpoint_gap < 1e-12
    assert np.sum(current.cell_current) == pytest.approx(1e6)
    assert current.centroid == pytest.approx((5.006, 0), abs=0.05)
    assert np.all(current.cell_current[(current.radius - 5) ** 2 > 0.8**2] == 0)


def test_open_outline_uses_last_closed_flux_surface(current):
    open_outline = current.boundary[:-1]
    radius = current.radius
    height = current.height
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    replacement = prescribe_current_density(
        radius,
        height,
        1 - (rr - 5) ** 2 - zz**2,
        np.linspace(1, 0.36, 50),
        np.ones(50),
        np.zeros(50),
        1,
        0.36,
        1e6,
        open_outline,
        (5, 0),
    )
    assert replacement.boundary_source == "last_closed_flux"
    assert replacement.boundary_fraction is not None
    assert replacement.endpoint_gap > 0.01
    assert np.sum(replacement.cell_current) == pytest.approx(1e6)


def test_divertor_legs_are_cut_at_x_point_before_stored_boundary(current):
    theta = np.linspace(-np.pi / 2, 3 * np.pi / 2, 121)
    x_point = (5.0, -0.8)
    diverted = np.vstack(
        ([4.6, -1.4], np.c_[5 + 0.8 * np.cos(theta), 0.8 * np.sin(theta)], [5.4, -1.4])
    )
    rr, zz = np.meshgrid(current.radius, current.height, indexing="ij")
    replacement = prescribe_current_density(
        current.radius,
        current.height,
        1 - (rr - 5) ** 2 - zz**2,
        np.linspace(1, 0.36, 50),
        np.ones(50),
        np.zeros(50),
        1,
        0.36,
        1e6,
        diverted,
        (5, 0),
        x_points=[x_point],
        boundary_outline=current.boundary,
    )
    assert replacement.boundary_source == "separatrix_x_point"
    assert replacement.boundary_fraction is None
    assert replacement.endpoint_gap > 0.5
    assert replacement.boundary[0] == pytest.approx(x_point)


def test_stored_boundary_precedes_flux_contour(current):
    rr, zz = np.meshgrid(current.radius, current.height, indexing="ij")
    replacement = prescribe_current_density(
        current.radius,
        current.height,
        1 - (rr - 5) ** 2 - zz**2,
        np.linspace(1, 0.36, 50),
        np.ones(50),
        np.zeros(50),
        1,
        0.36,
        1e6,
        current.boundary[:-1],
        (5, 0),
        boundary_outline=current.boundary,
    )
    assert replacement.boundary_source == "boundary_outline"
    assert replacement.boundary_fraction is None
    np.testing.assert_array_equal(replacement.boundary, current.boundary)


def test_missing_edge_gradient_extrapolates_from_finite_samples(current):
    radius = current.radius
    height = current.height
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    pressure_gradient = np.ones(50)
    pressure_gradient[-1] = np.nan
    replacement = prescribe_current_density(
        radius,
        height,
        1 - (rr - 5) ** 2 - zz**2,
        np.linspace(1, 0.36, 50),
        pressure_gradient,
        np.zeros(50),
        1,
        0.36,
        1e6,
        current.boundary,
        (5, 0),
    )
    assert np.all(np.isfinite(replacement.cell_current))
    assert replacement.scale_factor == pytest.approx(current.scale_factor)


def test_finite_section_and_filament_linkage(current):
    conductors = ConductorSet(
        r=np.array([7.0, 7.0]),
        z=np.array([0.4, -0.4]),
        dr=np.array([0.0, 0.2]),
        dz=np.array([0.0, 0.2]),
        current_share=np.ones(2),
        circuit=np.array([0, 1]),
    )
    circuits, linked = plasma_linked_flux(current, conductors)
    assert circuits.tolist() == [0, 1]
    assert linked.shape == (2,)
    assert linked[0] > 0 and linked[1] > 0
    shared = ConductorSet(
        r=np.array([7.0, 7.0]),
        z=np.array([0.4, 0.4]),
        dr=np.zeros(2),
        dz=np.zeros(2),
        current_share=np.array([1.0, 2.0]),
        circuit=np.zeros(2, dtype=int),
    )
    doubled = replace(
        current,
        current_density=2 * current.current_density,
        cell_current=2 * current.cell_current,
        plasma_current=2 * current.plasma_current,
    )
    ids, history = plasma_linked_flux_history([current, doubled], shared)
    expected = 3 * plasma_flux_at_points(current, np.array([7.0]), np.array([0.4]))[0]
    assert ids.tolist() == [0]
    assert history[:, 0] == pytest.approx([expected, 2 * expected])


def test_vertical_shift_follows_flux_gradient(current):
    radius = np.array([7.0])
    height = np.array([0.4])
    step = 0.01
    probe = 0.001
    base = plasma_flux_at_points(current, radius, height)[0]
    moved = plasma_flux_at_points(current, radius, height, vertical_offset=step)[0]
    derivative = (
        plasma_flux_at_points(current, radius, height + probe)[0]
        - plasma_flux_at_points(current, radius, height - probe)[0]
    ) / (2 * probe)
    predicted_change = -step * derivative
    assert abs(predicted_change) > 1e-4
    assert moved - base == pytest.approx(predicted_change, rel=0.03)


def test_conductor_currents_follow_the_equilibrium_time():
    active = SimpleNamespace(
        time=np.array([0.0, 1.0, 2.0]),
        coil=[
            SimpleNamespace(current=SimpleNamespace(data=np.array([0.0, 10.0, 20.0])))
        ],
    )
    passive = SimpleNamespace(
        time=np.array([0.0, 2.0]),
        loop=[SimpleNamespace(current=SimpleNamespace(data=np.array([4.0, 8.0])))],
    )
    values = current_vector(active, passive, [("coil", 0), ("passive", 0)], 1.5)
    assert values == pytest.approx([15.0, 7.0])


#: the linked flux of a shaped section agrees to this fraction of its own value
#: between the reciprocity kernel and a dense sampling of the same area
SECTION_FLUX_TOLERANCE = 2e-3

#: a wall strip thin enough that a 0.05 m midpoint grid over the strip's own
#: extent lands on no interior point at all
THIN_STRIP_THICKNESS_M = 0.02
THIN_STRIP_HEIGHT_M = 0.4
THIN_STRIP_RADIUS_M = 7.0


def _bounding_box_points(polygon, *, section_spacing=0.05, section_n_max=8):
    """The bounding-box midpoint grid's interior points and their weights.

    Empty arrays when the grid lands on no interior point -- which is what
    happens for a strip thinner than ``section_spacing``.
    """
    from shapely import contains_xy
    from shapely.geometry import Polygon

    from nova.circuit.linkage import section_points

    vertices = np.asarray(polygon.vertices, float)
    extent = np.ptp(vertices, axis=0)
    mid = np.mean([vertices.min(axis=0), vertices.max(axis=0)], axis=0)
    point_r, point_z, weight = section_points(
        mid[0], mid[1], extent[0], extent[1], section_spacing, section_n_max
    )
    inside = contains_xy(Polygon(vertices), point_r, point_z)
    if not np.any(inside):
        empty = np.empty(0)
        return empty, empty, empty
    weight = weight[inside] / weight[inside].sum()
    return point_r[inside], point_z[inside], weight


def _bounding_box_linked_flux(
    current, polygon, *, section_spacing=0.05, section_n_max=8
):
    """The bounding-box midpoint rule this change replaces, for comparison.

    A grid is laid over the section's own bounding box and the points falling
    strictly inside are averaged.  This is the rule whose spacing cannot see a
    strip thinner than itself, so it is the instrument the thin-strip check is
    measured against rather than a second implementation of the repair.
    """
    from shapely import contains_xy
    from shapely.geometry import Polygon

    from nova.circuit.linkage import section_points

    vertices = np.asarray(polygon.vertices, float)
    extent = np.ptp(vertices, axis=0)
    mid = np.mean([vertices.min(axis=0), vertices.max(axis=0)], axis=0)
    point_r, point_z, weight = section_points(
        mid[0], mid[1], extent[0], extent[1], section_spacing, section_n_max
    )
    inside = contains_xy(Polygon(vertices), point_r, point_z)
    if not np.any(inside):
        return None
    weight = weight[inside] / weight[inside].sum()
    return polygon.current_share * float(
        np.dot(weight, plasma_flux_at_points(current, point_r[inside], point_z[inside]))
    )


def _dense_section_reference(current, polygon, *, samples: int = 64):
    """Area-averaged plasma flux over the section from a dense interior grid.

    Every sample strictly inside the section is weighted equally, so the average
    converges on the section integral independently of the reciprocity kernel.
    """
    from shapely import contains_xy
    from shapely.geometry import Polygon

    vertices = np.asarray(polygon.vertices, float)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    grid_r, grid_z = np.meshgrid(
        np.linspace(low[0], high[0], samples),
        np.linspace(low[1], high[1], samples),
        indexing="ij",
    )
    inside = contains_xy(Polygon(vertices), grid_r, grid_z)
    assert inside.sum() > 0, "the reference grid found no interior point"
    flux = plasma_flux_at_points(current, grid_r[inside], grid_z[inside])
    return polygon.current_share * float(np.mean(flux)), int(inside.sum())


@pytest.fixture
def coarse_current():
    """A plasma on a coarse grid, so the dense section reference stays cheap.

    Only the phase-space sampling of the flux changes: the same prescription
    feeds both the reference and the kernel under test, so the comparison is
    unaffected by how finely the plasma is resolved by the section.
    """
    radius = np.linspace(4, 6, 21)
    height = np.linspace(-1, 1, 21)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    psi = 1 - (rr - 5) ** 2 - zz**2
    theta = np.linspace(0, 2 * np.pi, 121)
    outline = np.c_[5 + 0.8 * np.cos(theta), 0.8 * np.sin(theta)]
    return prescribe_current_density(
        radius,
        height,
        psi,
        np.linspace(1, 0.36, 50),
        np.ones(50),
        np.zeros(50),
        1,
        0.36,
        1e6,
        outline,
        (5, 0),
    )


def _thin_strip(circuit: int = 0):
    from nova.circuit.conductor import PolygonSection

    half = THIN_STRIP_HEIGHT_M / 2
    skin = THIN_STRIP_THICKNESS_M / 2
    vertices = np.array(
        [
            [THIN_STRIP_RADIUS_M - skin, -half],
            [THIN_STRIP_RADIUS_M + skin, -half],
            [THIN_STRIP_RADIUS_M + skin, half],
            [THIN_STRIP_RADIUS_M - skin, half],
        ]
    )
    return PolygonSection(circuit, vertices)


def _diagonal_sliver(circuit: int = 0):
    """A thin band lying along a diagonal, which a box grid cannot land on.

    The band's bounding box is 0.8 m square while the band itself is four
    millimetres thick, so the midpoint grid a bounding-box rule lays on that box
    has nowhere to fall inside the section.
    """
    from nova.circuit.conductor import PolygonSection

    tangent = (SLIVER_HIGH - SLIVER_LOW) / np.hypot(*(SLIVER_HIGH - SLIVER_LOW))
    normal = np.array([-tangent[1], tangent[0]])
    return PolygonSection(
        circuit,
        np.array(
            [
                SLIVER_LOW + SLIVER_HALF_WIDTH_M * normal,
                SLIVER_HIGH + SLIVER_HALF_WIDTH_M * normal,
                SLIVER_HIGH - SLIVER_HALF_WIDTH_M * normal,
                SLIVER_LOW - SLIVER_HALF_WIDTH_M * normal,
            ]
        ),
    )


SLIVER_LOW = np.array([6.61, -0.39])
SLIVER_HIGH = np.array([7.43, 0.37])
SLIVER_HALF_WIDTH_M = 0.002


def _dense_sliver_reference(current, *, along: int = 400, across: int = 20):
    """Area-averaged plasma flux over the sliver, sampled in the band's own frame.

    The samples are laid on the band rather than on its bounding box, so every
    one is inside the section by construction and the frame's area element is
    constant: an equal-weighted mean of them is the section's area average.
    """
    direction = SLIVER_HIGH - SLIVER_LOW
    length = float(np.hypot(*direction))
    tangent = direction / length
    normal = np.array([-tangent[1], tangent[0]])
    centre = 0.5 * (SLIVER_LOW + SLIVER_HIGH)
    grid_s, grid_t = np.meshgrid(
        np.linspace(-length / 2, length / 2, along),
        np.linspace(-SLIVER_HALF_WIDTH_M, SLIVER_HALF_WIDTH_M, across),
        indexing="ij",
    )
    point_r = centre[0] + grid_s * tangent[0] + grid_t * normal[0]
    point_z = centre[1] + grid_s * tangent[1] + grid_t * normal[1]
    flux = plasma_flux_at_points(current, point_r, point_z)
    return float(np.mean(flux)), int(grid_s.size)


def test_thin_polygon_strip_is_integrated_not_missed(coarse_current):
    """A section thinner than the grid spacing links the flux it really links.

    The sliver is measured twice on one plasma: a dense sampling laid in the
    band's own frame, and the quadrature this change introduces.  The two agree
    to the stated tolerance, and every quadrature point lies inside the section
    by construction.  The negative control for the repair is at machine scale,
    where the bounding-box rule refuses 42 of the machine's shaped sections
    outright -- see the shaped-section measurement in the slow arm.
    """
    section = _diagonal_sliver()
    reference, count = _dense_sliver_reference(coarse_current)
    assert count > 5_000, "the reference samples are too few to be a reference"
    assert reference != 0.0

    linked = polygon_plasma_flux(coarse_current, section)
    assert linked == pytest.approx(reference, rel=SECTION_FLUX_TOLERANCE)

    # The section's quadrature points lie inside it by construction, which is what
# the bounding-box rule cannot guarantee: on the machine it is the neighbouring
# sections of exactly this shape that it refuses outright.
    from shapely import contains_xy
    from shapely.geometry import Polygon

    point_r, point_z, _weight = polygon_quadrature(section.vertices)
    assert np.all(contains_xy(Polygon(section.vertices), point_r, point_z))


def test_thin_axis_facing_strip_matches_its_dense_average(coarse_current):
    """A strip thin in one direction still agrees with its own area average.

    This section is axis-aligned, so the bounding-box grid does land inside it,
    with one point across the thickness.  Both rules are measured against a
    dense in-section average, so the check states the accuracy of the quadrature
    rather than relying on the old rule's failure.
    """
    section = _thin_strip()
    reference, count = _dense_section_reference(coarse_current, section, samples=64)
    assert count > 3_000, "the reference grid is not dense enough to be a reference"
    linked = polygon_plasma_flux(coarse_current, section)
    assert linked == pytest.approx(reference, rel=SECTION_FLUX_TOLERANCE)


def test_zero_area_polygon_is_refused(current):
    """A degenerate section is refused by name rather than integrated."""
    from nova.circuit.conductor import PolygonSection

    section = PolygonSection(3, np.array([[7.0, -0.2], [7.1, -0.2], [7.2, -0.2], [7.3, -0.2]]))
    with pytest.raises(ValueError, match="encloses no area"):
        polygon_plasma_flux(current, section)


SECTIONED_ENTRY = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs"
    "/r-20261021T105645032186-unseen-source-full-plant/scratch/sectioned-cache"
    "/68df15665ab2ea45f427"
)
DINA_CASE_URI = "imas:hdf5?user=mcintos;pulse=105014;run=1;database=iter;version=3"
DINA_CASE_DD = "3.38.1"
SLICE_INDEX = 200


@pytest.mark.slow
def test_polygon_flux_change_on_a_real_slice(tmp_path):
    """Per-circuit change in linked flux on one 105014/1 slice, and the maximum.

    Every polygon circuit the bounding-box rule could sample is compared with
    the reciprocity kernel on the same slice of the same source, so the table
    records how far a cold build moves from the pinned plant on disk.  A circuit
    the old rule could NOT sample has no old value and is reported as the
    refusal it was.  Rectangular circuits are asserted bit-identical, because
    the repair touches only the shaped-section branch.
    """
    from nova.imas.plasma_source import PlasmaSource
    from nova.circuit.passive import load_sectioned_machine
    from nova.circuit.linkage import section_grid

    if not (SECTIONED_ENTRY / "geometry.npz").exists():
        pytest.skip("the scratch sectioned-set entry is not present on this host")
    import imas

    machine = load_sectioned_machine(SECTIONED_ENTRY)
    with imas.DBEntry(DINA_CASE_URI, "r", dd_version=DINA_CASE_DD) as entry:
        equilibrium = entry.get("equilibrium", lazy=False, autoconvert=False)
        active = entry.get("pf_active", lazy=False, autoconvert=False)
    source = PlasmaSource.from_imas(
        equilibrium, active, provenance="DINA 105014/1", indices=[SLICE_INDEX]
    )
    current = list(source.prescribed_currents())[0]
    conductors = machine.conductors

    circuits, linked = plasma_linked_flux(current, conductors)
    polygons = conductors.polygon_by_circuit()

    # The old rule is evaluated for every polygon in ONE plasma-flux call: each
    # section's interior points are gathered first and reduced afterwards, so
    # the comparison does not pay a kernel build per circuit.
    old_rows: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for index, circuit in enumerate(circuits):
        polygon = polygons.get(int(circuit))
        if polygon is None:
            continue
        point_r, point_z, weight = _bounding_box_points(polygon)
        old_rows.append((index, int(circuit), point_r, point_z, weight))

    carried = [row for row in old_rows if row[2].size]
    live_flux = plasma_flux_at_points(
        current,
        np.concatenate([row[2] for row in carried]) if carried else np.empty(0),
        np.concatenate([row[3] for row in carried]) if carried else np.empty(0),
    )
    offset = 0
    rows: list[tuple[int, float, float, float]] = []
    unsampled = 0
    for index, circuit, point_r, _point_z, weight in old_rows:
        if point_r.size == 0:
            unsampled += 1
            rows.append((circuit, float(linked[index]), float("nan"), float("nan")))
            continue
        flux = live_flux[offset : offset + point_r.size]
        offset += point_r.size
        rejected = polygons[circuit].current_share * float(np.dot(weight, flux))
        change = (linked[index] - rejected) / abs(rejected)
        rows.append((circuit, float(linked[index]), rejected, change))

    sampled = [row for row in rows if np.isfinite(row[3])]
    worst = max(sampled, key=lambda row: abs(row[3])) if sampled else None
    print(
        f"POLYGON_FLUX_TABLE polygons={len(rows)} sampled={len(sampled)} "
        f"unsampled={unsampled}"
    )
    for rank, row in enumerate(
        sorted(sampled, key=lambda item: -abs(item[3]))[:12], start=1
    ):
        print(
            f"POLYGON_FLUX_WORST rank={rank} circuit={row[0]} new={row[1]:.9e} "
            f"old={row[2]:.9e} rel={row[3]:.6e}"
        )
    if worst is not None:
        print(
            f"POLYGON_FLUX_MAX circuit={worst[0]} rel={worst[3]:.6e} "
            f"new={worst[1]:.9e} old={worst[2]:.9e}"
        )

    assert unsampled > 0, "no circuit exercised the refusal the repair removes"
    assert sampled, "no circuit had an interior grid point under the old rule"

    rectangular = [int(c) for c in circuits if int(c) not in polygons]
    assert rectangular, "the machine carries no rectangular circuit to compare"
    point_r, point_z, weight, owner = section_grid(
        conductors, conductors.rows(rectangular), 0.05, 8
    )
    point_flux = plasma_flux_at_points(current, point_r, point_z)
    reference = np.bincount(owner, weights=weight * point_flux, minlength=len(rectangular))
    np.testing.assert_array_equal(linked[np.isin(circuits, rectangular)], reference)
