"""Poloidal view in the locked style, measured on a solved Solov'ev frame.

The frame is the playable Solov'ev machine — the same 15-by-15 free-boundary
problem ``tests/test_reduced_newton.py`` pins — solved on the CPU lane.  The
gate pins the styled figure contract: no axis, grid, frame or bounding box;
the first wall as one dark grey polyline; each coil outline with no fill
columns drawn from the machine's own conductor geometry (sixteen fitted
conductor outlines centred on the fixture conductors, never a decorative
ring); the plasma as light purple cell polygons at one alpha with every
separatrix-cut cell drawn as its solved clipped polygon, so no cell polygon
crosses the separatrix by more than the lattice pitch and each clipped vertex
lies within that pitch of the separatrix; dark grey unfilled psi contours at a
fixed level count with the separatrix outermost and no filled flux image; the
O-point and primary X-point as distinct markers with the secondary X-point and
strike points lighter, every topology marker inside the closed first wall;
every served channel source initialised with its bound columns; and a styled
PNG of the frame committed under
``docs/figures/playable-forward-solve/view/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from nova.utilities.importmanager import skip_import

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEW_FIGURE = REPO_ROOT / "docs/figures/playable-forward-solve/view"

with skip_import("bokeh"):
    from bokeh.models import Image, GlyphRenderer

    from apps.pulsedesign.poloidal_view import (
        N_CONTOURS,
        CONTOUR_COLOR,
        O_POINT_COLOR,
        PLASMA_FILL_ALPHA,
        PLASMA_FILL_COLOR,
        WALL_COLOR,
        WALL_LINE_WIDTH,
        X_POINT_COLOR,
        X_POINT_SECONDARY_COLOR,
        channel_sources,
        compensation_figure,
        contour_levels,
        keyframe_receipt,
        poloidal_channels,
        poloidal_figure,
    )


@pytest.fixture(scope="module")
def solved_frame():
    """Solve the playable Solov'ev machine on the CPU lane as the styled frame."""
    from apps.playable.machines import build_machine

    machine = build_machine("solovev")
    equilibrium = machine.profile.solve(machine.seed, route="host")
    return machine, equilibrium


@pytest.fixture(scope="module")
def channels(solved_frame):
    """Return the styled channels assembled from the solved frame.

    The coil outlines are the fixture machine's own conductor geometry,
    carried beside its wall.
    """
    machine, equilibrium = solved_frame
    return poloidal_channels(
        equilibrium,
        machine.profile,
        wall=machine.wall,
        coils=machine.coils,
        n_contours=N_CONTOURS,
    )


def _styled_figure():
    """Return the styled figure bound to the initialised channel sources."""
    return poloidal_figure(channel_sources())


def _glyph(poloidal, kind: str, position: int = 0):
    """Return the nth glyph bound on the styled figure."""
    renderers = [
        renderer
        for renderer in poloidal.renderers
        if isinstance(renderer, GlyphRenderer) and type(renderer.glyph).__name__ == kind
    ]
    return renderers[position].glyph


def _bound_fields(glyph) -> set[str]:
    """Return the data-field column names one glyph binds."""
    fields = set()
    for value in glyph.properties_with_values().values():
        field = getattr(value, "field", None)
        if isinstance(field, str) and field:
            fields.add(field)
    return fields


def _css(colour):
    """Return the CSS string of a Bokeh colour constant."""
    if isinstance(colour, str):
        return colour
    serializer = getattr(colour, "to_css", None)
    return serializer() if serializer is not None else None


def _hex_luminance(colour: str) -> float:
    """Return a coarse perceptual luminance for a ``#rrggbb`` colour."""
    colour = colour.strip("#")
    channels = [int(colour[index : index + 2], 16) for index in (0, 2, 4)]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _reference_endpoint(point) -> tuple[float, float]:
    """Return the endpoint identity used by the replaced raster tracer."""
    return round(float(point[0]), 10), round(float(point[1]), 10)


def _reference_join_segments(segments) -> list[np.ndarray]:
    """Join raster-edge segments as the replaced renderer did."""
    neighbours = {}
    for index, (start, end) in enumerate(segments):
        neighbours.setdefault(_reference_endpoint(start), []).append(index)
        neighbours.setdefault(_reference_endpoint(end), []).append(index)
    used = [False] * len(segments)
    lines: list[np.ndarray] = []
    for start in range(len(segments)):
        if used[start]:
            continue
        used[start] = True
        line = [list(segments[start][0]), list(segments[start][1])]
        extended = True
        while extended:
            extended = False
            for end_index in (0, -1):
                endpoint = _reference_endpoint(line[end_index])
                for other in neighbours.get(endpoint, ()):
                    if used[other]:
                        continue
                    used[other] = True
                    first, second = segments[other]
                    following = (
                        second if _reference_endpoint(first) == endpoint else first
                    )
                    if end_index == 0:
                        line.insert(0, list(following))
                    else:
                        line.append(list(following))
                    extended = True
                    break
        lines.append(np.asarray(line))
    return lines


def _reference_level_lines(psi2d, radius, height, level) -> list[np.ndarray]:
    """Regenerate the replaced renderer's output for one raster level."""
    psi2d = np.asarray(psi2d, dtype=float)
    radius = np.asarray(radius, dtype=float)
    height = np.asarray(height, dtype=float)
    radial_step = radius[1] - radius[0]
    vertical_step = height[1] - height[0]
    segments = []
    for radial_index in range(psi2d.shape[0] - 1):
        for vertical_index in range(psi2d.shape[1] - 1):
            values = (
                psi2d[radial_index, vertical_index],
                psi2d[radial_index, vertical_index + 1],
                psi2d[radial_index + 1, vertical_index + 1],
                psi2d[radial_index + 1, vertical_index],
            )
            above = [value > level for value in values]
            if sum(above) in (0, 4):
                continue
            crossings = []
            edges = ((0, 1), (1, 2), (2, 3), (3, 0))
            for edge, (first, second) in enumerate(edges):
                if above[first] == above[second]:
                    continue
                fraction = (level - values[first]) / (values[second] - values[first])
                if edge == 0:
                    point = (
                        radius[radial_index],
                        height[vertical_index] + fraction * vertical_step,
                    )
                elif edge == 1:
                    point = (
                        radius[radial_index] + fraction * radial_step,
                        height[vertical_index + 1],
                    )
                elif edge == 2:
                    point = (
                        radius[radial_index + 1],
                        height[vertical_index + 1] - fraction * vertical_step,
                    )
                else:
                    point = (
                        radius[radial_index + 1] - fraction * radial_step,
                        height[vertical_index],
                    )
                crossings.append(point)
            if len(crossings) == 2:
                segments.append((crossings[0], crossings[1]))
            elif len(crossings) == 4:
                centre = 0.25 * sum(values)
                if centre <= level:
                    segments.extend(
                        ((crossings[0], crossings[3]), (crossings[1], crossings[2]))
                    )
                else:
                    segments.extend(
                        ((crossings[0], crossings[1]), (crossings[2], crossings[3]))
                    )
            else:  # pragma: no cover - a square has two or four crossings
                raise AssertionError(f"unexpected crossing count {len(crossings)}")
    return _reference_join_segments(segments)


def _polyline_separation(first: np.ndarray, second: np.ndarray) -> float:
    """Return the symmetric nearest-vertex separation of two polylines."""
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))


def _measure_contour_steps(equilibrium, repeats: int = 200) -> dict[str, float]:
    """Measure legacy, Nova and precomputed contour-channel steps in seconds."""
    from time import perf_counter

    import apps.pulsedesign.poloidal_view as poloidal_view

    raster = equilibrium.raster_flux
    radius = np.asarray(raster.radius, dtype=float)
    height = np.asarray(raster.height, dtype=float)
    shape = (len(radius), len(height))
    psi_norm = np.asarray(raster.psi_norm, dtype=float).reshape(shape)
    psi2d = np.asarray(raster.psi, dtype=float).reshape(shape)
    x2d, z2d = np.meshgrid(radius, height, indexing="ij")
    normalised_levels = contour_levels(N_CONTOURS)
    axis_flux = float(np.asarray(equilibrium.topology.axis_flux))
    flux_span = float(np.asarray(equilibrium.topology.flux_span))
    physical_levels = axis_flux + flux_span * normalised_levels

    def legacy_step():
        return [
            line
            for level in normalised_levels
            for line in _reference_level_lines(psi_norm, radius, height, float(level))
        ]

    def nova_step():
        return poloidal_view._closed_contour_lines(x2d, z2d, psi2d, physical_levels)

    surface_rows = nova_step()

    def precomputed_step():
        return [poloidal_view.close_outline(line) for line in surface_rows]

    timings = {}
    for name, operation in (
        ("legacy_marching_squares_seconds", legacy_step),
        ("nova_contour_seconds", nova_step),
        ("precomputed_surfaces_seconds", precomputed_step),
    ):
        operation()
        started = perf_counter()
        for _ in range(repeats):
            operation()
        timings[name] = (perf_counter() - started) / repeats
    timings["repeats"] = float(repeats)
    return timings


# ---------------------------------------------------------------------------
# the styled figure: no chrome, wall, coils, plasma, contours, markers
# ---------------------------------------------------------------------------


def test_figure_has_no_visible_axis_grid_frame_or_bounding_box():
    poloidal = _styled_figure()
    assert not any(poloidal.axis.visible)
    assert not any(poloidal.grid.visible)
    assert poloidal.outline_line_color is None
    # the styled view draws contours, not a filled flux image
    assert not any(
        isinstance(renderer.glyph, Image)
        for renderer in poloidal.renderers
        if isinstance(renderer, GlyphRenderer)
    )


def test_wall_source_carries_one_dark_grey_polyline(channels):
    from bokeh.models import MultiLine

    wall = channels["wall"]
    assert len(wall["x"]) == len(wall["z"])
    ring = np.c_[wall["x"], wall["z"]]
    # one polyline, closed on itself
    assert ring.shape[0] == 62
    np.testing.assert_allclose(ring[0], ring[-1])

    glyph = _glyph(_styled_figure(), "MultiLine")
    assert isinstance(glyph, MultiLine)
    assert _css(glyph.line_color) == WALL_COLOR
    assert glyph.line_width == WALL_LINE_WIDTH


def test_coil_source_outlines_only_with_no_fill_columns(channels):
    from bokeh.models import MultiPolygons

    assert set(channels["coil"]) == {"x", "z"}  # no fill columns
    assert len(channels["coil"]["x"]) > 0  # the machine carries coils
    glyph = _glyph(_styled_figure(), "MultiPolygons", position=0)
    assert isinstance(glyph, MultiPolygons)
    # outlines only: the glyph binds no fill field and fills with alpha zero
    assert "fill_color" not in _bound_fields(glyph)
    assert "fill_alpha" not in _bound_fields(glyph)
    assert glyph.fill_alpha == 0


def test_decorative_coil_ring_helper_is_removed():
    """The view no longer places decorative rings on a circle around the wall."""
    import apps.pulsedesign.poloidal_view as poloidal_view

    assert not hasattr(poloidal_view, "external_coil_rings")


def test_coil_channel_carries_the_sixteen_fitted_conductor_outlines(
    solved_frame, channels
):
    """The coil channel is the fixture machine's conductor geometry.

    The Solov'ev machine exposes one outline per fitted conductor, each a
    small ring centred on its conductor to the stated tolerance, and the same
    outlines ride on the fixture machine beside its wall.
    """
    from apps.playable.solovev import conductor_centres, coil_outlines

    machine, _equilibrium = solved_frame
    centres = conductor_centres()
    expected = coil_outlines()
    assert centres.shape == (16, 2)
    rows = channels["coil"]
    assert len(rows["x"]) == len(rows["z"]) == len(centres) == 16

    for row, (x_row, z_row) in enumerate(zip(rows["x"], rows["z"], strict=True)):
        assert len(x_row) == 1 and len(z_row) == 1
        ring = np.c_[x_row[0], z_row[0]]
        # the drawn outline is exactly one fitted conductor's ring
        np.testing.assert_allclose(ring, expected[row], atol=1.0e-9)
        # ... and the ring is centred on the fitted conductor
        np.testing.assert_allclose(ring.mean(axis=0), centres[row], atol=1.0e-9)

    # the fixture machine carries the same outlines beside its wall
    assert len(machine.coils) == 16
    for outline, expected_ring in zip(machine.coils, expected, strict=True):
        np.testing.assert_allclose(outline, expected_ring, atol=1.0e-9)


def test_no_topology_marker_lies_outside_the_first_wall(channels):
    """O-point, X-point and strike markers are drawn only inside the wall."""
    from nova.equilibrium.wall_mask import inside_polygon

    wall = np.c_[channels["wall"]["x"], channels["wall"]["z"]]
    for name in ("o_points", "x_points", "x_points_secondary"):
        xs = np.asarray(channels[name]["x"], dtype=float)
        zs = np.asarray(channels[name]["z"], dtype=float)
        assert xs.size == zs.size
        for x, z in zip(xs, zs, strict=True):
            inside = inside_polygon(
                np.asarray([x]), np.asarray([z]), wall[:, 0], wall[:, 1]
            )
            assert inside[0], (
                f"{name} marker ({x:.4f}, {z:.4f}) lies outside the first wall"
            )


def test_plasma_source_is_light_purple_at_one_alpha_per_plasma_cell(
    solved_frame, channels
):
    from bokeh.models import MultiPolygons

    machine, equilibrium = solved_frame
    core_count = int(np.count_nonzero(np.asarray(equilibrium.domains.core)))
    polys = channels["plasma"]
    assert len(polys["x"]) == core_count, "one polygon per plasma cell"
    # every polygon is a single outer ring with no holes
    for x_row, z_row in zip(polys["x"], polys["z"], strict=True):
        assert len(x_row) == 1 and len(z_row) == 1
        assert len(x_row[0]) == len(z_row[0]) >= 3

    glyph = _glyph(_styled_figure(), "MultiPolygons", position=1)
    assert isinstance(glyph, MultiPolygons)
    assert _css(glyph.fill_color) == PLASMA_FILL_COLOR
    # one constant alpha, not a per-cell column
    assert isinstance(glyph.fill_alpha, float)
    assert glyph.fill_alpha == PLASMA_FILL_ALPHA
    assert "fill_alpha" not in _bound_fields(glyph)


def test_cut_cells_are_the_solve_clipped_polygons_with_vertices_within_pitch(
    solved_frame, channels
):
    """The separatrix-cut cells carry their clipped polygon from the clipped
    supports, vertices land on the solved LCFS within one lattice pitch, and no
    cell polygon penetrates the separatrix by more than that pitch."""
    from nova.equilibrium.stencil_mesh import StencilMesh
    from nova.geometry.hexstencil import hex_stencil

    machine, equilibrium = solved_frame
    profile = machine.profile
    operator = profile.operator
    lattice = profile.lattice
    geometry = operator.moment_geometry
    pitch = max(lattice.radial_step, lattice.vertical_step)

    # reproduce the assembly's cut: the operator's own shared-node flux
    # reconstruction and AtomicCellMesh.clip, the same cut the solve makes
    node_flux = np.asarray(operator.shared_node_flux(equilibrium.flux))
    axis_flux = float(np.asarray(equilibrium.topology.axis_flux))
    flux_span = float(np.asarray(equilibrium.topology.flux_span))
    boundary_flux = axis_flux + flux_span
    signed = (boundary_flux - node_flux) / flux_span
    supports = geometry.atomic_mesh.clip(signed)

    core = np.asarray(equilibrium.domains.core).astype(bool)
    cells = np.flatnonzero(supports.included & core)
    cut_cells = np.flatnonzero(supports.boundary & core)
    assert len(cut_cells) > 0, "the separatrix cuts at least one plasma cell"

    # the rendered rows map one-to-one to the included core cells in order
    assert len(channels["plasma"]["x"]) == len(cells)
    for row, cell in enumerate(cells):
        count = int(supports.vertex_count[cell])
        clipped = supports.support_vertices[cell, :count]
        rendered = np.c_[
            channels["plasma"]["x"][row][0], channels["plasma"]["z"][row][0]
        ]
        np.testing.assert_allclose(rendered, clipped, atol=1.0e-12)

    # a cut cell's rendered polygon is its clipped polygon, not the full cell
    cut_row = {int(cell): row for row, cell in enumerate(cells)}
    changed = 0
    for cell in cut_cells:
        full = np.asarray(geometry.polygons[int(cell)])
        rendered = np.c_[
            channels["plasma"]["x"][cut_row[int(cell)]][0],
            channels["plasma"]["z"][cut_row[int(cell)]][0],
        ]
        if len(full) != len(rendered) or not np.allclose(full, rendered, atol=1.0e-12):
            changed += 1
    assert changed == len(cut_cells), "every cut cell renders its clipped polygon"

    # the solved LCFS polyline is the separatrix the cut cells meet
    labelled = equilibrium.labelled_flux
    lcfs_count = int(np.asarray(labelled.lcfs_vertex_count))
    lcfs = np.asarray(labelled.lcfs)[:lcfs_count]

    def segment_distance(point, polyline):
        point = np.asarray(point, dtype=float)
        distances = []
        for start, end in zip(polyline[:-1], polyline[1:]):
            delta = end - start
            length_squared = delta @ delta
            if length_squared == 0.0:
                continue  # a repeated vertex carries no segment
            fraction = np.clip((point - start) @ delta / length_squared, 0.0, 1.0)
            distances.append(np.linalg.norm(point - (start + fraction * delta)))
        return min(distances) if distances else np.inf

    # vertices the clip created lie on the separatrix, within one lattice pitch
    grid_flux = np.asarray(equilibrium.flux)[: lattice.node_count]
    mesh = StencilMesh(
        lattice.coordinate, hex_stencil(lattice.shape), lattice.cell_area
    )

    def reconstructed_signed(points):
        query = mesh.shared_node_flux_stencil(np.asarray(points, dtype=float))
        return (boundary_flux - np.asarray(query(grid_flux))) / flux_span

    atomic_mesh = geometry.atomic_mesh
    atomic_nodes = atomic_mesh.node_coordinates

    def is_crossing(vertex) -> bool:
        distance = np.linalg.norm(atomic_nodes - np.asarray(vertex), axis=1).min()
        scale = atomic_mesh.tolerance * 10.0
        return distance > scale

    max_clipped_distance = 0.0
    for cell in cut_cells:
        count = int(supports.vertex_count[cell])
        polygon = (
            channels["plasma"]["x"][cut_row[int(cell)]][0],
            channels["plasma"]["z"][cut_row[int(cell)]][0],
        )
        for r, z in zip(polygon[0], polygon[1], strict=True):
            vertex = (r, z)
            if not is_crossing(vertex):
                continue
            max_clipped_distance = max(
                max_clipped_distance, segment_distance(vertex, lcfs)
            )
    assert max_clipped_distance <= pitch, (
        "each clipped vertex lies within the lattice pitch of the separatrix; "
        f"measured {max_clipped_distance:.3e} m against pitch {pitch:.4f} m"
    )

    # no rendered polygon crosses the separatrix by more than the lattice pitch
    vertices = []
    for row in range(len(channels["plasma"]["x"])):
        vertices.extend(
            (r, z)
            for r, z in zip(
                channels["plasma"]["x"][row][0], channels["plasma"]["z"][row][0]
            )
        )
    signed_at = reconstructed_signed(np.asarray(vertices, dtype=float))
    # outward penetration, scaled to metres by the normalised flux gradient
    gradient = 1.0 / pitch  # psi_norm changes by about one over one pitch
    penetration = np.maximum(0.0, -signed_at) / gradient
    assert penetration.max() <= pitch, (
        "no cell polygon crosses the separatrix by more than the lattice pitch; "
        f"measured {penetration.max():.3e} m against pitch {pitch:.4f} m"
    )


def test_raster_contours_instantiate_nova_contourer_and_local_tracer_is_absent(
    solved_frame, monkeypatch
):
    """The raster fallback delegates every level cut to Nova's Contour."""
    import apps.pulsedesign.poloidal_view as poloidal_view

    actual_contour = poloidal_view.Contour
    calls = []

    def recording_contour(*args, **kwargs):
        calls.append((args, kwargs))
        return actual_contour(*args, **kwargs)

    monkeypatch.setattr(poloidal_view, "Contour", recording_contour)
    machine, equilibrium = solved_frame
    result = poloidal_view.poloidal_channels(
        equilibrium,
        machine.profile,
        wall=machine.wall,
        coils=machine.coils,
        n_contours=N_CONTOURS,
    )

    assert len(calls) == 1
    x2d, z2d, psi2d = calls[0][0]
    levels = calls[0][1]["levels"]
    raster = equilibrium.raster_flux
    shape = (len(raster.radius), len(raster.height))
    assert x2d.shape == z2d.shape == psi2d.shape == shape
    axis_flux = float(np.asarray(equilibrium.topology.axis_flux))
    flux_span = float(np.asarray(equilibrium.topology.flux_span))
    np.testing.assert_allclose(
        levels, axis_flux + flux_span * contour_levels(N_CONTOURS)
    )
    for name in ("_trace_level", "_chain", "_key"):
        assert not hasattr(poloidal_view, name)
    for x_line, z_line in zip(
        result["levelset"]["x"][:-1], result["levelset"]["z"][:-1], strict=True
    ):
        np.testing.assert_allclose((x_line[0], z_line[0]), (x_line[-1], z_line[-1]))


def test_nova_contours_agree_with_replaced_raster_tracer_within_one_cell(
    solved_frame, channels
):
    """The contourer change does not move a loop by one raster cell."""
    machine, equilibrium = solved_frame
    del machine
    raster = equilibrium.raster_flux
    radius = np.asarray(raster.radius, dtype=float)
    height = np.asarray(raster.height, dtype=float)
    psi_norm = np.asarray(raster.psi_norm, dtype=float).reshape(
        len(radius), len(height)
    )
    pitch = max(float(np.diff(radius).max()), float(np.diff(height).max()))

    reference = [
        line
        for level in contour_levels(N_CONTOURS)
        for line in _reference_level_lines(psi_norm, radius, height, float(level))
        if len(line) > 2 and np.allclose(line[0], line[-1], atol=1.0e-9)
    ]
    actual = [
        np.c_[x_line, z_line]
        for x_line, z_line in zip(
            channels["levelset"]["x"][:-1],
            channels["levelset"]["z"][:-1],
            strict=True,
        )
    ]
    assert reference and actual
    for line in reference:
        separation = min(_polyline_separation(line, candidate) for candidate in actual)
        assert separation <= pitch
    for line in actual:
        assert (
            min(_polyline_separation(line, candidate) for candidate in reference)
            <= pitch
        )


def test_precomputed_surfaces_are_closed_and_drawn_without_contouring(
    solved_frame, monkeypatch
):
    """Given surface coordinates pass to the channel verbatim, plus closure."""
    import apps.pulsedesign.poloidal_view as poloidal_view

    class UnexpectedContour:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("precomputed surfaces must bypass contouring")

    monkeypatch.setattr(poloidal_view, "Contour", UnexpectedContour)
    surfaces = np.asarray(
        [
            [[0.4, -0.1], [0.5, -0.2], [0.6, -0.1], [0.5, 0.0]],
            [[0.3, -0.2], [0.5, -0.3], [0.7, -0.2], [0.5, 0.1]],
            [[0.2, -0.3], [0.5, -0.4], [0.8, -0.3], [0.5, 0.2]],
        ]
    )
    machine, equilibrium = solved_frame
    result = poloidal_view.poloidal_channels(
        equilibrium,
        machine.profile,
        wall=machine.wall,
        coils=machine.coils,
        surfaces=surfaces,
        stride=2,
    )
    lines = [
        np.c_[x_line, z_line]
        for x_line, z_line in zip(
            result["levelset"]["x"], result["levelset"]["z"], strict=True
        )
    ]
    assert len(lines) == 2
    for line, supplied in zip(lines, surfaces[::2], strict=True):
        np.testing.assert_array_equal(line[:-1], supplied)
        np.testing.assert_array_equal(line[-1], supplied[0])


def test_contour_source_is_dark_grey_unfilled_levels_with_separatrix_outermost(
    solved_frame, channels
):
    from bokeh.models import MultiLine

    machine, equilibrium = solved_frame
    lines = channels["levelset"]
    assert len(lines["x"]) == len(lines["z"])
    assert len(lines["x"]) >= N_CONTOURS

    # the outermost line is the solved separatrix
    separatrix_count = int(np.asarray(equilibrium.raster_flux.separatrix_vertex_count))
    separatrix = np.asarray(equilibrium.raster_flux.separatrix)[:separatrix_count]
    np.testing.assert_allclose(lines["x"][-1], separatrix[:, 0].tolist(), atol=1.0e-12)
    np.testing.assert_allclose(lines["z"][-1], separatrix[:, 1].tolist(), atol=1.0e-12)

    # the interior levels sit between the axis and the boundary at the stated
    # fixed count; sample each traced polyline's midpoint to confirm the level
    levels = contour_levels(N_CONTOURS)
    assert len(levels) == N_CONTOURS - 1
    radius = np.asarray(equilibrium.raster_flux.radius, dtype=float)
    height = np.asarray(equilibrium.raster_flux.height, dtype=float)
    psi_norm = np.asarray(equilibrium.raster_flux.psi_norm, dtype=float)
    shape = (len(radius), len(height))
    psi2d = psi_norm.reshape(shape)

    def interpolated(point):
        row = min(max(np.searchsorted(radius, point[0]) - 1, 0), len(radius) - 2)
        column = min(max(np.searchsorted(height, point[1]) - 1, 0), len(height) - 2)
        fa = np.clip((point[0] - radius[row]) / (radius[row + 1] - radius[row]), 0, 1)
        span = height[column + 1] - height[column]
        fb = np.clip((point[1] - height[column]) / span, 0, 1)
        return (
            psi2d[row, column] * (1 - fa) * (1 - fb)
            + psi2d[row + 1, column] * fa * (1 - fb)
            + psi2d[row, column + 1] * (1 - fa) * fb
            + psi2d[row + 1, column + 1] * fa * fb
        )

    traced_levels = set()
    for x_line, z_line in zip(lines["x"][:-1], lines["z"][:-1], strict=True):
        midpoint = (x_line[len(x_line) // 2], z_line[len(z_line) // 2])
        value = interpolated(midpoint)
        inside = 0.0 < value < 1.0
        assert inside, f"an interior contour leaves (axis, boundary): {value}"
        match = float(levels[np.argmin(np.abs(levels - value))])
        assert abs(value - match) < 0.05, f"contour level {value} not on the grid"
        traced_levels.add(match)
    assert len(traced_levels) == N_CONTOURS - 1

    glyph = _glyph(_styled_figure(), "MultiLine", position=1)
    assert isinstance(glyph, MultiLine)
    assert _css(glyph.line_color) == CONTOUR_COLOR


def test_o_point_and_primary_x_point_carry_distinct_markers_with_secondary_lighter(
    channels,
):
    # the axis marker is drawn on the limited fixture; the off-vessel X-point
    # and strike markers are filtered out, so each channel's rows stay paired
    assert len(channels["o_points"]["x"]) == 1
    for name in ("o_points", "x_points", "x_points_secondary"):
        assert len(channels[name]["x"]) == len(channels[name]["z"])

    sources = channel_sources()
    for name in ("o_points", "x_points", "x_points_secondary"):
        sources[name].data = channels[name]
    poloidal = poloidal_figure(sources)
    scatter = {
        renderer.data_source: renderer.glyph
        for renderer in poloidal.renderers
        if isinstance(renderer, GlyphRenderer)
        and type(renderer.glyph).__name__ == "Scatter"
    }
    o_point = scatter[sources["o_points"]]
    x_point = scatter[sources["x_points"]]
    secondary = scatter[sources["x_points_secondary"]]
    # distinct marker kinds for the axis and the primary X-point
    assert o_point.marker == "circle" and x_point.marker == "x"
    assert _css(o_point.fill_color) == O_POINT_COLOR
    assert _css(x_point.line_color) == X_POINT_COLOR
    # the secondary X-point and strike points are lighter when present
    primary_luminance = _hex_luminance(_css(x_point.line_color))
    assert primary_luminance < _hex_luminance(X_POINT_SECONDARY_COLOR)
    assert _css(secondary.line_color) == X_POINT_SECONDARY_COLOR


def test_every_channel_source_initialises_its_bound_columns():
    """A served document logs no BAD_COLUMN_NAME from a missing bound column."""
    sources = channel_sources()
    poloidal = poloidal_figure(sources)
    compensation = compensation_figure(sources)
    receipt = keyframe_receipt(sources)

    for renderer in [*poloidal.renderers, *compensation.renderers]:
        if not isinstance(renderer, GlyphRenderer):
            continue
        owner = renderer.data_source
        bound = _bound_fields(renderer.glyph)
        assert bound <= set(owner.data), (
            f"{type(renderer.glyph).__name__} binds {bound}"
        )
    table_fields = {column.field for column in receipt.columns}
    assert table_fields <= set(receipt.source.data)


def test_pulse_design_still_imports_the_styled_renderers():
    import apps.pulsedesign.poloidal_view as poloidal_view

    assert poloidal_view.poloidal_figure is not None
    import apps.pulsedesign as pds

    assert pds.Simulator is not None


def test_a_styled_frame_png_is_committed_under_the_view_figure():
    candidates = list(VIEW_FIGURE.glob("*.png"))
    assert candidates, f"no styled PNG committed under {VIEW_FIGURE}"
    # a rendered, non-trivial frame for the record
    assert any(candidate.stat().st_size > 10_000 for candidate in candidates)
