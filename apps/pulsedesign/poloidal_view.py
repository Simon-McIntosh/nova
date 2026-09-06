"""Shared poloidal renderers for the pulse-design and playable solve apps.

Both Bokeh applications draw the machine and its plasma from one set of
:class:`~bokeh.models.ColumnDataSource` channels, so the glyph-to-column
bindings live in one place and a session that pushes a frame never guesses a
shape the renderer does not bind.  The pushed column shapes follow from the
bindings and are the contract the playable session's gate asserts:

- ``levelset`` — multi_line flux surfaces — ``x``, ``z`` — (n,), (n,)
- ``wall`` — multi_line machine wall — ``x``, ``z`` — (n,), (n,)
- ``coil`` — multi_polygons coil outlines — ``x``, ``z`` — (n,), (n,)
- ``plasma`` — multi_polygons plasma cells — ``x``, ``z`` — (n,), (n,)
- ``o_points`` — scatter magnetic-axis markers — ``x``, ``z`` — (n,), (n,)
- ``x_points`` — scatter primary X-point markers — ``x``, ``z`` — (n,), (n,)
- ``x_points_secondary`` — scatter secondary X-point and strike points — ``x``, ``z``
- ``points`` — scatter control points — ``x``, ``z`` — (n,), (n,)
- ``separatrix`` — multi_line LCFS — ``x``, ``z`` — (n,), (n,) NaN-padded
- ``compensation`` — vbar currents/circuit — ``circuit``, ``current`` — (n,), (n,)
- ``receipt`` — DataTable receipt — ``action``, ``wall``, ``trips`` — (1,), (1,), (1,)

``poloidal_figure`` binds the channels the two applications share and draws
them in the locked style: no axis, grid, frame or bounding box; the first
wall as one thin dark polyline; each coil as a simple outline with no fill;
the plasma as light purple cells at one alpha with the separatrix-cut cells
drawn as their solved clipped polygons; dark grey unfilled psi contours with
the separatrix as the outermost line; and distinct markers on the magnetic
axis and primary X-point with the secondary X-point and strike points drawn
lighter, every topology marker drawn only inside the closed first-wall
polygon.  ``poloidal_channels`` assembles the styled channels from one solved
:class:`~nova.equilibrium.forward.ForwardEquilibrium` and its forward
profile: the plasma cells come from the operator's own moment geometry and
the cut cells from :meth:`~nova.equilibrium.separatrix_clip.AtomicCellMesh.clip`
evaluated on the operator's shared-node flux reconstruction, so the plasma
edge is the same cut the solve integrates over (a hexagonal production mesh
renders as hex cells under this same path, because the cells come from the
mesh, not from an assumed shape).

Every channel source is initialised with its bound columns through
:func:`channel_sources`, so a served document logs no BAD_COLUMN_NAME warning
before its first keyframe.
"""

from bokeh.models import (
    ColumnDataSource,
    DataTable,
    GlyphRenderer,
    LinearColorMapper,
    TableColumn,
)
from bokeh.plotting import figure
import numpy as np

from nova.equilibrium.wall_mask import inside_polygon

# ---------------------------------------------------------------------------
# locked style: weights follow pulse-design's plasma rendering and imas-ink's
# InkStyle (coil edge grey with no face, wall a thin dark line, axis marker a
# dot, X-point marker an x) without reproducing either exactly
# ---------------------------------------------------------------------------

#: First wall polyline.
WALL_COLOR = "#3c3c3c"
WALL_LINE_WIDTH = 1.5

#: Coil outlines (edge only, no face).
COIL_COLOR = "#777777"
COIL_LINE_WIDTH = 0.8

#: Unfilled flux contours, the separatrix outermost.
CONTOUR_COLOR = "#3f3f3f"
CONTOUR_LINE_WIDTH = 0.9

#: Plasma cell faces at one constant alpha.
PLASMA_FILL_COLOR = "#d7c3f0"
PLASMA_FILL_ALPHA = 0.85
PLASMA_LINE_COLOR = "#a98fd0"
PLASMA_LINE_ALPHA = 0.45
PLASMA_LINE_WIDTH = 0.5

#: Topology markers: the magnetic axis is a dot, the primary X-point an x.
O_POINT_COLOR = "#222222"
O_POINT_SIZE = 7
X_POINT_COLOR = "#111111"
X_POINT_SIZE = 12
X_POINT_SECONDARY_COLOR = "#8c8c8c"
X_POINT_SECONDARY_SIZE = 8

#: Commanded control points, kept subtle in the shared figure.
POINTS_COLOR = "#7a7a7a"
POINTS_SIZE = 5

#: Number of flux-contour lines including the separatrix (outermost).
N_CONTOURS = 7


def _fallback_source(columns):
    """Return an empty source pre-bound with ``columns`` for absent channels."""
    return ColumnDataSource(data={column: [] for column in columns})


def channel_sources() -> dict[str, ColumnDataSource]:
    """Return one ColumnDataSource per channel, its bound columns initialised.

    Rendering a glyph whose field is missing from its source logs a
    BAD_COLUMN_NAME warning; pre-binding the columns keeps a served document
    clean from first load.
    """
    sources = {
        "levelset": _fallback_source(("x", "z")),
        "wall": _fallback_source(("x", "z")),
        "coil": _fallback_source(("x", "z")),
        "plasma": _fallback_source(("x", "z")),
        "o_points": _fallback_source(("x", "z")),
        "x_points": _fallback_source(("x", "z")),
        "x_points_secondary": _fallback_source(("x", "z")),
        "points": _fallback_source(("x", "z")),
        "separatrix": _fallback_source(("x", "z")),
        "compensation": _fallback_source(("circuit", "current")),
        "receipt": _fallback_source(("action", "wall", "trips")),
    }
    return sources


def poloidal_figure(
    source: dict[str, ColumnDataSource], *, height: int = 650, name: str = "poloidal"
) -> figure:
    """Return the shared poloidal figure in the locked view style.

    The figure binds the two applications' common channels.  A channel absent
    from ``source`` (pulse-design does not carry the secondary markers) is
    bound to a local empty source, so one renderer set serves both apps and
    the session gate's minimal channel dict alike.
    """
    levelset = source.get("levelset") or _fallback_source(("x", "z"))
    wall = source.get("wall") or _fallback_source(("x", "z"))
    coil = source.get("coil") or _fallback_source(("x", "z"))
    plasma = source.get("plasma") or _fallback_source(("x", "z"))
    o_points = source.get("o_points") or _fallback_source(("x", "z"))
    x_points = source.get("x_points") or _fallback_source(("x", "z"))
    x_points_secondary = source.get("x_points_secondary") or _fallback_source(
        ("x", "z")
    )
    points = source.get("points") or _fallback_source(("x", "z"))

    poloidal = figure(name=name, match_aspect=True, height=height)
    poloidal.axis.visible = False
    poloidal.grid.visible = False
    poloidal.outline_line_color = None

    poloidal.multi_line(
        "x",
        "z",
        source=wall,
        color=WALL_COLOR,
        width=WALL_LINE_WIDTH,
        level="underlay",
    )
    poloidal.multi_polygons(
        "x",
        "z",
        source=coil,
        line_color=COIL_COLOR,
        line_width=COIL_LINE_WIDTH,
        fill_alpha=0,
        level="underlay",
    )
    poloidal.multi_polygons(
        "x",
        "z",
        source=plasma,
        fill_color=PLASMA_FILL_COLOR,
        fill_alpha=PLASMA_FILL_ALPHA,
        line_color=PLASMA_LINE_COLOR,
        line_alpha=PLASMA_LINE_ALPHA,
        line_width=PLASMA_LINE_WIDTH,
    )
    poloidal.multi_line(
        "x", "z", source=levelset, color=CONTOUR_COLOR, width=CONTOUR_LINE_WIDTH
    )
    poloidal.scatter(
        "x",
        "z",
        source=o_points,
        marker="circle",
        size=O_POINT_SIZE,
        fill_color=O_POINT_COLOR,
        line_color=O_POINT_COLOR,
    )
    poloidal.scatter(
        "x",
        "z",
        source=x_points,
        marker="x",
        size=X_POINT_SIZE,
        line_color=X_POINT_COLOR,
        line_width=1.6,
    )
    poloidal.scatter(
        "x",
        "z",
        source=x_points_secondary,
        marker="x",
        size=X_POINT_SECONDARY_SIZE,
        line_color=X_POINT_SECONDARY_COLOR,
    )
    poloidal.scatter(
        "x",
        "z",
        source=points,
        marker="circle",
        size=POINTS_SIZE,
        line_color=POINTS_COLOR,
        fill_alpha=0,
    )
    return poloidal


def add_separatrix(
    poloidal: figure, source: dict[str, ColumnDataSource]
) -> GlyphRenderer:
    """Bind the LCFS polyline channel as a thin dark line."""
    separator = source.get("separatrix") or _fallback_source(("x", "z"))
    return poloidal.multi_line(
        "x", "z", source=separator, color=CONTOUR_COLOR, width=CONTOUR_LINE_WIDTH
    )


def add_flux_image(
    poloidal: figure,
    source: dict[str, ColumnDataSource],
    radius: tuple[float, float],
    height: tuple[float, float],
) -> GlyphRenderer:
    """Bind the raster flux image on the machine's fixed rectangular grid.

    The styled playable view draws contours instead of the filled image, so
    the playable app does not call this; the function is retained for the
    pulse-design path that still pushes a filled flux image.
    """
    flux = source.get("flux") or _fallback_source(("psi",))
    mapper = LinearColorMapper(palette="Viridis256", low=0.0, high=1.0)
    return poloidal.image(
        image="psi",
        x=radius[0],
        y=height[0],
        dw=radius[1] - radius[0],
        dh=height[1] - height[0],
        source=flux,
        color_mapper=mapper,
        global_alpha=0.6,
    )


def compensation_figure(
    source: dict[str, ColumnDataSource],
    *,
    height: int = 150,
    name: str = "compensation",
) -> figure:
    """Return the per-circuit compensating current chart."""
    compensation = source.get("compensation") or _fallback_source(
        ("circuit", "current")
    )
    chart = figure(
        name=name,
        height=height,
        x_axis_label="circuit",
        y_axis_label="compensating current / A",
    )
    chart.vbar(x="circuit", top="current", source=compensation, width=0.7)
    return chart


def keyframe_receipt(
    source: dict[str, ColumnDataSource], *, height: int = 150, name: str = "receipt"
) -> DataTable:
    """Return the keyframe receipt table bound to the wall-and-trips row."""
    receipt = source.get("receipt") or _fallback_source(("action", "wall", "trips"))
    columns = [
        TableColumn(field="action", title="action"),
        TableColumn(field="wall", title="wall / s"),
        TableColumn(field="trips", title="trips"),
    ]
    return DataTable(
        source=receipt,
        columns=columns,
        height=height,
        name=name,
        autosize_mode="force_fit",
    )


# ---------------------------------------------------------------------------
# contour tracing: marching squares on the structured raster
# ---------------------------------------------------------------------------


def _key(point) -> tuple[float, float]:
    """Return a rounded endpoint key for polyline chaining."""
    return round(float(point[0]), 10), round(float(point[1]), 10)


def _trace_level(psi2d, radius, height, level) -> list[np.ndarray]:
    """Return one level's iso-lines as a list of (n, 2) polylines.

    A linear-crossing marching-squares trace on the rectangular raster, with
    saddle cells paired through their cell-centre value so the returned
    polylines never cross.  Adjacent cells interpolate their shared edge with
    matching orientation, so every polyline chains into whole level lines.
    """
    psi2d = np.asarray(psi2d, dtype=float)
    radius = np.asarray(radius, dtype=float)
    height = np.asarray(height, dtype=float)
    n_radius, n_height = psi2d.shape
    dr = radius[1] - radius[0]
    dh = height[1] - height[0]

    segments = []
    for i in range(n_radius - 1):
        for j in range(n_height - 1):
            v = (psi2d[i, j], psi2d[i, j + 1], psi2d[i + 1, j + 1], psi2d[i + 1, j])
            pos = [value > level for value in v]
            inside = sum(pos)
            if inside in (0, 4):
                continue
            crossed: list = []
            for edge, (first, second) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
                if pos[first] == pos[second]:
                    continue
                va, vb = v[first], v[second]
                t = (level - va) / (vb - va)
                if edge == 0:  # r fixed at radius[i], z from height[j] to j+1
                    point = (radius[i], height[j] + t * dh)
                elif edge == 1:  # z fixed at height[j+1], r from i to i+1
                    point = (radius[i] + t * dr, height[j + 1])
                elif edge == 2:  # r fixed at radius[i+1], z from j+1 down to j
                    point = (radius[i + 1], height[j + 1] - t * dh)
                else:  # z fixed at height[j], r from i+1 down to i
                    point = (radius[i + 1] - t * dr, height[j])
                crossed.append(point)

            if len(crossed) == 2:
                segments.append((crossed[0], crossed[1]))
            elif len(crossed) == 4:
                centre = 0.25 * (v[0] + v[1] + v[2] + v[3])
                # pair through the cell centre so the low region stays connected
                if centre <= level:
                    segments.append((crossed[0], crossed[3]))
                    segments.append((crossed[1], crossed[2]))
                else:
                    segments.append((crossed[0], crossed[1]))
                    segments.append((crossed[2], crossed[3]))
            else:  # pragma: no cover - a square has two or four crossings
                raise AssertionError(f"unexpected crossing count {len(crossed)}")

    return _chain(segments)


def _chain(segments) -> list[np.ndarray]:
    """Chain oriented crossing segments into whole iso-lines."""
    neighbours = {}
    for index, (start, end) in enumerate(segments):
        neighbours.setdefault(_key(start), []).append(index)
        neighbours.setdefault(_key(end), []).append(index)
    used = [False] * len(segments)
    lines: list[np.ndarray] = []
    for start in range(len(segments)):
        if used[start]:
            continue
        used[start] = True
        chain = [list(segments[start][0]), list(segments[start][1])]
        extended = True
        while extended:
            extended = False
            for end_index in (0, -1):
                key = _key(chain[end_index])
                for other in neighbours.get(key, ()):
                    if used[other]:
                        continue
                    used[other] = True
                    other_start, other_end = segments[other]
                    if _key(other_start) == key:
                        following = other_end
                    else:
                        following = other_start
                    if end_index == 0:
                        chain.insert(0, list(following))
                    else:
                        chain.append(list(following))
                    extended = True
                    break
        lines.append(np.asarray(chain))
    return lines


def contour_levels(n_contours: int = N_CONTOURS) -> np.ndarray:
    """Return the interior flux-surface levels between the axis and boundary.

    ``n_contours`` is the total line count including the separatrix, so the
    interior levels are the ``n_contours - 1`` values strictly between zero
    (the axis) and one (the boundary).
    """
    return np.linspace(0.0, 1.0, n_contours + 1)[1:-1]


# ---------------------------------------------------------------------------
# frame channel assembly
# ---------------------------------------------------------------------------


def close_outline(outline: np.ndarray) -> np.ndarray:
    """Return a polyline closed by repeating its first point."""
    outline = np.asarray(outline, dtype=float)
    if outline.size and not np.allclose(outline[0], outline[-1]):
        return np.vstack([outline, outline[:1]])
    return outline


def poloidal_channels(
    equilibrium,
    profile,
    *,
    wall: np.ndarray,
    coils=(),
    n_contours: int = N_CONTOURS,
) -> dict[str, dict[str, list]]:
    """Return the styled poloidal channels for one solved frame.

    The plasma cells are the forward operator's own moment polygons (hexagonal
    on a production hexagonal mesh, rectangular on a structured lattice),
    clipped where the separatrix cuts them with the same
    :class:`~nova.equilibrium.separatrix_clip.AtomicCellMesh.clip` call and
    the same shared-node flux reconstruction the solve integrates over; only
    the solved core cells are drawn, so the clipped edge is the solved
    boundary rather than a staircase.  Contours are unfilled level lines on
    the raster normalised flux between the axis and the boundary, with the
    solved separatrix as the outermost line.  Markers are read from the
    frame's labelled points.
    """
    operator = profile.operator
    geometry = operator.moment_geometry
    if geometry is None:
        raise ValueError("the forward operator carries no moment geometry")

    atomic = geometry.atomic_mesh
    node_flux = np.asarray(operator.shared_node_flux(equilibrium.flux))
    boundary_flux = float(
        np.asarray(equilibrium.topology.axis_flux)
        + np.asarray(equilibrium.topology.flux_span)
    )
    flux_span = float(np.asarray(equilibrium.topology.flux_span))
    # Positive inside for either polarity: unit-normalised distance below the
    # separatrix level, the same cut the solve integrates over.
    signed = (boundary_flux - node_flux) / flux_span
    supports = atomic.clip(signed)

    core = np.asarray(equilibrium.domains.core).astype(bool)
    cells = np.flatnonzero(supports.included & core)
    plasma_xs: list[list[list[float]]] = []
    plasma_zs: list[list[list[float]]] = []
    for cell in cells:
        count = int(supports.vertex_count[cell])
        ring = supports.support_vertices[cell, :count]
        plasma_xs.append([ring[:, 0].tolist()])
        plasma_zs.append([ring[:, 1].tolist()])

    raster = equilibrium.raster_flux
    radius = np.asarray(raster.radius, dtype=float)
    height = np.asarray(raster.height, dtype=float)
    n_radius, n_height = radius.size, height.size
    psi_norm = np.asarray(raster.psi_norm, dtype=float).reshape(n_radius, n_height)
    contour_xs: list[list[float]] = []
    contour_zs: list[list[float]] = []
    for level in contour_levels(n_contours):
        for polyline in _trace_level(psi_norm, radius, height, float(level)):
            contour_xs.append(polyline[:, 0].tolist())
            contour_zs.append(polyline[:, 1].tolist())
    separatrix_count = int(np.asarray(raster.separatrix_vertex_count))
    separatrix = np.asarray(raster.separatrix, dtype=float)[:separatrix_count]
    contour_xs.append(separatrix[:, 0].tolist())
    contour_zs.append(separatrix[:, 1].tolist())

    labelled = equilibrium.labelled_flux
    o_point = np.asarray(labelled.o_point, dtype=float)
    primary = np.asarray(labelled.primary_x_point, dtype=float)
    secondary = np.asarray(labelled.secondary_x_point, dtype=float)
    strikes = np.reshape(labelled.strike_points, (-1, 2)).astype(float)
    finite_strikes = strikes[np.isfinite(strikes).all(axis=1)]

    wall = close_outline(wall)

    def inside_wall(point: np.ndarray) -> bool:
        """Return whether one finite point lies inside the closed first wall.

        Topology markers are drawn only inside the vessel so a limited fixture
        never shows an off-vessel cross.
        """
        if not np.isfinite(point).all():
            return False
        return bool(
            inside_polygon(
                np.asarray([float(point[0])]),
                np.asarray([float(point[1])]),
                wall[:, 0],
                wall[:, 1],
            )[0]
        )

    def finite_point(point: np.ndarray) -> tuple[list, list]:
        """Return (xs, zs) for one point, or empty lists when outside the wall."""
        if inside_wall(point):
            return [float(point[0])], [float(point[1])]
        return [], []

    o_xs, o_zs = finite_point(o_point)
    primary_xs, primary_zs = finite_point(primary)
    secondary_xs, secondary_zs = finite_point(secondary)
    in_wall_strikes = [row for row in finite_strikes if inside_wall(row)]
    secondary_xs += [float(row[0]) for row in in_wall_strikes]
    secondary_zs += [float(row[1]) for row in in_wall_strikes]

    coil_xs: list[list[list[float]]] = []
    coil_zs: list[list[list[float]]] = []
    for outline in coils:
        outline = np.asarray(outline, dtype=float)
        coil_xs.append([outline[:, 0].tolist()])
        coil_zs.append([outline[:, 1].tolist()])

    return {
        "wall": {"x": wall[:, 0].tolist(), "z": wall[:, 1].tolist()},
        "coil": {"x": coil_xs, "z": coil_zs},
        "plasma": {"x": plasma_xs, "z": plasma_zs},
        "levelset": {"x": contour_xs, "z": contour_zs},
        "o_points": {"x": o_xs, "z": o_zs},
        "x_points": {"x": primary_xs, "z": primary_zs},
        "x_points_secondary": {"x": secondary_xs, "z": secondary_zs},
    }
