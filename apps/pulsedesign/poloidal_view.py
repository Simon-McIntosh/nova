"""Shared poloidal renderers for the pulse-design and playable solve apps.

Both Bokeh applications draw the machine and its plasma from one set of
:class:`~bokeh.models.ColumnDataSource` channels, so the glyph-to-column
bindings live in one place and a session that pushes a frame never guesses a
shape the renderer does not bind.  The pushed column shapes follow from the
bindings and are the contract the playable session's gate asserts:

- ``levelset`` — multi_line flux surfaces — ``x``, ``z`` — (n,), (n,)
- ``wall`` — multi_line machine wall — ``x``, ``z`` — (n,), (n,)
- ``x_points`` — scatter turning points — ``x``, ``z`` — (n,), (n,)
- ``plasma`` — multi_polygons cells — ``x``, ``z``, ``ionize`` — (n,), (n,), (n,)
- ``points`` — scatter control points — ``x``, ``z`` — (n,), (n,)
- ``flux`` — image raster flux — ``psi`` (2-D per row) — (n_height, n_radius)
- ``separatrix`` — multi_line LCFS — ``x``, ``z`` — (n,), (n,) NaN-padded
- ``compensation`` — vbar currents/circuit — ``circuit``, ``current`` — (n,), (n,)
- ``receipt`` — DataTable receipt — ``action``, ``wall``, ``trips`` — (1,), (1,), (1,)

``poloidal_figure`` binds the channels the two applications share; the
playable solve binds the keyframe channels on top of it.
"""

from bokeh.models import (
    ColumnDataSource,
    DataTable,
    GlyphRenderer,
    LinearColorMapper,
    TableColumn,
)
from bokeh.plotting import figure


def poloidal_figure(
    source: dict[str, ColumnDataSource], *, height: int = 650, name: str = "poloidal"
) -> figure:
    """Return the shared poloidal figure bound to the common channels."""
    poloidal = figure(name=name, match_aspect=True, height=height)
    poloidal.axis.visible = False
    poloidal.multi_line(
        "x", "z", source=source["levelset"], color="gray", alpha=0.5, level="overlay"
    )
    poloidal.multi_line(
        "x", "z", source=source["wall"], color="black", width=2, level="underlay"
    )
    poloidal.scatter(
        "x", "z", source=source["x_points"], marker="x", size=8, line_color="red"
    )
    poloidal.multi_polygons(
        "x", "z", fill_alpha="ionize", line_alpha=0, source=source["plasma"]
    )
    poloidal.scatter("x", "z", source=source["points"], marker="circle_cross", size=8)
    return poloidal


def add_separatrix(
    poloidal: figure, source: dict[str, ColumnDataSource]
) -> GlyphRenderer:
    """Bind the LCFS polyline channel on top of the shared poloidal figure."""
    return poloidal.multi_line(
        "x", "z", source=source["separatrix"], color="dodgerblue", width=3
    )


def add_flux_image(
    poloidal: figure,
    source: dict[str, ColumnDataSource],
    radius: tuple[float, float],
    height: tuple[float, float],
) -> GlyphRenderer:
    """Bind the raster flux image on the machine's fixed rectangular grid.

    ``radius`` and ``height`` are the physical (minimum, maximum) bounds of
    the raster the image is drawn on; only ``psi`` is pushed per keyframe, so
    the image anchor stays part of the machine, not the frame.
    """
    mapper = LinearColorMapper(palette="Viridis256", low=0.0, high=1.0)
    return poloidal.image(
        image="psi",
        x=radius[0],
        y=height[0],
        dw=radius[1] - radius[0],
        dh=height[1] - height[0],
        source=source["flux"],
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
    chart = figure(
        name=name,
        height=height,
        x_axis_label="circuit",
        y_axis_label="compensating current / A",
    )
    chart.vbar(x="circuit", top="current", source=source["compensation"], width=0.7)
    return chart


def keyframe_receipt(
    source: dict[str, ColumnDataSource], *, height: int = 150, name: str = "receipt"
) -> DataTable:
    """Return the keyframe receipt table bound to the wall-and-trips row."""
    columns = [
        TableColumn(field="action", title="action"),
        TableColumn(field="wall", title="wall / s"),
        TableColumn(field="trips", title="trips"),
    ]
    return DataTable(
        source=source["receipt"],
        columns=columns,
        height=height,
        name=name,
        autosize_mode="force_fit",
    )
