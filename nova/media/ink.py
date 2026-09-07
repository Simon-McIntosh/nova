"""The imas-ink house style for Nova figures, in one place.

Every visual constant Nova's figures use lives on :class:`InkStyle`, so a
figure cannot carry a weight or a grey that disagrees with its neighbour: a
value that exists once cannot drift from itself. Upstream imas-ink remains the
authority for what the values ARE; this module is the single place Nova states
them, and the pinning test asserts the two still agree.

Field spellings follow the call sites that already read them, which differ from
upstream imas-ink in three places. The correspondence is fixed and worth
recording, because the upstream names describe what a contour *is* while these
describe where it is drawn:

===========================  ==============================
This module                   imas-ink ``InkStyle``
===========================  ==============================
``contour_color``             ``sol_color``
``contour_linewidth``         ``sol_linewidth``
``separatrix_color``          ``sep_color``
``separatrix_linewidth``      ``sep_linewidth``
===========================  ==============================

The plasma-cell fields have a different provenance: they are the purple the
lead locked for the interactive poloidal view. That view's palette is
deliberately *not* imas-ink -- its wall, coil and contour greys are darker so
they read on a screen rather than on paper -- so only the purple is shared
here, and the rest of that palette stays where it is. Sharing the whole thing
would silently restyle a served application.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib


@dataclass(frozen=True)
class InkStyle:
    """Visual constants for a poloidal scene and its one-dimensional traces."""

    # Confined flux surfaces -- closed contours enclosing the magnetic axis.
    flux_color: str = "#3366cc"
    flux_linewidth: float = 0.7

    # Open and scrape-off contours. Subordinate to the confined region, so
    # they are visible without competing with it.
    contour_color: str = "#999999"
    contour_linewidth: float = 0.35

    # The boundary flux surface.
    separatrix_color: str = "#cc0000"
    separatrix_linewidth: float = 1.5

    # First wall.
    wall_color: str = "#000000"
    wall_linewidth: float = 1.0

    # Active-coil sections: an outline and no fill, so a coil never hides a
    # contour crossing it.
    coil_edgecolor: str = "#888888"
    coil_facecolor: str = "none"
    coil_linewidth: float = 0.4

    # Plasma cells. A wall- or separatrix-clipped cell is drawn as its clipped
    # polygon, never as the unclipped hexagon.
    plasma_facecolor: str = "#d7c3f0"
    plasma_alpha: float = 0.85
    plasma_edgecolor: str = "#a98fd0"
    plasma_edge_alpha: float = 0.45
    plasma_linewidth: float = 0.5

    # Topology markers. The three shapes stay distinguishable in greyscale.
    axis_marker: str = "."
    axis_markersize: float = 6.0
    axis_color: str = "#cc0000"
    xpoint_marker: str = "x"
    xpoint_markersize: float = 6.0
    xpoint_markeredgewidth: float = 1.2
    xpoint_color: str = "#cc0000"
    strike_marker: str = "o"
    strike_markersize: float = 4.0
    strike_markeredgewidth: float = 0.8
    strike_color: str = "#cc0000"
    strike_markeredgecolor: str = "white"

    # Magnetic diagnostics.
    probe_color: str = "#888888"
    probe_markersize: float = 2.5
    flux_loop_color: str = "#666666"
    flux_loop_markersize: float = 3.0

    # Thomson scattering. Two colours so co-located strings of different
    # orientation or system stay separable; the second is imas-ink's own
    # secondary-orientation colour rather than a fresh invention.
    thomson_primary_color: str = "#3366cc"
    thomson_secondary_color: str = "#cc7722"
    thomson_markersize: float = 2.0
    thomson_chord_linewidth: float = 0.5
    thomson_chord_alpha: float = 0.55

    # One-dimensional traces.
    trace_linewidth: float = 1.2
    trace_markersize: float = 3.0
    trace_cursor_color: str = "#cc0000"
    trace_cursor_linewidth: float = 1.0

    # Text.
    label_fontsize: float = 8.0
    label_bbox: dict = field(
        default_factory=lambda: {
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "none",
            "pad": 2,
        }
    )

    # Figure.
    figure_facecolor: str = "white"
    figure_dpi: int = 120

    # Draw order. Contours sit under the machine so a wall or coil edge stays
    # readable where flux crowds it; markers and labels sit above everything.
    zorder_flux: int = 2
    zorder_plasma: int = 2
    zorder_coils: int = 3
    zorder_wall: int = 4
    zorder_separatrix: int = 5
    zorder_thomson: int = 6
    zorder_markers: int = 6
    zorder_label: int = 7

    def variant(self, **overrides) -> InkStyle:
        """Return this style with ``overrides`` applied.

        A figure that needs one different weight takes a variant rather than
        mutating the shared default, which is frozen for exactly that reason.
        """
        return replace(self, **overrides)


DEFAULT_INK = InkStyle()
"""The style every Nova figure uses unless it is handed another."""


def poloidal_axes(
    axes: matplotlib.axes.Axes, style: InkStyle = DEFAULT_INK
) -> matplotlib.axes.Axes:
    """Prepare ``axes`` for a poloidal scene: equal aspect, no chrome at all.

    A poloidal plot carries its own scale through the machine drawn in it, so
    spines and ticks add furniture without adding information, and a distorted
    aspect would misreport the shape that is the whole subject of the figure.
    """
    axes.set_aspect("equal")
    axes.set_axis_off()
    axes.grid(False)
    axes.set_facecolor(style.figure_facecolor)
    return axes


def trace_axes(
    axes: matplotlib.axes.Axes, style: InkStyle = DEFAULT_INK
) -> matplotlib.axes.Axes:
    """Prepare ``axes`` for a one-dimensional trace: despined, never gridded.

    Only the left and bottom spines survive. Gridlines are removed rather than
    left to the active rcParams, so a figure looks the same whatever style
    sheet a caller loaded before importing this module.
    """
    axes.grid(False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.tick_params(
        axis="both", which="both", labelsize=style.label_fontsize, direction="out"
    )
    for side in ("left", "bottom"):
        axes.spines[side].set_linewidth(0.8)
    axes.set_facecolor(style.figure_facecolor)
    return axes
