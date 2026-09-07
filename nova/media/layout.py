"""The three-view figure: one tall poloidal panel, two stacked traces.

The layout is built on an explicit :class:`~matplotlib.figure.Figure` and its
own gridspec rather than through :meth:`nova.graphics.plot.Axes.generate`,
for one reason that matters to an animation: ``generate`` places axes with
``pyplot.subplot`` on the current figure, so a frame loop would depend on
global state that another figure can steal. Here the caller holds the figure
it draws into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nova.media.ink import DEFAULT_INK, InkStyle, poloidal_axes, trace_axes

if TYPE_CHECKING:
    import matplotlib


@dataclass(frozen=True)
class ThreeView:
    """One poloidal panel on the left, two trace panels stacked on the right.

    The trace panels are named by position rather than by content, because the
    same layout carries flux-function profiles in one figure and Thomson
    channels in another.
    """

    figure: matplotlib.figure.Figure
    poloidal: matplotlib.axes.Axes
    upper: matplotlib.axes.Axes
    lower: matplotlib.axes.Axes
    extent: tuple[float, float, float, float]

    @property
    def traces(self) -> tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
        """Return both trace panels, upper first."""
        return (self.upper, self.lower)

    def clear(self) -> ThreeView:
        """Empty all three panels and restore their prepared state.

        A frame loop redraws into one figure instead of building a new one per
        frame, which keeps the compositor's memory flat over a long pulse.
        Clearing an axes discards its spines, ticks AND limits, so both the
        preparers and the poloidal extent are reapplied here -- otherwise the
        second frame would autoscale to its own contours and the machine would
        appear to breathe from frame to frame.
        """
        for axes in (self.poloidal, self.upper, self.lower):
            axes.clear()
        poloidal_axes(self.poloidal)
        trace_axes(self.upper)
        trace_axes(self.lower)
        self.poloidal.set_xlim(self.extent[0], self.extent[1])
        self.poloidal.set_ylim(self.extent[2], self.extent[3])
        self.poloidal.set_autoscale_on(False)
        return self


@dataclass(frozen=True)
class PoloidalView:
    """One poloidal panel on its own figure, sized from its machine."""

    figure: matplotlib.figure.Figure
    poloidal: matplotlib.axes.Axes
    extent: tuple[float, float, float, float]

    def clear(self) -> PoloidalView:
        """Empty the panel and restore its prepared state and extent."""
        self.poloidal.clear()
        poloidal_axes(self.poloidal)
        self.poloidal.set_xlim(self.extent[0], self.extent[1])
        self.poloidal.set_ylim(self.extent[2], self.extent[3])
        self.poloidal.set_autoscale_on(False)
        return self


def poloidal_view(
    extent: tuple[float, float, float, float],
    height: float = 6.5,
    style: InkStyle = DEFAULT_INK,
    margin: float = 0.04,
) -> PoloidalView:
    """Return a bare poloidal panel whose figure is the machine's own shape.

    For a figure that is only a machine view -- one that stacks above a camera
    pane, or stands alone on a slide. The figure is sized so the panel fills
    it: with no chrome to leave room for, a margin exists only to keep the
    outermost conductor off the edge, so it defaults to a small fraction of
    the panel rather than the inch-scale margins a trace column needs.

    Prefer this to building a Figure and calling
    :func:`nova.media.ink.poloidal_axes` yourself, so the aspect arithmetic
    stays in one place and two figures of the same machine match.
    """
    r_min, r_max, z_min, z_max = (float(value) for value in extent)
    if not (r_max > r_min and z_max > z_min):
        raise ValueError("the poloidal extent must span a positive area")
    import matplotlib.figure

    inset = max(0.0, min(0.45, float(margin)))
    panel_height = height * (1.0 - 2.0 * inset)
    width = panel_height * (r_max - r_min) / (z_max - z_min) / (1.0 - 2.0 * inset)
    figure = matplotlib.figure.Figure(
        figsize=(width, height), dpi=style.figure_dpi, facecolor=style.figure_facecolor
    )
    axes = figure.add_axes((inset, inset, 1.0 - 2.0 * inset, 1.0 - 2.0 * inset))
    return PoloidalView(
        figure=figure, poloidal=axes, extent=(r_min, r_max, z_min, z_max)
    ).clear()


def three_view(
    extent: tuple[float, float, float, float],
    height: float = 6.5,
    trace_width: float = 4.4,
    style: InkStyle = DEFAULT_INK,
    height_ratios: tuple[float, float] = (1.0, 1.0),
    margins: tuple[float, float, float, float] = (0.12, 0.12, 0.55, 0.18),
    gap: float = 0.95,
    **gridspec,
) -> ThreeView:
    """Return a three-view figure whose poloidal panel matches its machine.

    ``extent`` is the poloidal data's ``(r_min, r_max, z_min, z_max)``, and it
    is required: a poloidal panel holds equal aspect, so a box chosen without
    reference to the machine leaves the drawing floating in margin instead of
    filling the panel. The panel is given exactly the machine's aspect and the
    figure width follows from it, which is what makes the contours fill the
    plot at any machine's proportions.

    Every length is in inches, so the arithmetic below is a layout in physical
    units rather than a set of ratios that only happens to look right at one
    figure size.

    ``margins`` is ``(left, right, bottom, top)`` and is asymmetric by default
    for a concrete reason: the poloidal panel draws no spines or ticks and so
    needs no room, while the trace column's tick labels and axis labels sit
    outside its box and are clipped by a margin sized for the poloidal panel.
    """
    r_min, r_max, z_min, z_max = (float(value) for value in extent)
    if not (r_max > r_min and z_max > z_min):
        raise ValueError("the poloidal extent must span a positive area")
    import matplotlib.figure

    left, right, bottom, top = (float(value) for value in margins)
    panel_height = height - bottom - top
    if panel_height <= 0.0:
        raise ValueError("the margins leave no height for the panels")
    panel_width = panel_height * (r_max - r_min) / (z_max - z_min)
    width = left + panel_width + gap + trace_width + right

    figure = matplotlib.figure.Figure(
        figsize=(width, height), dpi=style.figure_dpi, facecolor=style.figure_facecolor
    )
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(panel_width, trace_width),
        height_ratios=height_ratios,
        left=left / width,
        right=1.0 - right / width,
        bottom=bottom / height,
        top=1.0 - top / height,
        wspace=gap / (0.5 * (panel_width + trace_width)),
        **{"hspace": 0.30, **gridspec},
    )
    view = ThreeView(
        figure=figure,
        poloidal=figure.add_subplot(grid[:, 0]),
        upper=figure.add_subplot(grid[0, 1]),
        lower=figure.add_subplot(grid[1, 1]),
        extent=(r_min, r_max, z_min, z_max),
    )
    return view.clear()
