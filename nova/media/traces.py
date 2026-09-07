"""Trace panels whose limits are fixed once for a whole pulse.

A panel that autoscales per frame makes an animation unreadable: the curve
holds still while the axes move under it, so a profile that doubles looks
identical to one that halves. :class:`TraceScale` measures the limits once
over every frame that will be drawn, and each frame is then drawn inside
them.

The scale is therefore computed from the same arrays the frames will use --
not from the first frame, and not from a nominal range -- so a pulse whose
peak arrives late is still inside the axes when it does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from nova.media.ink import DEFAULT_INK, InkStyle

if TYPE_CHECKING:
    import matplotlib


@dataclass(frozen=True)
class TraceScale:
    """Fixed axis limits for one trace panel."""

    x_limit: tuple[float, float]
    y_limit: tuple[float, float]
    log: bool = False

    @classmethod
    def over(
        cls,
        abscissa: Iterable[Sequence[float]] | Sequence[float],
        ordinate: Iterable[Sequence[float]],
        pad: float = 0.05,
        symmetric: bool = False,
        quantile: float | None = None,
        log: bool = False,
    ) -> TraceScale:
        """Measure limits covering every frame's data.

        ``pad`` is a fraction of the measured span, added on both ends so a
        curve never touches the frame. ``symmetric`` centres the ordinate on
        zero, which keeps the sign of a quantity like ``FF'`` readable as a
        position rather than a colour.

        ``quantile`` bounds the ordinate by a central interval instead of by
        its extremes -- pass 0.99 to keep the middle 99 per cent. A pulse's
        first and last reconstructed slices carry a nearly extinguished plasma
        whose flux-function gradients run orders of magnitude above the flat
        top, and an extreme-valued scale spends the whole animation's vertical
        range on those two frames, flattening every frame anyone wants to see.
        The excluded frames are then drawn outside the axes rather than
        rescaling it, which is the intended trade and worth stating in a
        caption.

        ``log`` measures the ordinate over its positive samples only and
        returns limits a logarithmic axis can hold. Reach for it when the
        quantity spans the pulse by more than about a decade: measured on MAST
        27079, core electron temperature runs from a few hundred eV at the
        edge of the flat top to 6.9 keV, so no fixed LINEAR axis can both
        avoid clipping and leave the ordinary frames readable. A log axis
        satisfies the fixed-scale requirement and stays legible at both ends.
        """
        x = _finite_bounds(abscissa)
        positive = _positive(ordinate) if log else ordinate
        y = (
            _finite_bounds(positive)
            if quantile is None
            else _quantile_bounds(positive, quantile)
        )
        if symmetric:
            reach = max(abs(y[0]), abs(y[1]))
            y = (-reach, reach)
        if log:
            return cls(x_limit=_padded(x, pad), y_limit=_log_padded(y), log=True)
        return cls(x_limit=_padded(x, pad), y_limit=_padded(y, pad))

    def apply(self, axes: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """Pin ``axes`` to these limits and stop it autoscaling again."""
        if self.log:
            axes.set_yscale("log")
        axes.set_xlim(*self.x_limit)
        axes.set_ylim(*self.y_limit)
        axes.set_autoscale_on(False)
        return axes


def _series(values) -> list[np.ndarray]:
    """Return the input as a list of arrays, whether it is one or many.

    A single profile and a sequence of per-frame profiles are both accepted,
    so a caller does not have to wrap one frame to measure it.
    """
    array = np.asarray(values, dtype=object if _ragged(values) else float)
    if array.dtype == object or array.ndim > 1:
        return [np.asarray(item, dtype=float).reshape(-1) for item in values]
    return [array.reshape(-1)]


def _ragged(values) -> bool:
    """Return whether ``values`` is a sequence of differently shaped arrays."""
    try:
        lengths = {len(np.asarray(item, dtype=float)) for item in values}
    except TypeError:
        return False
    return len(lengths) > 1


def _finite_bounds(values) -> tuple[float, float]:
    """Return the (low, high) finite bounds over one or many arrays."""
    finite = np.concatenate(
        [array[np.isfinite(array)] for array in _series(values)] or [np.zeros(0)]
    )
    if finite.size == 0:
        raise ValueError("a trace scale needs at least one finite sample")
    return float(np.min(finite)), float(np.max(finite))


def _quantile_bounds(values, quantile: float) -> tuple[float, float]:
    """Return bounds covering the given fraction of FRAMES, not of samples.

    Each frame contributes only its own extremes, and the quantile is taken
    over those. Pooling every sample instead would weight a frame by how many
    points it carries, so one unconverged reconstruction whose profile spikes
    across its whole abscissa keeps its excursion inside almost any sample
    quantile -- measured on MAST 21978, a single 20 ms frame reaching
    -7.8e5 Pa/Wb against every other frame's -4.3e3 survived a 0.995 sample
    quantile and set the axis for the whole animation.
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError("a quantile must lie in (0, 1]")
    series = _series(values)
    lows, highs = [], []
    for array in series:
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        lows.append(float(np.min(finite)))
        highs.append(float(np.max(finite)))
    if not lows:
        raise ValueError("a trace scale needs at least one finite sample")
    tail = 1.0 - quantile
    return (
        float(np.quantile(lows, tail)),
        float(np.quantile(highs, 1.0 - tail)),
    )


def _positive(values) -> list[np.ndarray]:
    """Return each series with its non-positive samples masked out.

    A logarithmic axis cannot hold a zero or a negative sample, and a channel
    reporting one is a failed measurement rather than a cold plasma, so it is
    masked rather than clamped to a floor that would invent a value.
    """
    masked = []
    for array in _series(values):
        masked.append(np.where(array > 0.0, array, np.nan))
    return masked


def _log_padded(bounds: tuple[float, float]) -> tuple[float, float]:
    """Widen ``bounds`` by a fixed factor for a logarithmic axis."""
    low, high = bounds
    if not (low > 0.0 and high > 0.0):
        raise ValueError("a logarithmic scale needs positive bounds")
    return low / 1.6, high * 1.6


def _padded(bounds: tuple[float, float], pad: float) -> tuple[float, float]:
    """Widen ``bounds`` by ``pad`` of its span, or by one unit when flat."""
    low, high = bounds
    span = high - low
    if span == 0.0:
        reach = abs(high) * pad or 1.0
        return low - reach, high + reach
    return low - pad * span, high + pad * span


def draw_trace(
    axes: matplotlib.axes.Axes,
    abscissa: Sequence[float],
    ordinate: Sequence[float],
    style: InkStyle = DEFAULT_INK,
    label: str | None = None,
    color: str | None = None,
    **kwargs,
) -> None:
    """Draw one profile line into a prepared trace panel."""
    axes.plot(
        np.asarray(abscissa, dtype=float),
        np.asarray(ordinate, dtype=float),
        color=color or style.flux_color,
        linewidth=kwargs.pop("linewidth", style.trace_linewidth),
        label=label,
        **kwargs,
    )


def draw_samples(
    axes: matplotlib.axes.Axes,
    abscissa: Sequence[float],
    ordinate: Sequence[float],
    style: InkStyle = DEFAULT_INK,
    color: str | None = None,
    label: str | None = None,
    **kwargs,
) -> None:
    """Draw one set of measured samples as markers rather than a line.

    Measurements are drawn as points because the gaps between channels carry
    information: a line through them would invent a profile between volumes
    the instrument never sampled.
    """
    axes.plot(
        np.asarray(abscissa, dtype=float),
        np.asarray(ordinate, dtype=float),
        marker="o",
        markersize=kwargs.pop("markersize", style.trace_markersize),
        color=color or style.thomson_primary_color,
        linestyle="none",
        label=label,
        **kwargs,
    )


def annotate_time(
    axes: matplotlib.axes.Axes,
    text: str,
    style: InkStyle = DEFAULT_INK,
    position: tuple[float, float] = (0.02, 0.97),
) -> None:
    """Write the frame's time into a panel corner, in axes coordinates."""
    axes.text(
        *position,
        text,
        transform=axes.transAxes,
        fontsize=style.label_fontsize,
        va="top",
        ha="left",
        bbox=dict(style.label_bbox),
        zorder=style.zorder_label,
    )


def label_axes(
    axes: matplotlib.axes.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    style: InkStyle = DEFAULT_INK,
) -> None:
    """Label a trace panel at the house text size."""
    if xlabel is not None:
        axes.set_xlabel(xlabel, fontsize=style.label_fontsize)
    if ylabel is not None:
        axes.set_ylabel(ylabel, fontsize=style.label_fontsize)
