"""Animate Nova's solved equilibria beside the Thomson profiles they sample.

Left: one free-boundary forward solve per EFIT slice, from
``scripts/media/mast_pulse_forward.py``. It shows the hexagonal plasma cells
of the wall cut at the solved boundary, nested flux surfaces at stated
normalised flux, the boundary, the magnetic axis and the saddle the solve's own
topology read selected, with the Thomson scattering volumes overlaid by system.
Right: that system's measured
profile at the nearest laser pulse, core above and edge below.

The boundary is the edge of the axis-connected region below normalised flux
one inside the limiter (``nova.media.sources.pulse_forward``). A limited
plasma's edge therefore runs along the limiter where it touches it, and a
diverted plasma's edge passes through its X-point. Each frame's
boundary-to-limiter gap goes into the receipt by class, because a limited
frame with a gap above the solve pitch is a defect, not a standoff.

Only Thomson samples that resolve their quantity are drawn: a value at least
twice its own stored uncertainty (``ThomsonString.resolved``). Both right-hand
panels hold limits measured once over the whole pulse on those samples, on a
logarithmic axis, so a channel that heats looks like a channel that heats.

MAST's two Thomson systems both view the midplane, so the figure separates
them by system rather than by sightline orientation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from nova.media import gif, poloidal as pol, traces as tr
from nova.media.ink import DEFAULT_INK
from nova.media.poloidal import chord_crossings

DEFAULT_OUTPUT = Path("docs/figures/presentation-media")

#: The measured quantities a panel can carry, as (string attribute, label, unit).
_QUANTITIES = {
    "te": ("temperature", "$T_e$", "eV"),
    "ne": ("density", "$n_e$", "m$^{-3}$"),
}

#: Poloidal flux contour spacing over the whole vessel, in Wb (total flux).
FLUX_SPACING = 0.04

#: Thin flux contours, in the ink style's grey for vacuum and blue for plasma.
CONTOUR_LINEWIDTH = 1.2
BOUNDARY_LINEWIDTH = 2.4

#: The figure on a widescreen page: full slide width, 0.9 in short of the
#: 16:9 height so the slide keeps a footer band below it.
SLIDE_WIDTH = 13.333
SLIDE_HEIGHT = 6.6

#: Text size on the slide, in points: legible from the back of a room.
SLIDE_FONTSIZE = 18.0
SLIDE_TICK_FONTSIZE = 16.0

#: Margin around the first wall in the cropped poloidal panel, in metres.
WALL_MARGIN = 0.03

#: Height of the graph as a fraction of the column below the title room; the
#: graph keeps its bottom edge and the space it gives up opens above it.
GRAPH_HEIGHT_FRACTION = 0.75

#: Room above the graph column for the slide title, in inches.
TITLE_ROOM = 1.5


def slide_view(extent, style):
    """Return the graph on the left and the machine full height on the right.

    The figure is the full slide width. The machine's panel runs the full
    figure height at the extent's aspect, on the right, and the graph takes
    the rest of the width to its left, stopping short of the top to leave
    room for the slide title.
    """
    import matplotlib.figure

    from nova.media.layout import ThreeView

    r_min, r_max, z_min, z_max = (float(value) for value in extent)
    inset, gap, left, bottom = 0.1, 0.35, 1.4, 0.9
    panel_height = SLIDE_HEIGHT - 2 * inset
    panel_width = panel_height * (r_max - r_min) / (z_max - z_min)
    width = SLIDE_WIDTH
    column_width = width - left - gap - panel_width - inset
    figure = matplotlib.figure.Figure(
        figsize=(width, SLIDE_HEIGHT),
        dpi=style.figure_dpi,
        facecolor=style.figure_facecolor,
    )
    poloidal = figure.add_axes(
        (
            (width - inset - panel_width) / width,
            inset / SLIDE_HEIGHT,
            panel_width / width,
            panel_height / SLIDE_HEIGHT,
        )
    )
    graph_height = GRAPH_HEIGHT_FRACTION * (SLIDE_HEIGHT - TITLE_ROOM - bottom)
    upper = figure.add_axes(
        (
            left / width,
            bottom / SLIDE_HEIGHT,
            column_width / width,
            graph_height / SLIDE_HEIGHT,
        )
    )
    # ThreeView carries a second graph; this slide has one, so the second is
    # a hidden placeholder that is never drawn.
    lower = figure.add_axes((0.0, 0.0, 1e-3, 1e-3))
    return ThreeView(
        figure=figure,
        poloidal=poloidal,
        upper=upper,
        lower=lower,
        extent=(r_min, r_max, z_min, z_max),
    ).clear()


#: Line colour of the edge inferred from the Thomson profile alone.
MEASURED_EDGE_COLOR = "#222222"

#: A Thomson sample is drawn when its value is at least this many sigma.
SIGNAL_TO_NOISE = 2.0


def _gap_summary(gaps: np.ndarray) -> dict[str, float] | None:
    """Return the millimetre gap statistics of one topology class."""
    finite = gaps[np.isfinite(gaps)]
    if finite.size == 0:
        return None
    return {
        "count": int(finite.size),
        "median_mm": float(1e3 * np.median(finite)),
        "max_mm": float(1e3 * finite.max()),
    }


def render_thomson(
    shot: int,
    stack: list[Path],
    path: Path,
    quantile: float | None = 0.98,
    quantity: str = "te",
    solved_edge: bool = True,
    cells: int = 4000,
    slowdown: float = 10.0,
    dpi: float = 144.0,
) -> dict[str, object]:
    """Write the solved-equilibria-and-Thomson animation and return its receipt.

    Every frame marks the edge the Thomson profile itself implies, dashed.
    ``solved_edge`` adds the solved boundary's chord crossing, solid, so the
    two can be compared; without it the profile panels carry the measurement
    and its own edge only.
    """
    from nova.diagnostics.thomson_edge import SEPARATRIX_TEMPERATURE, outboard_edge
    from nova.media.sources.mast_efit import read_pulse
    from nova.media.sources.plasma_mesh import clip_to_boundary, hex_mesh
    from nova.media.sources.mast_thomson import read_thomson
    from contourpy import contour_generator
    from matplotlib.path import Path as MplPath

    from nova.media.sources.pulse_forward import read_pulse_forward

    style = replace(DEFAULT_INK, label_fontsize=SLIDE_FONTSIZE)
    if quantity not in _QUANTITIES:
        raise ValueError(
            f"unknown quantity {quantity!r}, not one of {list(_QUANTITIES)}"
        )
    column, label, unit = _QUANTITIES[quantity]
    pulse = read_pulse_forward(stack)
    # A boundary held by neither the X-point nor the wall is not a plasma
    # state, so it is never drawn: one solve pitch is the tolerance, the
    # resolution the boundary itself is known to.
    tolerance = max(pulse.pitch)
    frames = tuple(item for item in pulse.slices if item.bounded(tolerance))
    unbounded = [
        {
            "time_s": item.time,
            "class": "diverted" if item.diverted else "limited",
            "x_point_gap_mm": 1e3 * item.x_point_gap,
            "wall_gap_mm": 1e3 * item.wall_gap,
            "qualified": item.qualified,
        }
        for item in pulse.slices
        if not item.bounded(tolerance)
    ]
    if not frames:
        raise ValueError(f"the forward stack {stack} carries no bounded slice")
    strings = read_thomson(shot)
    if not strings:
        raise ValueError(f"MAST {shot} carries no Thomson system")
    reference = read_pulse(shot)
    machine = reference.geometry

    palette = (style.thomson_primary_color, style.thomson_secondary_color)
    colour = {
        string.name: palette[index % len(palette)]
        for index, string in enumerate(strings)
    }
    positions = np.vstack([string.positions for string in strings])
    group = np.concatenate(
        [[string.name] * string.positions.shape[0] for string in strings]
    )

    # One scale per system over every drawn frame's resolved samples, so the
    # panels never move. Each frame is its own series so the quantile covers
    # measurements rather than pooled samples.
    resolved = {
        string.name: [
            string.resolved(frame.time, column, SIGNAL_TO_NOISE) for frame in frames
        ]
        for string in strings
    }

    # Each system's scale is measured over its resolved samples at every laser
    # pulse of the shot, not only the drawn frames, so a system with no
    # resolved sample in the drawn frames still holds a pulse-wide axis.
    def pulse_scale(string):
        samples = [
            string.resolved(time, column, SIGNAL_TO_NOISE) for time in string.time
        ]
        samples = [sample for sample in samples if sample[0].size]
        return tr.TraceScale.over(
            [sample[0] for sample in samples],
            [sample[1] for sample in samples],
            quantile=quantile,
            log=True,
        )

    scales = {string.name: pulse_scale(string) for string in strings}
    # The one panel's scale covers every system: the wider radial span and the
    # wider value range of the per-system scales.
    scale = replace(
        scales[strings[0].name],
        x_limit=(
            min(item.x_limit[0] for item in scales.values()),
            max(item.x_limit[1] for item in scales.values()),
        ),
        y_limit=(
            min(item.y_limit[0] for item in scales.values()),
            max(item.y_limit[1] for item in scales.values()),
        ),
    )

    # The measured edge comes from the electron temperature of every system
    # pooled, whichever quantity the panels show: it is one physical edge.
    measured_edges = []
    for frame in frames:
        parts = [
            string.resolved(frame.time, "temperature", SIGNAL_TO_NOISE)
            for string in strings
        ]
        fit = outboard_edge(
            *(np.concatenate([part[k] for part in parts]) for k in range(3))
        )
        measured_edges.append(None if fit is None else fit.edge)

    # The poloidal panel is cropped to the first wall, so the wall fills the
    # panel height and the plasma is drawn as large as the slide allows.
    wall_min = machine.limiter.min(axis=0)
    wall_max = machine.limiter.max(axis=0)
    view = slide_view(
        (
            wall_min[0] - WALL_MARGIN,
            wall_max[0] + WALL_MARGIN,
            wall_min[1] - WALL_MARGIN,
            wall_max[1] + WALL_MARGIN,
        ),
        style,
    )
    nulls: list[dict[str, int]] = []
    # The plasma cells are a hexagonal tessellation of the wall, each cell cut
    # at the solved boundary, so the drawn domain ends exactly on the boundary
    # with no stair-step. The tessellation is a property of the wall and is
    # built once; only the cut changes between frames.
    mesh, mesh_provenance = hex_mesh(machine.limiter, cells=cells)
    clipped = [clip_to_boundary(mesh, frame.boundary) for frame in frames]

    # Flux is contoured over the whole vessel at one fixed spacing, anchored on
    # each frame's boundary flux, so the boundary is itself a contour and the
    # plasma and vacuum families meet without a gap. Every level is drawn grey
    # over the vessel; the closed plasma surfaces are redrawn blue on top from
    # the same field and the same levels.
    fine_r, fine_z, _ = frames[0].field
    rr, zz = np.meshgrid(fine_r, fine_z, indexing="ij")
    inside = (
        MplPath(machine.limiter)
        .contains_points(np.column_stack([rr.ravel(), zz.ravel()]))
        .reshape(rr.shape)
    )

    def draw_flux(frame) -> int:
        _r, _z, psi = frame.field
        psi = np.asarray(psi, dtype=float)
        vessel = np.where(inside, psi, np.nan)
        lowest = np.floor((np.nanmin(vessel) - frame.boundary_flux) / FLUX_SPACING)
        highest = np.ceil((np.nanmax(vessel) - frame.boundary_flux) / FLUX_SPACING)
        levels = frame.boundary_flux + FLUX_SPACING * np.arange(lowest, highest + 1)
        view.poloidal.contour(
            fine_r,
            fine_z,
            vessel.T,
            levels=levels,
            colors=style.contour_color,
            linewidths=CONTOUR_LINEWIDTH,
            negative_linestyles="solid",
            zorder=2,
        )
        span = frame.boundary_flux - frame.axis_flux
        generator = contour_generator(fine_r, fine_z, psi.T)
        axis = frame.magnetic_axis
        drawn = 0
        for level in levels:
            fraction = (level - frame.axis_flux) / span
            if not 0.0 < fraction < 1.0 - 1e-9:
                continue
            for ring in generator.lines(float(level)):
                closed = np.allclose(ring[0], ring[-1])
                if closed and MplPath(ring).contains_point(axis):
                    view.poloidal.plot(
                        *ring.T,
                        color=style.flux_color,
                        linewidth=CONTOUR_LINEWIDTH,
                        zorder=3,
                    )
                    drawn += 1
        return drawn

    plasma_contours: list[int] = []

    def render(index: int) -> None:
        frame = frames[index]
        view.clear()
        view.lower.set_visible(False)
        view.upper.tick_params(labelsize=SLIDE_TICK_FONTSIZE)
        pol.draw_plasma_cells(view.poloidal, clipped[index])
        plasma_contours.append(draw_flux(frame))
        pol.draw_coils(view.poloidal, machine.coils)
        pol.draw_wall(view.poloidal, *machine.limiter.T)
        pol.draw_boundary(
            view.poloidal, *frame.boundary.T, linewidth=BOUNDARY_LINEWIDTH
        )
        # The solve's selected saddle is the boundary's X-point only on a
        # diverted slice; on a limited slice it is a qualified null the
        # boundary does not pass through, drawn hollow.
        admitted = frame.x_point[None, :] if frame.diverted else None
        other = None if frame.diverted else frame.x_point[None, :]
        nulls.append(
            pol.draw_nulls(
                view.poloidal,
                magnetic_axis=frame.magnetic_axis,
                x_points=admitted,
                contain=machine.limiter,
                other_x_points=other,
            )
        )
        pol.draw_thomson(view.poloidal, positions, group, chords=False)
        if measured_edges[index] is not None:
            view.poloidal.plot(
                [measured_edges[index]] * 2,
                [-0.12, 0.12],
                color=MEASURED_EDGE_COLOR,
                linewidth=1.6,
                linestyle="--",
                zorder=style.zorder_markers,
            )
        # One Thomson panel: every system on the same axes, core blue and
        # edge orange, with the solved and measured edges across both.
        axes = view.upper
        for string in strings:
            radius, values, _sigma = resolved[string.name][index]
            tr.draw_samples(
                axes,
                radius,
                values,
                color=colour[string.name],
                style=style,
                markersize=7.0,
            )
        scale.apply(axes)
        tr.label_axes(axes, "major radius  [m]", f"{label}  [{unit}]", style=style)
        if solved_edge:
            for crossing in chord_crossings(frame.boundary, strings[0].positions[0, 1]):
                axes.axvline(
                    crossing, color=style.separatrix_color, linewidth=2.4, zorder=1
                )
        if measured_edges[index] is not None:
            axes.axvline(
                measured_edges[index],
                color=MEASURED_EDGE_COLOR,
                linewidth=2.4,
                linestyle="--",
                zorder=1,
            )

    # Equal pulse time per unit of animation time: each frame is held for the
    # gap to the next solved slice, scaled by the slowdown, so a missing slice
    # is a held frame rather than a jump. The last frame is held for the
    # median gap.
    times = np.array([frame.time for frame in frames])
    gaps_s = np.diff(times)
    hold = np.append(gaps_s, np.median(gaps_s) if gaps_s.size else 0.005)
    durations = [max(10, int(round(1e3 * slowdown * gap / 10.0)) * 10) for gap in hold]
    receipt = gif.animate(
        view.figure,
        range(len(frames)),
        render,
        path,
        durations=durations,
        dpi=dpi,
        contact_sheet=Path(path).with_name(f"{Path(path).stem}-frames.png"),
    )
    gaps = np.array([frame.wall_gap for frame in frames])
    limited = np.array([not frame.diverted for frame in frames])
    current_ratio = np.array(
        [frame.plasma_current / frame.target_current for frame in frames]
    )
    receipt |= {
        "machine": "MAST",
        "pulse": str(shot),
        "left_panel": (
            "free-boundary forward solve per EFIT slice: hex plasma cells cut "
            "at the solved boundary, nested surfaces, boundary, axis and "
            "selected saddle"
        ),
        "forward_stack": pulse.provenance,
        "solve_pitch_m": list(pulse.pitch),
        "solve_grid": "rectangular lattice at solve_pitch_m; current on its nodes",
        "plasma_mesh": mesh_provenance
        | {
            "role": "display tessellation of the wall, each cell cut at the "
            "solved boundary",
        },
        "flux_contour_spacing_wb": FLUX_SPACING,
        "flux_contours": (
            "whole vessel at fixed spacing anchored on the boundary flux; grey "
            "for every level, blue over the closed surfaces enclosing the axis"
        ),
        "plasma_contours_per_frame": [min(plasma_contours), max(plasma_contours)],
        "timing": {
            "rule": "each frame held for the pulse-time gap to the next slice",
            "slowdown": slowdown,
            "frame_milliseconds_range": [min(durations), max(durations)],
        },
        "frames_drawn": len(frames),
        "boundary_hold_tolerance_m": tolerance,
        "frames_dropped_unbounded": unbounded,
        "frames_converged": int(sum(frame.qualified for frame in frames)),
        "frames_limited": int(limited.sum()),
        "frames_diverted": int((~limited).sum()),
        "plasma_current_ratio_range": [
            float(current_ratio.min()),
            float(current_ratio.max()),
        ],
        "x_points_drawn": sum(item["x_points_drawn"] for item in nulls),
        "other_x_points_drawn": sum(item["other_x_points_drawn"] for item in nulls),
        "quantity": quantity,
        "quantity_label": f"{label} [{unit}]",
        "thomson_screen": (
            f"value at least {SIGNAL_TO_NOISE:g} times its stored one-sigma uncertainty"
        ),
        "profile_axis": "logarithmic, fixed over the pulse",
        "solved_edge_drawn": solved_edge,
        "thomson_edge": {
            "model": (
                "uncertainty-weighted mtanh fit of the outboard electron "
                "temperature of every system pooled; edge where the fit falls "
                f"through {SEPARATRIX_TEMPERATURE:g} eV, the two-point-model "
                "upstream separatrix temperature"
            ),
            "frames_resolved": int(sum(edge is not None for edge in measured_edges)),
            "solved_minus_thomson_m": [
                float(crossing[-1] - edge)
                for frame, edge in zip(frames, measured_edges)
                if edge is not None
                and (crossing := chord_crossings(frame.boundary, 0.0))
            ],
        },
        "profile_limit_quantile": quantile,
        "boundary_wall_gap_mm": {
            "limited": _gap_summary(gaps[limited]),
            "diverted": _gap_summary(gaps[~limited]),
            "note": (
                "a limited boundary must touch its limiter, so a limited gap "
                "above the solve pitch is a defect; the diverted gap is expected"
            ),
        },
        "thomson_systems": [
            {
                "name": string.name,
                "channels": int(string.positions.shape[0]),
                "rows": len(string),
                "colour": colour[string.name],
                "value_limit": list(scales[string.name].y_limit),
                "resolved_samples_drawn": int(
                    sum(len(sample[1]) for sample in resolved[string.name])
                ),
                "provenance": string.provenance,
            }
            for string in strings
        ],
        "time_span_s": [float(frames[0].time), float(frames[-1].time)],
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Render the Nova-plus-Thomson animation from the command line."""
    from nova.database.filepath import derived_cache_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=27079)
    parser.add_argument(
        "--stack",
        type=Path,
        nargs="+",
        default=None,
        help="forward stacks written by mast_pulse_forward.py, one per shard; "
        "defaults to the shot's vertically pinned 40 mm shards",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=10.0,
        help="animation time per unit of pulse time; frames are held for their "
        "own pulse-time gap, so playback is uniform in pulse time",
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=144.0,
        help="rasterisation resolution; 144 gives 1920x1080 on the 16:9 page",
    )
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument(
        "--quantity",
        choices=tuple(_QUANTITIES),
        default="te",
        help="which measured profile the right-hand panels carry",
    )
    parser.add_argument(
        "--label",
        default="",
        help="appended to the figure basename so a variant coexists with the "
        "default rather than superseding it",
    )
    parser.add_argument(
        "--thomson-edge-only",
        action="store_true",
        help="mark only the edge the Thomson profile implies, not the solved one",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    stack = arguments.stack or sorted(
        (derived_cache_root() / "pulse-forward").glob(
            f"{arguments.shot}-pitch40mm-shard-pinned*.npz"
        )
    )
    suffix = "" if arguments.quantity == "te" else f"-{arguments.quantity}"
    label = f"-{arguments.label}" if arguments.label else ""
    if arguments.thomson_edge_only:
        label = f"-thomson-edge{label}"
    name = f"mast-{arguments.shot}-nova-thomson{suffix}{label}"
    receipt = render_thomson(
        arguments.shot,
        stack,
        arguments.output / f"{name}.gif",
        quantile=arguments.quantile,
        quantity=arguments.quantity,
        solved_edge=not arguments.thomson_edge_only,
        slowdown=arguments.slowdown,
        dpi=arguments.dpi,
    )
    (arguments.output / f"{name}-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
