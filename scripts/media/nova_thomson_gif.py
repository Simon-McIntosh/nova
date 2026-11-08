"""Animate Nova's solved equilibria beside the Thomson profiles they sample.

Left: one free-boundary forward solve per EFIT slice, from
``scripts/media/mast_pulse_forward.py``. It shows the grid cells that carry
plasma current, nested flux surfaces at stated normalised flux, the boundary,
the magnetic axis and the saddle the solve's own topology read selected, all
inside the limiter, with the Thomson scattering volumes overlaid by system.
EFIT's boundary is drawn dashed for reference. Right: that system's measured
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
from nova.media.layout import three_view
from nova.media.poloidal import chord_crossings

DEFAULT_OUTPUT = Path("docs/figures/presentation-media")

#: The measured quantities a panel can carry, as (string attribute, label, unit).
_QUANTITIES = {
    "te": ("temperature", "$T_e$", "eV"),
    "ne": ("density", "$n_e$", "m$^{-3}$"),
}

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


def _cell_outlines(centres: np.ndarray, pitch: tuple[float, float]) -> list:
    """Return each current-carrying grid cell as its rectangle outline."""
    half = 0.5 * np.asarray(pitch)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * half
    return [centre + corners for centre in centres]


def render_thomson(
    shot: int,
    stack: list[Path],
    path: Path,
    duration: float = 10.0,
    height: float = 6.5,
    quantile: float | None = 0.98,
    quantity: str = "te",
    solved_edge: bool = True,
) -> dict[str, object]:
    """Write the solved-equilibria-and-Thomson animation and return its receipt.

    Every frame marks the edge the Thomson profile itself implies, dashed.
    ``solved_edge`` adds the solved boundary's chord crossing, solid, so the
    two can be compared; without it the profile panels carry the measurement
    and its own edge only.
    """
    from nova.diagnostics.thomson_edge import SEPARATRIX_TEMPERATURE, outboard_edge
    from nova.media.sources.mast_efit import read_pulse
    from nova.media.sources.mast_thomson import read_thomson
    from nova.media.sources.pulse_forward import SURFACE_LEVELS, read_pulse_forward

    style = DEFAULT_INK
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
    reference_time = np.array([frame.time for frame in reference.frames])

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
    scales = {
        name: tr.TraceScale.over(
            [sample[0] for sample in samples],
            [sample[1] for sample in samples],
            quantile=quantile,
            log=True,
        )
        for name, samples in resolved.items()
    }

    # Headroom above the core profile for the frame annotation, so the text
    # never sits on a measurement: two decades on the logarithmic axis.
    core = strings[0].name
    lower, upper = scales[core].y_limit
    scales[core] = replace(scales[core], y_limit=(lower, upper * 100.0))

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

    view = three_view(extent=machine.bounds(), height=height, style=style)
    nulls: list[dict[str, int]] = []

    def render(index: int) -> None:
        frame = frames[index]
        view.clear()
        pol.draw_plasma_cells(
            view.poloidal, _cell_outlines(frame.current_cells, pulse.pitch)
        )
        pol.draw_surfaces(view.poloidal, frame.surfaces)
        pol.draw_coils(view.poloidal, machine.coils)
        pol.draw_wall(view.poloidal, *machine.limiter.T)
        nearest = reference.frames[int(np.argmin(np.abs(reference_time - frame.time)))]
        view.poloidal.plot(
            *nearest.boundary.T,
            color=style.wall_color,
            linewidth=0.8,
            linestyle="--",
            zorder=style.zorder_markers - 1,
        )
        pol.draw_boundary(view.poloidal, *frame.boundary.T)
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
        for axes, string in zip((view.upper, view.lower), strings):
            radius, values, _sigma = resolved[string.name][index]
            tr.draw_samples(
                axes,
                radius,
                values,
                color=colour[string.name],
                style=style,
                markersize=2.5,
            )
            scales[string.name].apply(axes)
            tr.label_axes(
                axes,
                "major radius  [m]" if axes is view.lower else None,
                f"{string.name} {label}  [{unit}]",
            )
            if solved_edge:
                # Where the solved boundary crosses this string's own chord.
                for crossing in chord_crossings(frame.boundary, string.positions[0, 1]):
                    axes.axvline(
                        crossing,
                        color=style.separatrix_color,
                        linewidth=1.2,
                        zorder=1,
                    )
            if measured_edges[index] is not None:
                axes.axvline(
                    measured_edges[index],
                    color=MEASURED_EDGE_COLOR,
                    linewidth=1.2,
                    linestyle="--",
                    zorder=1,
                )
        state = "converged" if frame.qualified else f"residual {frame.residual:.1e}"
        topology = "diverted" if frame.diverted else "limited"
        legend = (
            "solid: solved edge   dashed: Thomson edge"
            if solved_edge
            else "dashed: Thomson edge"
        )
        tr.annotate_time(
            view.upper,
            f"MAST {shot}  nova forward\n{topology}, {state}\n"
            f"t = {1e3 * frame.time:.0f} ms\n{legend}",
        )

    receipt = gif.animate(
        view.figure,
        range(len(frames)),
        render,
        path,
        duration=duration,
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
            "free-boundary forward solve per EFIT slice: current-carrying grid "
            "cells, nested surfaces, boundary, axis and selected saddle; EFIT "
            "boundary dashed for reference"
        ),
        "forward_stack": pulse.provenance,
        "solve_pitch_m": list(pulse.pitch),
        "surface_levels_psi_norm": [float(level) for level in SURFACE_LEVELS],
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
        "defaults to the shot's 40 mm shards under the derived cache root",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--height", type=float, default=6.5)
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
            f"{arguments.shot}-pitch40mm-shard*.npz"
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
        duration=arguments.duration,
        height=arguments.height,
        quantile=arguments.quantile,
        quantity=arguments.quantity,
        solved_edge=not arguments.thomson_edge_only,
    )
    (arguments.output / f"{name}-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
