"""Animate Nova's solved surfaces beside the Thomson profiles they sample.

Left: the wall-clipped hexagonal plasma mesh in light purple, cut to each
frame's solved boundary, under Nova's nested flux surfaces at their own
absolute flux with the boundary, divertor legs and topology points, and the
Thomson scattering volumes overlaid and coloured by system. Right: that
system's measured profiles as the pulse steps, core above and edge below.

Both right-hand panels hold limits measured once over the whole pulse, so a
channel that heats looks like a channel that heats rather than a still curve
under moving axes. Temperature is drawn on a logarithmic axis: core electron
temperature on this shot runs from a few hundred eV to 6.9 keV, so a fixed
linear axis would either clip the hot frames or flatten every other one.

MAST's two Thomson systems both view the midplane, so the figure separates
them by system rather than by sightline orientation -- there is no vertical
string on this machine to contrast with.

The machine outline comes from the shot's own EFIT record, which is the same
geometry the labelled solve was driven from.

Only free solves are drawn. A slice the labeller conditioned on the reference
current centroid is not something Nova found unaided, so presenting one as
"nova solve" would credit it with a position it was handed; those slices are
excluded and the counts recorded. Pass ``--include-conditioned`` to draw them,
in which case each such frame is labelled in the figure itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova.media import gif, poloidal as pol, traces as tr
from nova.media.ink import DEFAULT_INK
from nova.media.layout import three_view

DEFAULT_OUTPUT = Path("docs/figures/presentation-media")


def render_thomson(
    shot: int,
    path: Path,
    duration: float = 10.0,
    height: float = 6.5,
    quantile: float | None = 0.98,
    include_conditioned: bool = False,
    cells: int = 1200,
) -> dict[str, object]:
    """Write the surfaces-and-Thomson animation and return its receipt."""
    from nova.media.sources.mast_efit import read_pulse
    from nova.media.sources.mast_thomson import read_thomson
    from nova.media.sources.nova_labels import read_labels
    from nova.media.sources.plasma_mesh import clip_to_boundary, hex_mesh

    style = DEFAULT_INK
    frames, labels = read_labels(shot, free_only=not include_conditioned)
    strings = read_thomson(shot)
    if not strings:
        raise ValueError(f"MAST {shot} carries no Thomson system")
    machine = read_pulse(shot).geometry
    # Built once: the tessellation is a property of the wall, and only the
    # boundary cut changes between frames.
    mesh, mesh_provenance = hex_mesh(machine.limiter, cells=cells)
    clipped = [clip_to_boundary(mesh, frame.boundary) for frame in frames]

    palette = (style.thomson_primary_color, style.thomson_secondary_color)
    colour = {
        string.name: palette[index % len(palette)]
        for index, string in enumerate(strings)
    }
    positions = np.vstack([string.positions for string in strings])
    group = np.concatenate(
        [[string.name] * string.positions.shape[0] for string in strings]
    )

    # One scale per system over every admitted row, so the panels never move.
    # Each row is passed as its own series, because the quantile must cover a
    # fraction of the MEASUREMENTS rather than of the pooled samples: pooling
    # lets one hot row set the axis for the whole pulse.
    scales = {}
    for string in strings:
        usable = np.isfinite(string.temperature) & (string.temperature > 0.0)
        rows = list(np.where(usable, string.temperature, np.nan))
        scales[string.name] = tr.TraceScale.over(
            [string.positions[:, 0]], rows, quantile=quantile, log=True
        )

    view = three_view(extent=machine.bounds(), height=height, style=style)
    nulls: list[dict[str, int]] = []

    def render(index: int) -> None:
        frame = frames[index]
        view.clear()
        pol.draw_plasma_cells(view.poloidal, clipped[index])
        pol.draw_surfaces(view.poloidal, frame.surfaces)
        pol.draw_coils(view.poloidal, machine.coils)
        pol.draw_wall(view.poloidal, *machine.limiter.T)
        pol.draw_legs(view.poloidal, frame.legs)
        pol.draw_boundary(view.poloidal, *frame.boundary.T)
        # Strike points are not drawn: the brief asks for the O-point and the
        # X-points, and a strike point's containment against a 36-point
        # limiter is unresolved, so drawing one would put an unsettled
        # quantity on the slide.
        nulls.append(
            pol.draw_nulls(
                view.poloidal,
                magnetic_axis=frame.magnetic_axis,
                x_points=frame.x_points,
                contain=machine.limiter,
            )
        )
        pol.draw_thomson(view.poloidal, positions, group, chords=False)
        for axes, string in zip((view.upper, view.lower), strings):
            radius, temperature, _ = string.finite(frame.time)
            tr.draw_samples(
                axes,
                radius,
                temperature,
                color=colour[string.name],
                style=style,
                markersize=2.5,
            )
            scales[string.name].apply(axes)
            tr.label_axes(
                axes,
                "major radius  [m]" if axes is view.lower else None,
                f"{string.name} $T_e$  [eV]",
            )
        conditioning = (
            "\ncentroid pinned to EFIT" if frame.conditioned else "\nfree solve"
        )
        tr.annotate_time(
            view.upper,
            f"MAST {shot}  nova{conditioning}\nt = {1e3 * frame.time:.0f} ms",
        )

    receipt = gif.animate(
        view.figure,
        range(len(frames)),
        render,
        path,
        duration=duration,
        contact_sheet=Path(path).with_name(f"{Path(path).stem}-frames.png"),
    )
    receipt |= {
        "machine": "MAST",
        "pulse": str(shot),
        "left_panel": "nova nested flux surfaces over the clipped hex mesh",
        "plasma_mesh": mesh_provenance
        | {
            "boundary_clip": "geometric intersection with the solved boundary "
            "polygon; a rasterless session carries no flux field to cut at",
            "clipped_cell_count_range": [
                min(len(item) for item in clipped),
                max(len(item) for item in clipped),
            ],
        },
        "conditioned_frames_drawn": sum(1 for f in frames if f.conditioned),
        "x_points_drawn": sum(item["x_points_drawn"] for item in nulls),
        "x_points_dropped_outside_wall": sum(
            item["x_points_dropped_outside_wall"] for item in nulls
        ),
        "strike_points_drawn": "none; outside the brief and containment unresolved",
        "profile_limits_fixed": True,
        "profile_limit_quantile": quantile,
        "temperature_axis": "logarithmic, fixed over the pulse",
        "thomson_systems": [
            {
                "name": string.name,
                "channels": int(string.positions.shape[0]),
                "rows": len(string),
                "colour": colour[string.name],
                "temperature_limit_ev": list(scales[string.name].y_limit),
                "provenance": string.provenance,
            }
            for string in strings
        ],
        "orientation_note": (
            "both MAST systems view the midplane, so the split is core "
            "against edge rather than vertical against horizontal"
        ),
        "time_span_s": [float(frames[0].time), float(frames[-1].time)],
        "labels": labels,
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Render the Nova-plus-Thomson animation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=27079)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--height", type=float, default=6.5)
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--cells", type=int, default=1200)
    parser.add_argument(
        "--include-conditioned",
        action="store_true",
        help="also draw slices pinned to the reference current centroid",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    name = f"mast-{arguments.shot}-nova-thomson"
    receipt = render_thomson(
        arguments.shot,
        arguments.output / f"{name}.gif",
        duration=arguments.duration,
        height=arguments.height,
        quantile=arguments.quantile,
        include_conditioned=arguments.include_conditioned,
        cells=arguments.cells,
    )
    (arguments.output / f"{name}-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
