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
from nova.media.poloidal import boundary_wall_gap, chord_crossings
from nova.media.ink import DEFAULT_INK
from nova.media.layout import three_view

DEFAULT_OUTPUT = Path("docs/figures/presentation-media")

#: The measured quantities a panel can carry, as (string attribute, label, unit).
_QUANTITIES = {
    "te": ("temperature", "$T_e$", "eV"),
    "ne": ("density", "$n_e$", "m$^{-3}$"),
}


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


def _admit_by_area(frames, reference, floor: float | None) -> dict[str, object]:
    """Return which frames carry a plasma area worth drawing, and what dropped.

    The criterion is the drawn boundary area against the reconstruction own
    boundary area at the nearest slice. Wall contact is the wrong instrument
    here even though the defect is a boundary standing off the wall: a
    diverted boundary is bounded by the separatrix rather than by contact and
    correctly stands about 100 mm off, so a standoff threshold fires on the
    healthy population as well as the defective one and selects for nothing.

    The floor is stated rather than tuned. The two populations sit either
    side of half the reference area -- diverted frames do not fall below
    about 0.69, and the frames that collapse fall to 0.04 -- so a floor at
    one half lies in the gap between them rather than among the samples.

    Nothing here repairs anything: the corpus is unchanged and the frames are
    selected. Both counts and every dropped ratio are returned so the
    selection appears in the receipt instead of as a shorter animation.
    """
    from shapely.geometry import Polygon

    stored = np.array([frame.time for frame in reference])
    areas = np.array([Polygon(frame.boundary).area for frame in reference])
    ratios = np.array(
        [
            Polygon(frame.boundary).area
            / areas[int(np.argmin(np.abs(stored - frame.time)))]
            for frame in frames
        ]
    )
    keep = np.ones(len(frames), dtype=bool) if floor is None else ratios >= floor
    return {
        "rule": (
            "drawn boundary area at least this fraction of the reconstruction "
            "boundary area on the nearest slice"
        ),
        "floor": floor,
        "floor_basis": (
            "one half, lying in the gap between the two populations rather "
            "than among the samples: diverted frames stay above about 0.69 "
            "and collapsed frames fall to 0.04"
        ),
        "wall_contact_rejected_as_instrument": (
            "a standoff threshold fires on the healthy population too, since "
            "a diverted boundary correctly stands about 100 mm off the wall"
        ),
        "corpus": (
            "unrepaired; these frames were selected, not fixed, and the "
            "limited class still fails wall contact where it is drawn"
        ),
        "keep": keep,
        "admitted_frame_count": int(keep.sum()),
        "dropped_frame_count": int((~keep).sum()),
        "dropped": [
            {"time_s": float(frame.time), "area_ratio": float(ratio)}
            for frame, ratio, ok in zip(frames, ratios, keep)
            if not ok
        ],
        "admitted_area_ratio_range": (
            [float(ratios[keep].min()), float(ratios[keep].max())]
            if keep.any()
            else None
        ),
    }


def render_thomson(
    shot: int,
    path: Path,
    duration: float = 10.0,
    height: float = 6.5,
    quantile: float | None = 0.98,
    include_conditioned: bool = False,
    cells: int = 1200,
    quantity: str = "te",
    session: Path | None = None,
    min_area_fraction: float | None = None,
) -> dict[str, object]:
    """Write the surfaces-and-Thomson animation and return its receipt.

    ``session`` names the labeller session root to read. It is worth being an
    argument rather than a module constant: a corpus relabelled after a solver
    repair is written to a NEW root so the earlier artefact survives, and a
    figure that could only read one root could not be regenerated against the
    repair. The root reached is recorded in the receipt's label provenance, so
    two receipts can be told apart by what they read rather than by their
    timestamps.
    """
    from nova.media.sources.mast_efit import read_pulse
    from nova.media.sources.mast_thomson import read_thomson
    from nova.media.sources.nova_labels import read_labels
    from nova.media.sources.plasma_mesh import clip_to_boundary, hex_mesh

    style = DEFAULT_INK
    keywords = {} if session is None else {"dirname": session}
    frames, labels = read_labels(shot, free_only=not include_conditioned, **keywords)
    strings = read_thomson(shot)
    if not strings:
        raise ValueError(f"MAST {shot} carries no Thomson system")
    reference = read_pulse(shot)
    machine = reference.geometry
    admission = _admit_by_area(frames, reference.frames, min_area_fraction)
    frames = [frame for frame, keep in zip(frames, admission["keep"]) if keep]
    if not frames:
        raise ValueError(
            f"no frame of MAST {shot} carries a plasma area at least "
            f"{min_area_fraction} of the reconstruction, so there is nothing "
            "to animate; the corpus is unrepaired and this is a result"
        )
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

    if quantity not in _QUANTITIES:
        raise ValueError(
            f"unknown quantity {quantity!r}, not one of {list(_QUANTITIES)}"
        )
    column, label, unit = _QUANTITIES[quantity]

    # One scale per system over every admitted row, so the panels never move.
    # Each row is passed as its own series, because the quantile must cover a
    # fraction of the MEASUREMENTS rather than of the pooled samples: pooling
    # lets one hot row set the axis for the whole pulse.
    scales = {}
    for string in strings:
        stored = getattr(string, column)
        usable = np.isfinite(stored) & (stored > 0.0)
        rows = list(np.where(usable, stored, np.nan))
        scales[string.name] = tr.TraceScale.over(
            [string.positions[:, 0]], rows, quantile=quantile, log=True
        )

    view = three_view(extent=machine.bounds(), height=height, style=style)
    nulls: list[dict[str, int]] = []

    # A limited boundary must touch its limiter. Reporting the standoff per
    # class turns a boundary that has collapsed inward into a receipt number
    # rather than something a reader has to catch in the animation.
    gaps = np.array(
        [boundary_wall_gap(frame.boundary, machine.limiter) for frame in frames]
    )
    limited = np.array([frame.diverted is False for frame in frames])

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
            radius, temperature, density = string.finite(frame.time)
            values = temperature if column == "temperature" else density
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
            # The boundary is the thing the profile is meant to reveal, so mark
            # where the solved separatrix crosses this string's own chord.
            for crossing in chord_crossings(frame.boundary, string.positions[0, 1]):
                axes.axvline(
                    crossing,
                    color=style.separatrix_color,
                    linewidth=0.8,
                    alpha=0.7,
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
        "quantity": quantity,
        "quantity_label": f"{label} [{unit}]",
        "profile_axis": "logarithmic, fixed over the pulse",
        "separatrix_crossing_marked": True,
        "boundary_wall_gap_mm": {
            "limited": _gap_summary(gaps[limited]),
            "diverted": _gap_summary(gaps[~limited]),
            "note": (
                "a limited boundary must touch its limiter, so a non-zero "
                "limited gap is a solve defect rather than a standoff; the "
                "diverted gap is expected"
            ),
        },
        "thomson_systems": [
            {
                "name": string.name,
                "channels": int(string.positions.shape[0]),
                "rows": len(string),
                "colour": colour[string.name],
                "value_limit": list(scales[string.name].y_limit),
                "provenance": string.provenance,
            }
            for string in strings
        ],
        "orientation_note": (
            "both MAST systems view the midplane, so the split is core "
            "against edge rather than vertical against horizontal"
        ),
        "time_span_s": [float(frames[0].time), float(frames[-1].time)],
        "frame_admission": {
            key: value for key, value in admission.items() if key != "keep"
        },
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
        "--quantity",
        choices=tuple(_QUANTITIES),
        default="te",
        help="which measured profile the right-hand panels carry",
    )
    parser.add_argument(
        "--include-conditioned",
        action="store_true",
        help="also draw slices pinned to the reference current centroid",
    )
    parser.add_argument(
        "--min-area-fraction",
        type=float,
        default=None,
        help="drop a frame whose boundary area falls below this fraction of "
        "the reconstruction area on the nearest slice; selects frames, "
        "repairs nothing",
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="labeller session root to read; defaults to the root the source "
        "module names, and a corpus relabelled after a solver repair lands in "
        "a new one",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    suffix = "" if arguments.quantity == "te" else f"-{arguments.quantity}"
    name = f"mast-{arguments.shot}-nova-thomson{suffix}"
    receipt = render_thomson(
        arguments.shot,
        arguments.output / f"{name}.gif",
        duration=arguments.duration,
        height=arguments.height,
        quantile=arguments.quantile,
        include_conditioned=arguments.include_conditioned,
        cells=arguments.cells,
        quantity=arguments.quantity,
        session=arguments.session,
        min_area_fraction=arguments.min_area_fraction,
    )
    (arguments.output / f"{name}-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
