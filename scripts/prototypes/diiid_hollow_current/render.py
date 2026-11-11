"""Show source terms and saved-map current with shared physical scales."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from scripts.prototypes.diiid_xpoint_offset.style import (
    SOLVED,
    REFERENCE,
    WALL,
    CANVAS,
    save,
)
from scripts.prototypes.diiid_xpoint_offset.render import draw_critical_points
from scripts.prototypes.diiid_xpoint_offset.decompose import read_case
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_wall, draw_nulls, draw_flux_contours

HERE = Path(__file__).resolve().parent


def line_panels(record, output, previews):
    """Compare both source contributions on one fixed EFIT midplane."""
    states = {row["stage"]: row for row in record["states"]}
    figure, axes = plt.subplots(1, 3, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.80, wspace=0.38)
    for ax, key, name in zip(
        axes,
        ("density", "pressure_term", "field_term"),
        ("total current", "pressure contribution", "field contribution"),
        strict=True,
    ):
        for stage, color, linestyle in [
            ("reference", REFERENCE, "--"),
            ("terminal", SOLVED, "-"),
        ]:
            cut = states[stage]["cuts"]["midplane"]
            values = np.asarray(cut[key]) / 1e6
            if key != "density":
                values = values * states[stage]["amplitude"]
                values = np.where(cut["support"], values, 0)
            ax.plot(
                cut["x"],
                values,
                color=color,
                ls=linestyle,
                lw=3 if stage == "terminal" else 2.6,
                zorder=4 if stage == "terminal" else 2,
            )
            if key == "density":
                native = np.asarray(cut["map_density"]) / 1e6
                native = np.where(cut["support"], native, np.nan)
                ax.plot(cut["x"], native, color=color, ls=":", lw=1.5, zorder=1)
        ax.set_xlabel("radius [m]")
        ax.set_ylabel("current density [MA/m²]")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.tick_params(width=1.2, labelsize=14)
        ax.text(0.5, 1.05, name, transform=ax.transAxes, ha="center", fontsize=16)
    handles = [
        Line2D([], [], color=SOLVED, lw=3, label="solved source"),
        Line2D(
            [], [], color=REFERENCE, ls="--", lw=2.6, label="EFIT-coordinate source"
        ),
        Line2D([], [], color=SOLVED, ls=":", lw=1.5, label="solved map"),
        Line2D([], [], color=REFERENCE, ls=":", lw=1.5, label="EFIT map"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    path = output / (record["case"] + "-midplane.svg")
    save(figure, axes, [], path, previews / (path.stem + ".png"))
    plt.close(figure)


def contour_panels(record, entry, geometry, measured, output, previews):
    """Overlay source and Ampere currents, each on common solved/EFIT levels."""
    bank, _ = read_case(entry)
    states = {
        stage: dict(np.load(measured / f"{record['case']}-{stage}.npz"))
        for stage in ("reference", "terminal")
    }
    levels = np.array([0.1, 0.3, 0.6, 1.0, 1.5, 2.0])
    figure, axes = plt.subplots(1, 2, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.13, top=0.81, wspace=0.20)
    solved_style = replace(
        DEFAULT_INK, axis_color=SOLVED, xpoint_color=SOLVED, zorder_markers=10
    )
    reference_style = replace(
        DEFAULT_INK,
        axis_color=REFERENCE,
        xpoint_color=REFERENCE,
        axis_marker="v",
        xpoint_marker="P",
        zorder_markers=20,
    )
    notes = []
    for ax, key, name in zip(
        axes,
        ("density", "map_density"),
        ("prescribed source current", "saved-map Ampere current"),
        strict=True,
    ):
        poloidal_axes(ax)
        for stage, color, linestyle in [
            ("reference", REFERENCE, "--"),
            ("terminal", SOLVED, "-"),
        ]:
            data = states[stage]
            values = np.asarray(data[key]) / 1e6
            mask = np.asarray(data["support"], bool)
            values = np.where(mask, values, np.nan)
            contours = draw_flux_contours(
                ax,
                data["radius"],
                data["height"],
                values.T,
                levels,
                color=color,
                linewidth=2.0,
                wall=bank["limiter"],
                zorder=3 if stage == "terminal" else 2,
            )
            contours.set_linestyle(linestyle)
        draw_wall(ax, units=bank["limiter"], color=WALL, linewidth=1.5, zorder=5)
        draw_critical_points(
            ax, geometry["solved_core"], np.zeros(2), solved_style, "o", 6
        )
        draw_critical_points(
            ax, geometry["reference_core"], np.zeros(2), reference_style, "s", 6
        )
        draw_nulls(
            ax, x_points=np.asarray(geometry["saddle_rz_m"])[None], style=solved_style
        )
        draw_nulls(
            ax,
            x_points=np.asarray(geometry["reference_saddle_rz_m"])[None],
            style=reference_style,
        )
        ax.set_xlim(bank["radius"][0], bank["radius"][-1])
        ax.set_ylim(bank["height"][0], bank["height"][-1])
        box = ax.get_position()
        notes.append(
            figure.text((box.x0 + box.x1) / 2, 0.85, name, ha="center", fontsize=16)
        )
    handles = [
        Line2D([], [], color=SOLVED, lw=3, label="solved"),
        Line2D([], [], color=REFERENCE, ls="--", lw=2.6, label="EFIT"),
        Line2D([], [], color=WALL, lw=1.5, label="wall"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    notes.append(
        figure.text(
            0.5,
            0.065,
            "current-density levels [MA/m²]: 0.1, 0.3, 0.6, 1.0, 1.5, 2.0",
            ha="center",
            fontsize=14,
        )
    )
    path = output / (record["case"] + "-poloidal.svg")
    save(figure, axes, notes, path, previews / (path.stem + ".png"))
    plt.close(figure)


def profile_panels(records, output, previews):
    """Expose the central source hole before any topology or amplitude read."""
    figure, axes = plt.subplots(1, 3, figsize=CANVAS, dpi=100)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.80, wspace=0.38)
    from nova.equilibrium.source import DomainProfile
    from nova.equilibrium.solve_request import SampledFluxFunction
    from scripts.prototypes.diiid_hollow_current.trace import split_current

    for ax, record in zip(axes, records, strict=True):
        closures = {
            name: SampledFluxFunction(data["coordinate"], data["values"])
            for name, data in record["profiles"].items()
        }
        closure = DomainProfile(**closures)
        coordinate = np.linspace(0, 0.4, 401)
        row = split_current(
            1.7 if "frame" in record["case"] else 0.9, coordinate, closure
        )
        ax.plot(
            coordinate, np.asarray(row["unscaled_density"]) / 1e6, color=SOLVED, lw=3
        )
        ax.axvline(
            record["profiles"]["p_prime"]["coordinate"][0], color=WALL, ls=":", lw=1.2
        )
        ax.set_xlabel("normalised flux")
        ax.set_ylabel("source current density [MA/m²]")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.text(
            0.5,
            1.05,
            record["case"].replace("frame-", "").replace("-diverted", "D"),
            transform=ax.transAxes,
            ha="center",
            fontsize=16,
        )
    handles = [
        Line2D([], [], color=SOLVED, lw=3, label="closure at fixed radius"),
        Line2D([], [], color=WALL, ls=":", lw=1.2, label="first profile knot"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    path = output / "source-axis-cap.svg"
    save(figure, axes, [], path, previews / (path.stem + ".png"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    plt.style.use("data-ink")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})
    entries = json.loads((HERE.parent / "diiid_xpoint_offset/inputs.json").read_text())
    geometry = json.loads(
        (HERE.parent / "diiid_xpoint_offset/results.json").read_text()
    )["rows"]
    records = []
    for case in ("frame-137-diverted", "frame-146-diverted", "mast-21978-35"):
        record = json.loads((args.measurement / (case + ".json")).read_text())
        records.append(record)
        line_panels(record, args.output, args.previews)
        if case.startswith("frame"):
            entry = next(
                row for row in entries if row["case"] == case and row["arm"] == "head"
            )
            row = next(
                row for row in geometry if row["case"] == case and row["arm"] == "head"
            )
            contour_panels(
                record, entry, row, args.measurement, args.output, args.previews
            )
    profile_panels(records, args.output, args.previews)


if __name__ == "__main__":
    main()
