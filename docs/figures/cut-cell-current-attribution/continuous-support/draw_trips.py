"""Draw captured trip fields, null sets, and boundary-band current glyphs."""
# ruff: noqa: E501 -- Captions and persisted HTML retain their literal text.

import argparse
import json
import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from nova.jax.config import configure_dtypes

configure_dtypes()
from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.media import poloidal  # noqa: E402
from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402

ROOT = Path(os.environ.get("CONTINUOUS_EVIDENCE_ROOT", Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    args = parser.parse_args()
    source = ROOT / "record" / f"cells-{abs(args.cells)}" / "receipt.json"
    if not source.exists():
        print("NO_RECEIPT_TO_RENDER", flush=True)
        return
    payload = json.loads(source.read_text())
    if "raw_receipt" in payload:
        source = Path(payload["raw_receipt"])
        payload = json.loads(source.read_text())
    entries = list(payload["trips"])
    for name in ("analytic_input", "analytic_mapped"):
        if name in payload:
            entries.append(payload[name])
    coordinates = np.c_[payload["coordinates_r_m"], payload["coordinates_z_m"]]
    wall = np.c_[payload["wall_r_m"], payload["wall_z_m"]]
    radial, height, analytic = certificate._raster_field(
        coordinates, np.asarray(payload["analytic_state"]), wall
    )
    reference = payload.get("analytic_input", {}).get("record", {})
    levels = poloidal.contour_levels(
        analytic,
        12,
        axis=reference.get("axis_flux_wb"),
        boundary=reference.get("boundary_flux_wb"),
    )
    scale = max(
        (
            abs(value)
            for entry in entries
            for value in entry["record"]["cells"]["cell_current_a"]
        ),
        default=1,
    )
    written = []
    for entry in entries:
        record = entry["record"]
        cells = record["cells"]
        figure, ax = plt.subplots(figsize=(9.0, 9.5), constrained_layout=True)
        _, _, solved = certificate._raster_field(
            coordinates, np.asarray(entry["flux"]), wall
        )
        for field, color in (
            (analytic, certificate.ANALYTIC_INK_COLOR),
            (solved, certificate.SOLVED_INK_COLOR),
        ):
            poloidal.draw_flux_contours(ax, radial, height, field, levels, color=color)
        units = (wall,)
        poloidal.draw_wall(ax, units=units)
        poloidal.draw_nulls(
            ax,
            magnetic_axis=payload["analytic_axis_rz_m"],
            x_points=payload["analytic_x_points_rz_m"],
            style=DEFAULT_INK.variant(
                axis_color=certificate.ANALYTIC_INK_COLOR,
                xpoint_color=certificate.ANALYTIC_INK_COLOR,
                axis_marker="^",
                axis_markersize=7,
                xpoint_marker="X",
            ),
            contain=units,
        )
        axis = record.get("read_axis_rz_m")
        tally = poloidal.draw_nulls(
            ax,
            magnetic_axis=axis[0] if axis else None,
            x_points=record.get("read_x_point_rz_m"),
            style=DEFAULT_INK.variant(
                axis_color=certificate.SOLVED_INK_COLOR,
                xpoint_color=certificate.SOLVED_INK_COLOR,
                axis_marker="^",
                axis_markersize=4,
                xpoint_marker="X",
            ),
            contain=units,
        )
        polygons = payload["cell_polygons"]

        def outlines(indices, color, width):
            segments = []
            for idx in indices:
                p = np.asarray(polygons[idx])
                segments.append(np.vstack((p, p[:1])))
            if segments:
                ax.add_collection(
                    LineCollection(segments, colors=color, linewidths=width, zorder=3)
                )

        band = np.asarray(cells["in_boundary_band"], dtype=bool)
        indices = np.asarray(cells["index"], dtype=int)
        outlines(indices[band], "0.4", 0.8)
        outlines(entry.get("mask_difference_cell_indices", []), "#aa3377", 2)
        outlines(entry.get("current_support_changed_cells", []), "#0099aa", 1.4)
        poloidal_axes(ax)
        ax.set_title(
            f"{payload['realised_cells']} cells · trip {entry['trip']} · residual {entry['residual']:.5g} · converged {entry['converged']}\n"
            f"Analytic blue / solved ochre; shared Wb levels; both axes as triangles; wall black\n"
            f"Axis error {entry['axis_error_m']:.5g} m; amplitude {record['amplitude']:.6g}; no admitted saddles",
            fontsize=10,
        )
        stem = ROOT / "panels" / f"cells-{abs(args.cells)}-trip-{entry['trip']}"
        for extension in ("png", "svg"):
            figure.savefig(str(stem) + "." + extension, dpi=145)
        plt.close(figure)
        written.append(
            {
                "trip": entry["trip"],
                "png": str(stem) + ".png",
                "svg": str(stem) + ".svg",
                "solved_null_tally": tally,
            }
        )
        print("PANEL " + str(stem), flush=True)
    (source.parent / "panels.json").write_text(
        json.dumps(
            {
                "shared_levels_wb": levels.tolist(),
                "current_glyph_scale_a": scale,
                "panels": written,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
