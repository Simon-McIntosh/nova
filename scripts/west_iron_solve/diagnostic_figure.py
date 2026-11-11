"""Map residual-dominant iron cells and the unchanged Newton continuation."""

import argparse
import json
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from nova.imas.machine import StaticMachineDescription  # noqa: E402
from scripts.west_iron_solve.figure import IRON, draw_machine, save_figure  # noqa: E402
from scripts.west_iron_solve.measure import MACHINE, save  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    diagnostic = json.loads(args.receipt.read_text())
    saved = Path(diagnostic["saved"])
    record = json.loads((saved / "receipt.json").read_text())
    with (saved / "solution.pkl").open("rb") as stream:
        solution = pickle.load(stream)
    with Path(record["source"]).open("rb") as stream:
        source = pickle.load(stream)
    machine = StaticMachineDescription.read_iron_core(
        f"imas:hdf5?path={MACHINE}", dd_version="4.1.0"
    )
    segments = [
        shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
        for s in machine.iron_segments
    ]
    assert len(segments) == 6
    table = []
    for raw in diagnostic["top_cells"]:
        row = dict(raw)
        polygon = shapely.Polygon(solution._mesh.polygons[row["cell"]])
        overlaps = [polygon.intersection(region).area for region in segments]
        index = int(np.argmax(overlaps))
        assert overlaps[index] > 0
        row.update(
            segment_index=index,
            segment=f"iron_core.segment[{index}]",
            segment_bounds_m=list(segments[index].bounds),
            cell_bounds_m=list(polygon.bounds),
        )
        table.append(row)
        print("CELL " + json.dumps(row), flush=True)
    polygons = [shapely.Polygon(p) for p in solution._mesh.polygons]
    clearance = np.array([p.centroid.distance(p.boundary) for p in polygons])
    enriched = dict(
        rows=table,
        population_clearance_min_m=float(clearance.min()),
        population_clearance_median_m=float(np.median(clearance)),
        top_three_residual_fraction=sum(x["residual_fraction"] for x in table[:3]),
        top_twenty_residual_fraction=diagnostic["top_twenty_residual_fraction"],
        state_norm_A_per_m=float(np.linalg.norm(solution.magnetisation)),
        authored_segment_names=[s.name for s in machine.iron_segments],
    )
    save(args.receipt.parent / "cell-table.json", enriched)
    plt.style.use("data-ink")
    plt.rcParams.update({"svg.fonttype": "none"})
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    geometry = fig.add_axes([0.02, 0.04, 0.43, 0.87])
    geometry.add_collection(
        PolyCollection(
            solution._mesh.polygons,
            facecolors="none",
            edgecolors="#929292",
            linewidths=0.3,
            zorder=1,
        )
    )
    draw_machine(geometry, source)
    geometry.add_collection(
        PolyCollection(
            [solution._mesh.polygons[x["cell"]] for x in table],
            facecolors="none",
            edgecolors=IRON,
            linewidths=1.8,
            zorder=6,
        )
    )
    fig.legend(
        handles=[
            Line2D([], [], color="#929292", linewidth=2.6, label="iron cells"),
            Line2D([], [], color=IRON, linewidth=3.0, label="largest residuals"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.05, 1.0),
        ncol=2,
    )
    trace = fig.add_axes([0.59, 0.44, 0.38, 0.45])
    history = record["residual_history"] + [
        x["relative_residual"] for x in diagnostic["history"][1:]
    ]
    trace.plot(
        np.arange(len(history)),
        history,
        color=IRON,
        linewidth=3.0,
        marker="o",
        markersize=4,
        zorder=3,
    )
    trace.set_yscale("log")
    trace.set_ylim(3e-4, 1.3)
    trace.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    trace.set_ylabel("relative material residual")
    trace.tick_params(labelbottom=False)
    steps = fig.add_axes([0.59, 0.15, 0.38, 0.2], sharex=trace)
    scales = record["step_scales"] + [
        x["step_scale"] for x in diagnostic["history"] if x.get("step_scale", 0) > 0
    ]
    steps.step(
        np.arange(1, len(scales) + 1),
        scales,
        where="mid",
        color="#929292",
        linewidth=2.6,
        zorder=3,
    )
    steps.set_yscale("log")
    steps.set_ylim(1e-3, 1.4)
    steps.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    steps.set_ylabel("step scale")
    steps.set_xlabel("accepted Newton corrections")
    steps.set_xlim(-0.3, len(history) - 0.5)
    labels = save_figure(fig, args.output, args.preview)
    print(
        "FIGURE "
        + json.dumps(
            dict(
                output=str(args.output),
                preview=str(args.preview),
                text_artists=labels,
                highlighted_cells=len(table),
            )
        )
    )
    print("GEOMETRY " + json.dumps({k: v for k, v in enriched.items() if k != "rows"}))


if __name__ == "__main__":
    main()
