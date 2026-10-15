"""Draw measured lean-map walls, stage costs, and terminal poloidal contours."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import solovev_certificate as certificate
from scripts.prototypes.fast_read.render_contours import panel
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
from scripts.analytic_oracle_fixtures import measure as fixture


COLORS = {"limited": "#356c86", "diverted": "#a05d49"}


def style(axis, xlabel, ylabel, *, log=False):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    axis.set_xlabel(xlabel, fontsize=22)
    axis.set_ylabel(ylabel, fontsize=22)
    axis.tick_params(labelsize=20)
    if log:
        axis.set_yscale("log")


def save(figure, path):
    figure.savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(figure)


def curves(rows, output, key, ylabel, target=None):
    figure, axis = plt.subplots(figsize=(14, 6), dpi=100)
    style(axis, "plasma cells", ylabel, log=True)
    for kind, color in COLORS.items():
        selected = sorted(
            (row for row in rows if row["kind"] == kind and key in row),
            key=lambda row: row["cells"],
        )
        if not selected:
            continue
        x = [row["cells"] for row in selected]
        y = [row[key] for row in selected]
        axis.plot(x, y, color=color, linewidth=3, marker="o")
        axis.annotate(
            kind,
            (x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=20,
        )
    if target is not None:
        axis.axhline(target, color="#666666", linestyle=":", linewidth=1.2)
        axis.annotate(
            f"{target:g}",
            (min(row["cells"] for row in rows), target),
            xytext=(0, 5),
            textcoords="offset points",
            color="#666666",
            fontsize=20,
        )
    axis.margins(x=0.15, y=0.15)
    save(figure, output)


def batch_walls(rows, output):
    figure, axis = plt.subplots(figsize=(14, 6), dpi=100)
    style(axis, "states in batch", "per-state wall [ms]", log=True)
    for kind, color in COLORS.items():
        for cells, marker in ((289, "o"), (550, "s"), (2098, "^")):
            selected = [
                row
                for row in rows
                if row["kind"] == kind
                and abs(row["cells"] - cells) < max(25, cells * 0.1)
            ]
            if not selected:
                continue
            row = selected[0]
            widths = [
                width
                for width in (1, 16, 128)
                if f"batch_{width}" in row and "chord" in row[f"batch_{width}"]
            ]
            if not widths:
                continue
            walls = [row[f"batch_{width}"]["chord"]["per_state_ms"] for width in widths]
            axis.plot(widths, walls, color=color, linewidth=3, marker=marker)
            axis.annotate(
                f"{kind} {row['cells']}",
                (widths[-1], walls[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                color=color,
                fontsize=20,
            )
    axis.axhline(0.90, color="#666666", linestyle=":", linewidth=1.2)
    axis.annotate(
        "0.90",
        (1, 0.90),
        xytext=(0, 5),
        textcoords="offset points",
        color="#666666",
        fontsize=20,
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks((1, 16, 128), labels=("1", "16", "128"))
    axis.margins(x=0.15, y=0.15)
    save(figure, output)


def stages(receipts, output):
    labels = {
        "support_read": "connected support",
        "exact_cut": "exact cut booking",
        "values_matvec": "Gauss values",
        "band_matvec": "band jets",
        "root_solve": "fixed points",
        "interior_gauss": "interior Gauss",
    }
    figure, axis = plt.subplots(figsize=(14, 6), dpi=100)
    style(axis, "stage wall [ms]", "")
    ranked = sorted(
        ((label, receipts["walls_ms"][key]) for key, label in labels.items()),
        key=lambda row: row[1],
    )
    axis.barh([row[0] for row in ranked], [row[1] for row in ranked], color="#356c86")
    axis.tick_params(axis="y", labelsize=20)
    for index, (_, wall) in enumerate(ranked):
        axis.text(wall, index, f" {wall:.3g}", va="center", fontsize=20)
    axis.margins(x=0.12)
    save(figure, output)


def terminal_panel(row, cache, output):
    kind = row["kind"]
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, _ = _machine(cache, carrier, row["requested_cells"])
    assert len(machine.node) == row["cells"]
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = np.asarray(certificate._exact_state(name, exact, coordinates))
    moments = jnp.asarray(np.load(row["_receipt_path"].with_suffix(".npy")))
    exterior, _ = _exterior(cache, key)
    operator = fixture.forward_operator(source, machine)
    solved = np.asarray(exterior + _image(operator, moments))
    payload = {
        "coordinates": np.asarray(machine.node).tolist(),
        "wall": np.asarray(machine.wall_node).tolist(),
        "solved_flux": solved[: len(machine.node)].tolist(),
        "analytic_flux": analytic[: len(machine.node)].tolist(),
        "solved_roots": row["terminal_roots"],
        "analytic_roots": [],
    }
    from scripts.prototypes.support.geometry import certificate_field
    from tests.equilibrium.test_topology_read import _analytic_inputs
    from nova.equilibrium import topology
    from nova.equilibrium.solve_request import TopologyPolicy
    import jax

    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic_field, kind)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    payload["analytic_roots"] = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    ).tolist()
    panel(payload, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.receipts.glob("*-*.json")):
        row = json.loads(path.read_text())
        if isinstance(row, dict) and "analytic_map_error" in row:
            row["_receipt_path"] = path
            rows.append(row)
    if not rows:
        raise RuntimeError("no completed analytic-map receipt found")
    curves(
        rows,
        args.output / "map-error.svg",
        "analytic_map_error",
        "relative sup map error",
        1e-2,
    )
    curves(rows, args.output / "map-wall.svg", "map_wall_ms", "map wall [ms]")
    batch_walls(rows, args.output / "batch-wall.svg")
    for row in rows:
        if (
            "terminal_roots" in row
            and row["_receipt_path"].with_suffix(".npy").is_file()
        ):
            build = row["_receipt_path"].stem.split("-")[0]
            cache = args.receipts / f"cache-{build}"
            terminal_panel(
                row,
                cache if cache.is_dir() else args.cache,
                args.output / f"{row['_receipt_path'].stem}-terminal.svg",
            )
    for path in args.receipts.glob("stages-*.json"):
        stages(json.loads(path.read_text()), args.output / f"{path.stem}.svg")


if __name__ == "__main__":
    main()
