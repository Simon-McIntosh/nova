"""Compare the carrier's X-point wedges against the oracle's independent sectors.

Diagnostic instrument for the wedge oracle: it prints and draws, for one
resolution, the four regions the production spline-chain clip emits and the
four the oracle reconstructs from the exact separatrix edge roots and the cell
boundary.  It exists to localise a mismatch between the two, which the oracle
reports only as a relative moment error.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import benchmarks.xpoint_cell_wedge_oracle as oracle
from benchmarks import topology_read_resolution_ladder as topology_ladder
from nova.equilibrium.separatrix_clip import AtomicCellMesh
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

OUTPUT = Path(
    "/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/"
    "s19-codex-20260916/cca-production-clip-takes-the-typed-saddle/"
    "docs/figures/cut-cell-current-attribution/xpoint-cell"
)


def shoelace(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    radial, vertical = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(
        np.dot(radial, np.roll(vertical, -1)) - np.dot(vertical, np.roll(radial, -1))
    )


def centroid(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    radial, vertical = vertices[:, 0], vertices[:, 1]
    radial_next, vertical_next = np.roll(radial, -1), np.roll(vertical, -1)
    cross = radial * vertical_next - radial_next * vertical
    area = 0.5 * float(cross.sum())
    return np.asarray(
        [
            float(((radial + radial_next) * cross).sum()),
            float(((vertical + vertical_next) * cross).sum()),
        ]
    ) / (6.0 * area)


def _loop(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    return np.vstack((vertices, vertices[0]))


def main():
    cells = 110
    machine, _operator, _analytic = topology_ladder._machine_and_field(cells)
    exact = topology_ladder.ANALYTIC
    x_point = np.asarray(exact.x_point, dtype=np.float64)
    axis = np.asarray(exact.magnetic_axis, dtype=np.float64)
    boundary_flux = float(exact.flux(x_point[None, :])[0])
    polarity = float(
        np.sign(float(exact.flux(axis[None, :])[0]) - boundary_flux)
    )
    cell, _candidates = oracle._xpoint_cell(machine, x_point)
    polygon = np.asarray(machine.cell_polygons[cell], dtype=np.float64)
    centre = np.asarray(machine.node[cell], dtype=np.float64)
    mesh = AtomicCellMesh.from_cells([polygon], centroids=centre[None, :])
    node_flux = np.asarray(exact.flux(mesh.node_coordinates), dtype=np.float64)
    signed_flux = jnp.asarray(polarity * (node_flux - boundary_flux))
    edge_rows = oracle._edge_root_diagnostics(exact, polygon, x_point, boundary_flux)
    fractions, counts, positive_after = oracle._edge_root_arrays(edge_rows, polarity)
    wedges = jax.jit(
        lambda values: mesh.traced_saddle_wedges(
            values,
            saddle_vertex=jnp.asarray(x_point),
            core_reference=jnp.asarray(axis),
            edge_root_fraction=jnp.asarray(fractions),
            edge_root_count=jnp.asarray(counts),
            edge_root_positive_after=jnp.asarray(positive_after),
        )
    )(signed_flux)

    sectors = oracle._independent_branch_sectors(
        polygon, exact, x_point, boundary_flux, polarity
    )
    carrier = []
    for slot in range(4):
        count = int(np.asarray(wedges.vertex_count)[0, slot])
        vertices = np.asarray(oracle._support_vertices(wedges, slot), dtype=np.float64)[
            :count
        ]
        carrier.append(
            {
                "slot": slot,
                "vertices": vertices.tolist(),
                "area_m2": shoelace(vertices),
                "centroid_rz_m": centroid(vertices).tolist(),
            }
        )
    independent = []
    for slot, item in enumerate(sectors):
        vertices = np.asarray(item["vertices"], dtype=np.float64)
        independent.append(
            {
                "slot": slot,
                "confined": bool(item["confined"]),
                "vertices": vertices.tolist(),
                "area_m2": shoelace(vertices),
                "centroid_rz_m": centroid(vertices).tolist(),
            }
        )

    record = {
        "requested_cells": cells,
        "realised_cells": int(len(machine.node)),
        "xpoint_cell": int(cell),
        "saddle_rz_m": x_point.tolist(),
        "axis_rz_m": axis.tolist(),
        "polarity": polarity,
        "cell_polygon_rz_m": polygon.tolist(),
        "carrier_wedges": carrier,
        "independent_sectors": independent,
        "carrier_core_matches_independent_core": (
            oracle._closing_vertex_sequence(carrier[0]["vertices"])
            == oracle._closing_vertex_sequence(
                max(
                    (
                        item
                        for item in independent
                        if item["confined"]
                    ),
                    key=lambda item: float(
                        np.dot(
                            np.asarray(item["centroid_rz_m"]) - x_point,
                            axis - x_point,
                        )
                    ),
                )["vertices"]
            )
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "carrier-vs-independent-cells-110.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), constrained_layout=True)
    wall = np.asarray(machine.wall_node, dtype=np.float64)
    for axis_panel, title in zip(
        axes,
        ("production clip wedges", "independent sector reconstruction"),
        strict=True,
    ):
        poloidal_axes(axis_panel)
        axis_panel.set_title(title, fontsize=9)
        poloidal.draw_wall(axis_panel, units=(wall,), linewidth=0.75)
        poloidal.draw_nulls(
            axis_panel,
            magnetic_axis=axis,
            x_points=x_point[None, :],
            style=DEFAULT_INK,
            contain=(wall,),
        )
        cell_loop = _loop(polygon)
        axis_panel.plot(
            cell_loop[:, 0], cell_loop[:, 1], color="black", linewidth=2.0
        )
    colours = ("tab:red", "tab:purple", "tab:green", "tab:green")
    for slot, item in enumerate(carrier):
        loop = _loop(item["vertices"])
        axes[0].plot(
            loop[:, 0],
            loop[:, 1],
            color=colours[slot],
            linewidth=1.6,
            label=f"slot {slot}",
        )
    for slot, item in enumerate(independent):
        loop = _loop(item["vertices"])
        axes[1].plot(
            loop[:, 0],
            loop[:, 1],
            color="tab:gray",
            linewidth=1.2,
            label="confined" if item["confined"] else "common SOL",
        )
        axes[1].plot(
            loop[0, 0], loop[0, 1], marker="o", markersize=3, color="tab:gray"
        )
    for panel in axes:
        panel.set_xlim(
            float(np.min(polygon[:, 0])) - 0.03, float(np.max(polygon[:, 0])) + 0.03
        )
        panel.set_ylim(
            float(np.min(polygon[:, 1])) - 0.03, float(np.max(polygon[:, 1])) + 0.03
        )
    figure.suptitle(
        "X-point cell at 110 requested cells: the clip's core wedge is the quad "
        "beside the saddle; the reconstruction's confined sector is the quad above it",
        fontsize=8.5,
    )
    figure.savefig(OUTPUT / "carrier-vs-independent-cells-110.png", dpi=160)
    figure.savefig(OUTPUT / "carrier-vs-independent-cells-110.svg")
    plt.close(figure)
    print("CARRIER_CORE_AREA=%.12e" % carrier[0]["area_m2"])
    for item in independent:
        print(
            "SECTOR slot=%d confined=%s area=%.12e centroid=%s"
            % (
                item["slot"],
                item["confined"],
                item["area_m2"],
                np.array2string(np.asarray(item["centroid_rz_m"]), precision=9),
            )
        )
    print(
        "CARRIER_CORE_MATCHES_INDEPENDENT=%s"
        % record["carrier_core_matches_independent_core"]
    )
    print("DIAG-DONE")


if __name__ == "__main__":
    main()