"""Inspect high-magnitude radial sample couplings on a stored carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from nova.biot.polygonanalytic import _section_centroid
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import coupling_build


def inspect(
    cache: Path, group: str, output: Path, figure: Path, *, all_cells: bool
) -> None:
    configure_dtypes()
    import jax

    assert jax.config.jax_enable_x64
    with xr.open_zarr(cache, group=group, consolidated=False) as data:
        host = np.asarray(data["plasma_to_sample_r"].values)
        coordinates = np.asarray(data["sample_coordinates"].values)
        nodes = np.asarray(data["node"].values)
        wall = np.asarray(data["wall_node"].values)
        sampling = np.asarray(data["sampling_vertices"].values)
        offsets = np.asarray(data["cell_polygon_offsets"].values)
        vertices = np.asarray(data["cell_polygon_vertices"].values)
    magnitude = np.abs(host)
    global_scale = float(magnitude.max())
    tree = cKDTree(coordinates)
    distance, sample_index = tree.query(sampling.reshape(-1, 2))
    if float(distance.max()) > 1e-9:
        raise ValueError("stored sampling vertices do not map to sample coordinates")
    sample_index = sample_index.reshape(sampling.shape[:2])
    wall_distance = cKDTree(wall).query(nodes)[0]
    order = np.argsort(wall_distance)
    selected = (
        np.arange(len(nodes))
        if all_cells
        else np.concatenate(
            (
                order[:256],
                order[len(order) // 2 - 64 : len(order) // 2 + 64],
                order[-128:],
            )
        )
    )
    top: list[dict[str, object]] = []
    sampled_pair_count = 0
    for start in range(0, len(selected), 64):
        cells = selected[start : start + 64]
        targets = np.unique(sample_index[cells].ravel())
        padded = np.resize(targets, ((len(targets) + 63) // 64) * 64)
        polygons = tuple(vertices[offsets[i] : offsets[i + 1]] for i in cells)
        centres = np.asarray([_section_centroid(section) for section in polygons])
        device = coupling_build.build_flux_blocks(
            coordinates[padded], polygons, centres, tile_side=64
        )[1][: len(targets)]
        reference = host[np.ix_(targets, cells)]
        disagreement = np.abs(reference - device)
        sampled_pair_count += disagreement.size
        count = min(20, disagreement.size)
        ranking = np.argpartition(disagreement.ravel(), -count)[-count:]
        rows, columns = np.unravel_index(ranking, disagreement.shape)
        top.extend(
            {
                "target": int(targets[row]),
                "cell": int(cells[column]),
                "target_coordinate": coordinates[targets[row]].tolist(),
                "cell_coordinate": nodes[cells[column]].tolist(),
                "host": float(reference[row, column]),
                "device": float(device[row, column]),
                "absolute_difference": float(disagreement[row, column]),
                "norm_relative": float(disagreement[row, column] / global_scale),
                "source_wall_distance": float(wall_distance[cells[column]]),
                "target_source_distance": float(
                    np.linalg.norm(coordinates[targets[row]] - nodes[cells[column]])
                ),
            }
            for row, column in zip(rows, columns, strict=True)
        )
        top.sort(key=lambda item: item["absolute_difference"], reverse=True)
        del top[20:]
        result = {
            "cache": str(cache),
            "group": group,
            "shape": list(host.shape),
            "global_scale": global_scale,
            "candidate_rule": (
                "every cell"
                if all_cells
                else "256 closest to wall vertices, 128 middle, 128 farthest"
            ),
            "selected_cell_count": int(len(selected)),
            "sampled_cell_count": int(start + len(cells)),
            "sampled_pair_count": sampled_pair_count,
            "sampled_worst_relative": top[0]["norm_relative"],
            "completed": bool(start + len(cells) == len(selected)),
            "top": top,
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"SCAN cells={start + len(cells)}/{len(selected)} "
            f"pairs={sampled_pair_count} worst={top[0]['norm_relative']:.6e}",
            flush=True,
        )
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    ax.plot(*wall.T, color="0.35", linewidth=2.4)
    ax.plot(
        [wall[-1, 0], wall[0, 0]],
        [wall[-1, 1], wall[0, 1]],
        color="0.35",
        linewidth=2.4,
    )
    for pair in top[:8]:
        ax.plot(
            [pair["cell_coordinate"][0], pair["target_coordinate"][0]],
            [pair["cell_coordinate"][1], pair["target_coordinate"][1]],
            color="0.7",
            linewidth=1.2,
        )
        ax.plot(*pair["cell_coordinate"], marker="o", color="0.15", markersize=5)
        ax.plot(*pair["target_coordinate"], marker="x", color="#b22424", markersize=8)
    ax.set_aspect("equal")
    ax.set_axis_off()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure)
    plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if key != "top"}))
    print(json.dumps(top[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("group")
    parser.add_argument("output", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("--all-cells", action="store_true")
    arguments = parser.parse_args()
    inspect(
        arguments.cache,
        arguments.group,
        arguments.output,
        arguments.figure,
        all_cells=arguments.all_cells,
    )


if __name__ == "__main__":
    main()
