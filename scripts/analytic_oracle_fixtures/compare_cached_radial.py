"""Compare a stored host radial sample block with a fresh device assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

import nova
from nova.biot.polygonanalytic import _section_centroid
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import coupling_build


def compare(cache: Path, group: str, output: Path) -> None:
    configure_dtypes()
    assert jax.config.jax_enable_x64
    print(f"NOVA_FILE={nova.__file__}", flush=True)
    print(f"CWD={Path.cwd()}", flush=True)
    print(f"DEVICES={jax.devices()}", flush=True)
    with xr.open_zarr(cache, group=group, consolidated=False) as data:
        host = np.asarray(data["plasma_to_sample_r"].values)
        targets = np.asarray(data["sample_coordinates"].values)
        nodes = np.asarray(data["node"].values)
        wall = np.asarray(data["wall_node"].values)
        offsets = np.asarray(data["cell_polygon_offsets"].values)
        vertices = np.asarray(data["cell_polygon_vertices"].values)
        identity = json.loads(data.attrs["semantic_identity"])
    polygons = tuple(vertices[offsets[i] : offsets[i + 1]] for i in range(len(nodes)))
    centres = np.asarray([_section_centroid(section) for section in polygons])
    device = coupling_build.build_flux_blocks(targets, polygons, centres)[1]
    difference = np.abs(host - device)
    global_scale = float(max(np.abs(host).max(), np.abs(device).max()))
    candidates = np.argpartition(difference.ravel(), -20)[-20:]
    candidates = candidates[np.argsort(difference.ravel()[candidates])[::-1]]
    row, column = np.unravel_index(candidates, difference.shape)
    wall_distance = cKDTree(wall).query(nodes)[0]
    top = [
        {
            "target": int(r),
            "cell": int(c),
            "target_coordinate": targets[r].tolist(),
            "cell_coordinate": nodes[c].tolist(),
            "source_wall_distance": float(wall_distance[c]),
            "target_source_distance": float(np.linalg.norm(targets[r] - nodes[c])),
            "host": float(host[r, c]),
            "device": float(device[r, c]),
            "absolute_difference": float(difference[r, c]),
            "norm_relative": float(difference[r, c] / global_scale),
        }
        for r, c in zip(row, column, strict=True)
    ]
    result = {
        "cache": str(cache),
        "group": group,
        "host_identity": identity,
        "shape": list(host.shape),
        "global_scale": global_scale,
        "worst_relative": top[0]["norm_relative"],
        "top": top,
        "completed": True,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"RADIAL_WORST={top[0]['norm_relative']:.16e}", flush=True)
    print(json.dumps(top[0]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("group")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compare(args.cache, args.group, args.output)


if __name__ == "__main__":
    main()
