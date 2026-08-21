"""Re-evaluate one packed contour through the closed form in extended precision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova.biot.polygon import MU0
from nova.biot.polygonanalytic import _Edge, _Vertex


HERE = Path(__file__).resolve().parent
WORK = HERE / "work"


def _extended_flux(
    target_r: np.ndarray, target_z: np.ndarray, vertices: np.ndarray, *, nodes: int
) -> np.ndarray:
    """Evaluate the scalar host contour order with long-double antiderivatives."""
    dtype = np.longdouble
    section = np.asarray(vertices, dtype=dtype)
    local = section - section[0]
    following = np.roll(local, -1, axis=0)
    signed_twice_area = np.sum(
        local[:, 0] * following[:, 1] - following[:, 0] * local[:, 1]
    )
    sign = -np.sign(signed_twice_area)
    area = dtype(0.5) * np.abs(signed_twice_area)
    norm = sign * dtype(str(MU0)) / area
    rolled = np.roll(section, -1, axis=0)
    edges = np.column_stack([section[:, 0], section[:, 1], rolled[:, 0], rolled[:, 1]])

    radius = np.abs(np.asarray(target_r, dtype=dtype)).ravel()
    height = np.asarray(target_z, dtype=dtype).ravel()
    flux = np.zeros_like(radius)
    corners: dict[int, _Vertex] = {}

    def corner(index: int) -> _Vertex:
        if index not in corners:
            corners[index] = _Vertex(
                radius,
                height,
                section[index, 0],
                section[index, 1],
                nodes,
                residual=False,
                xp=np,
            )
        return corners[index]

    for index, edge in enumerate(edges):
        lower = corner(index)
        upper = corner((index + 1) % len(edges))
        part = _Edge(radius, height, edge, nodes, xp=np)
        flux -= part.terms(upper)[0] - part.terms(lower)[0]
    return dtype(0.5) * norm * radius * flux


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, default=184)
    parser.add_argument("--nodes", type=int, default=128)
    args = parser.parse_args()

    data = np.load(WORK / "coarse-input.npz")
    present = ~np.signbit(data["weight"][:, args.source])
    vertices = data["edge"][present, :2, args.source]
    oracle = _extended_flux(
        data["target_r"], data["target_z"], vertices, nodes=args.nodes
    )
    old = np.load(WORK / "coarse-numpy-reference.npy")[:, args.source]
    cpu = np.load(WORK / "coarse-jax-cpu.npy")[:, args.source]
    gpu = np.load(WORK / "coarse-jax-gpu.npy")[:, args.source]

    def deviation(values: np.ndarray) -> dict[str, float]:
        difference = np.abs(np.asarray(values, dtype=np.longdouble) - oracle)
        scale = np.maximum(np.abs(oracle), np.finfo(np.longdouble).tiny)
        return {
            "max_absolute": float(np.max(difference)),
            "max_relative": float(np.max(difference / scale)),
            "median_absolute": float(np.median(difference)),
        }

    precision = np.finfo(np.longdouble)
    receipt = {
        "arithmetic": "numpy.longdouble per-edge closed-form antiderivatives",
        "fixed_residual_nodes": args.nodes,
        "oracle_values": [
            np.format_float_scientific(value, unique=True) for value in oracle
        ],
        "precision": {
            "epsilon": float(precision.eps),
            "mantissa_bits": int(-precision.machep),
        },
        "source_column": args.source,
        "vertex_count": int(len(vertices)),
        "deviation": {
            "production_numpy": deviation(old),
            "single_source_cpu": deviation(cpu),
            "single_source_gpu": deviation(gpu),
        },
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
