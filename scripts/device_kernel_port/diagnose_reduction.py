"""Localise cross-backend flux discrepancies to packed source columns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORK = HERE / "work"


def _digest_equal(left: np.ndarray, right: np.ndarray) -> float:
    """Return the element-wise byte-identical fraction for two fp64 blocks."""
    return float(np.count_nonzero(left.view(np.uint64) == right.view(np.uint64))) / (
        left.size
    )


def main() -> None:
    """Report which packed source owns scale-aware cross-backend failures."""
    data = np.load(WORK / "coarse-input.npz")
    cpu = np.load(WORK / "coarse-jax-cpu.npy")
    gpu = np.load(WORK / "coarse-jax-gpu.npy")
    reference = np.load(WORK / "coarse-numpy-reference.npy")
    absolute = np.abs(gpu - cpu)
    tolerance = 1.0e-9 * float(np.max(np.abs(reference)))
    failed = absolute > tolerance
    source_counts = np.count_nonzero(failed, axis=0)
    worst_source = int(np.argmax(source_counts))
    present = ~np.signbit(data["weight"][:, worst_source])
    vertices = data["edge"][present, :2, worst_source]
    payload = {
        "byte_identical_fraction": _digest_equal(gpu, cpu),
        "failed_element_count": int(np.count_nonzero(failed)),
        "failure_tolerance": tolerance,
        "mechanism": (
            "device-level differences inside each closed-form edge are amplified "
            "by the inverse area normalization of one vanishing-area section"
        ),
        "nonzero_failure_source_count": int(np.count_nonzero(source_counts)),
        "worst_source": {
            "column": worst_source,
            "failed_element_count": int(source_counts[worst_source]),
            "gpu_cpu_max_absolute_difference": float(np.max(absolute[:, worst_source])),
            "norm": float(data["norm"][worst_source]),
            "vertex_count": int(len(vertices)),
            "vertices": vertices.tolist(),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
