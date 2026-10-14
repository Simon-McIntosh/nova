"""Measure how passive thickness refinement changes every coil terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def spectrum(receipt: dict) -> np.ndarray:
    return np.asarray(receipt["screened_real_ohm"]) + 1j * np.asarray(
        receipt["screened_imag_ohm"]
    )


def compare(coarse: dict, refined: dict) -> dict:
    if coarse["coil_names"] != refined["coil_names"]:
        raise ValueError("coil ordering differs between meshes")
    frequency = np.asarray(refined["frequency_hz"])
    if not np.array_equal(np.asarray(coarse["frequency_hz"]), frequency):
        raise ValueError("frequency samples differ between meshes")
    first, second = spectrum(coarse), spectrum(refined)
    error = np.abs(first - second) / np.maximum(np.abs(second), 1e-30)
    worst = np.max(error, axis=1)
    passed = np.cumprod((worst < 0.01).astype(int)).astype(bool)
    limit = float(frequency[np.flatnonzero(passed)[-1]]) if np.any(passed) else 0.0
    return {
        "coarse_passive_count": coarse["passive_count"],
        "refined_passive_count": refined["passive_count"],
        "coarse_max_layer_m": coarse["max_layer_thickness_m"],
        "refined_max_layer_m": refined["max_layer_thickness_m"],
        "sample_count": len(frequency),
        "upper_sampled_limit_hz": limit,
        "max_relative_difference_at_600_hz": float(
            worst[np.argmin(abs(frequency - 600))]
        ),
        "max_relative_difference_at_5000_hz": float(worst[-1]),
        "max_relative_difference": float(np.max(worst)),
        "coil_max_relative_difference": {
            name: float(value)
            for name, value in zip(
                refined["coil_names"], np.max(error, axis=0), strict=True
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.coarse.read_text()), json.loads(args.refined.read_text())
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(
        "SUMMARY upper_sampled_limit_hz={upper_sampled_limit_hz:.6g} "
        "max_relative_difference={max_relative_difference:.6g} "
        "at_600_hz={max_relative_difference_at_600_hz:.6g} "
        "at_5000_hz={max_relative_difference_at_5000_hz:.6g}".format(**result)
    )


if __name__ == "__main__":
    main()
