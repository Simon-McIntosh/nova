"""Compare section-summed ITER coil linkage with the prior nova export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova.circuit.conductor import ConductorSet
from nova.circuit.linkage import circuit_linkage_matrix
from scripts.iter_sectioned_conductor_set.build import (
    SOURCE,
    read_ids,
    unsectioned_coil_conductors,
)


POWER_MATRIX = Path(
    "/home/ITER/mcintos/Code/power_convertor/data/iter/passive_structure/matrices.json"
)


def _analytic_inductance_errors() -> tuple[float, float]:
    """Check the two thin-section limits with resolved observer quadrature."""
    errors = []
    for major, width, turns, radius, constant in (
        (1.5, 0.19, 1.0, np.sqrt(0.19**2 / np.pi), -1.75),
        (1.0, 0.05, 12.0, 0.447049 * 0.05, -2.0),
    ):
        conductor = ConductorSet(
            r=[major],
            z=[0.0],
            dr=[width],
            dz=[width],
            current_share=[turns],
            circuit=[0],
        )
        measured = circuit_linkage_matrix(
            conductor, [0], section_scale_frac=0.05, section_n_max=24
        )[0, 0]
        expected = (
            4e-7 * np.pi * major * (np.log(8 * major / radius) + constant) * turns**2
        )
        errors.append(float(abs(measured / expected - 1)))
    return tuple(errors)


def compare(receipt: dict, unsectioned_matrix: np.ndarray | None = None) -> dict:
    names = receipt["coil_names"]
    matrix = np.asarray(receipt["coil_inductance_h"])
    source = json.loads(POWER_MATRIX.read_text())
    source_names = [
        name.rsplit("(", 1)[-1].removesuffix(")") for name in source["coils"]
    ]
    if names != source_names:
        raise ValueError("sectioned and source coil ordering differ")
    reference = np.asarray(source["coil_inductance_H"])
    diagonal = np.diag_indices(len(names))
    off_diagonal = ~np.eye(len(names), dtype=bool)
    power_self = float(np.max(np.abs(matrix[diagonal] / reference[diagonal] - 1)))
    power_mutual = float(
        np.max(np.abs(matrix[off_diagonal] / reference[off_diagonal] - 1))
    )
    if unsectioned_matrix is None:
        active = read_ids(*SOURCE[0])
        conductors = unsectioned_coil_conductors(active)
        unsectioned_matrix = circuit_linkage_matrix(
            conductors,
            range(len(names)),
            section_scale_frac=0.05,
            section_n_max=24,
            finite_source=True,
        )
    unsectioned = np.asarray(unsectioned_matrix)
    if unsectioned.shape != matrix.shape:
        raise ValueError("unsectioned coil matrix shape differs")
    section_self = np.abs(matrix[diagonal] / unsectioned[diagonal] - 1)
    ring_error, rectangle_error = _analytic_inductance_errors()
    result = {
        "power_self_max_fraction": power_self,
        "power_mutual_max_fraction": power_mutual,
        "power_self_fraction_by_coil": {
            name: float(value)
            for name, value in zip(
                names, np.abs(matrix[diagonal] / reference[diagonal] - 1), strict=True
            )
        },
        "sectioning_self_max_fraction": float(np.max(section_self)),
        "sectioning_self_fraction_by_coil": {
            name: float(value) for name, value in zip(names, section_self, strict=True)
        },
        "thin_ring_error_fraction": ring_error,
        "rectangular_coil_error_fraction": rectangle_error,
    }
    if result["sectioning_self_max_fraction"] > 0.005:
        raise AssertionError(f"sectioned self-inductance exceeds 0.5%: {result}")
    if ring_error > 0.05 or rectangle_error > 0.01:
        raise AssertionError(f"analytic inductance exceeds tolerance: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.receipt.read_text()))
    args.output.write_text(json.dumps(result, indent=2))
    print(
        "SUMMARY "
        + " ".join(
            f"{key}={value:.6g}"
            for key, value in result.items()
            if isinstance(value, float)
        )
    )


if __name__ == "__main__":
    main()
