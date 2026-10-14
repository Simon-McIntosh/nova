"""Compare section-summed ITER coil linkage with the prior nova export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


POWER_MATRIX = Path(
    "/home/ITER/mcintos/Code/power_convertor/data/iter/passive_structure/matrices.json"
)


def compare(receipt: dict) -> dict:
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
    return {
        "power_self_max_fraction": power_self,
        "power_mutual_max_fraction": power_mutual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.receipt.read_text()))
    args.output.write_text(json.dumps(result, indent=2))
    print("SUMMARY " + " ".join(f"{key}={value:.6g}" for key, value in result.items()))


if __name__ == "__main__":
    main()
