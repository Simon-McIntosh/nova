"""Compare section-summed ITER coil linkage with independent matrix records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imas
import numpy as np


POWER_MATRIX = Path(
    "/home/ITER/mcintos/Code/power_convertor/data/iter/passive_structure/matrices.json"
)
SCENARIOS = Path("/work/imas/shared/imasdb/ITER/3")
DRIVES = (
    "CS3U",
    "CS2U",
    "CS1",
    "CS2L",
    "CS3L",
    "PF1",
    "PF2",
    "PF3",
    "PF4",
    "PF5",
    "PF6",
)
TURNS = np.array([554.0] * 5 + [248.6, 115.2, 185.9, 169.9, 216.8, 459.4])


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
    result = {
        "power_self_max_fraction": power_self,
        "power_mutual_max_fraction": power_mutual,
    }

    wiring = np.zeros((len(names), len(DRIVES)))
    for index, name in enumerate(DRIVES):
        if name == "CS1":
            wiring[names.index("CS1U"), index] = 1
            wiring[names.index("CS1L"), index] = 1
        else:
            wiring[names.index(name), index] = 1
    driven = wiring.T @ matrix @ wiring
    for pulse in (105013, 105014):
        with imas.DBEntry(
            f"imas:hdf5?path={SCENARIOS / str(pulse) / '1'}", "r", dd_version="3.38.1"
        ) as entry:
            ids = entry.get("em_coupling", autoconvert=False)
        source_matrix = np.asarray(ids.mutual_active_active)
        if source_matrix.shape != (15, 15):
            raise ValueError("DINA active coupling has an unexpected shape")
        dina = source_matrix[:11, :11] * TURNS[:, None] * TURNS[None, :]
        result[f"dina_{pulse}_self_max_fraction"] = float(
            np.max(np.abs(np.diag(driven) / np.diag(dina) - 1))
        )
        off = ~np.eye(11, dtype=bool)
        result[f"dina_{pulse}_mutual_max_fraction"] = float(
            np.max(np.abs(driven[off] / dina[off] - 1))
        )
    return result


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
