"""Bind saved iterates to terminal images and inactive constraint unknowns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verify import ROWS


def audit(folder):
    data = dict(np.load(folder / "terminal.npz"))
    receipt = json.loads((folder / "receipt.json").read_text())
    spatial = json.loads((folder / "spatial.json").read_text())
    coordinates = np.load(folder / "spatial-marks.npz")["node_coordinates"]
    size = len(coordinates)
    unknown = -float(receipt["constraint_residual"][0])
    difference = data["mapped"] - data["physical_image"]
    expected = np.where(
        data["shadow"], 0, -2 * np.pi * coordinates[:, 0] * coordinates[:, 1] * unknown
    )
    error = float(np.max(np.abs(difference - expected)))
    assert error < 1e-12, "inactive radial-field image must match its physical basis"
    seed = np.load(folder / "request-seed.npz")["state"]
    scale = float(np.max(np.abs(seed)))
    result = dict(
        terminal_unknown_t=unknown,
        image_difference_sup_wb=float(np.max(np.abs(difference))),
        predicted_image_sup_wb=float(np.max(np.abs(expected))),
        image_prediction_error_sup_wb=error,
        trips=[],
        stages=[],
    )
    for index, trip in enumerate(spatial["trips"]):
        arrays = dict(np.load(folder / f"inner-{index}.npz"))
        diagnostics = np.load(folder / f"diagnostics-{index}.npz")
        target = (
            np.load(folder / f"inner-{index + 1}.npz")["initial"][:size] * scale
            if index + 1 < len(spatial["trips"])
            else data["state"]
        )
        candidates = dict(
            incoming=arrays["initial"],
            returned=arrays["state"],
            midpoint=(arrays["initial"] + arrays["state"]) / 2,
            trajectory=diagnostics["trajectory_state"],
        )
        distances = {
            name: float(np.max(np.abs(value[:size] * scale - target)))
            for name, value in candidates.items()
        }
        result["trips"].append(
            dict(
                index=index,
                initial_unknown_t=float(arrays["initial"][size]),
                returned_unknown_t=float(arrays["state"][size]),
                next_state_matching=[k for k, v in distances.items() if v < 1e-12],
                next_state_distances_wb=distances,
            )
        )
    for stage in spatial.get("stages", []):
        state = np.load(folder / f"{stage['stage']}-{stage['index']}.npz")["state"]
        result["stages"].append(dict(**stage, inactive_unknown_t=float(state[size])))
    (folder / "iterate-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "ITERATE_AUDIT "
        + json.dumps(
            {
                "case": folder.name,
                **{k: v for k, v in result.items() if k not in ("trips", "stages")},
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    for frame, branch in ROWS:
        audit(args.input / "edge" / f"frame-{frame}-{branch}")


if __name__ == "__main__":
    main()
