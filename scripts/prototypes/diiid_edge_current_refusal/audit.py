"""Bind saved iterates to terminal images and inactive constraint unknowns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verify import ROWS, captured_iterations


def audit(folder):
    data = dict(np.load(folder / "terminal.npz"))
    receipt = json.loads((folder / "receipt.json").read_text())
    spatial = json.loads((folder / "spatial.json").read_text())
    captured_iterations(folder, receipt, spatial)
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

    def boundary_current(stem, physical):
        read = np.load(folder / f"{stem}.npz")
        current, connected = read["current"], read["connected"]
        np.testing.assert_allclose(
            current.sum(), physical["unscaled_enclosed_current_a"], rtol=2e-14
        )
        raw = float(current[connected].sum())
        amplitude = physical["current_amplitude"]
        return dict(
            raw_inside_connected_a=raw,
            normalized_inside_connected_a=raw * amplitude,
            normalized_outside_connected_a=float(current[~connected].sum()) * amplitude,
            normalized_total_a=float(current.sum()) * amplitude,
            definition="sum of cell currents whose centres belong to connected support",
        )

    result["terminal_boundary_current"] = boundary_current(
        "terminal-read", spatial["terminal"]
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
        selection = (
            "returned"
            if distances["returned"] < 1e-12
            else "initial"
            if distances["incoming"] < 1e-12
            else None
        )
        selection_basis = "next input or terminal matches the saved endpoint"
        returned_residual = float(
            np.max(np.abs(arrays["mapped"] - arrays["state"]))
            / np.maximum(np.max(np.abs(arrays["mapped"])), 1e-30)
        )
        returned_shadow = np.load(folder / f"returned-read-{index}.npz")["shadow"]
        if (
            selection is None
            and np.array_equal(arrays["state"][:size] * scale, data["state"])
            and np.array_equal(returned_shadow, diagnostics["frozen_shadow"][:size])
            and abs(returned_residual - receipt["residual_history"][index]) < 1e-14
        ):
            selection = "returned"
            selection_basis = (
                "returned state equals terminal, unchanged shadow makes its saved "
                "map live, and its residual equals the reported outer residual"
            )
        if selection is None:
            raise ValueError(f"trip {index}: cannot bind the selected endpoint")
        selected = np.load(folder / f"{selection}-read-{index}.npz")["shadow"]
        frozen = diagnostics["frozen_shadow"][:size]
        entering, leaving = selected & ~frozen, frozen & ~selected
        measured = int(np.sum(entering | leaving))
        reported = receipt["changed_mask_entries"][index]
        if measured != reported:
            raise ValueError(
                f"trip {index}: measured {measured} shadow changes, reported {reported}"
            )
        shadow_transition = dict(
            selected_endpoint=selection,
            selection_basis=selection_basis,
            frozen_shadow_count=int(frozen.sum()),
            selected_shadow_count=int(selected.sum()),
            changed_entries=measured,
            reported_changed_entries=reported,
            matches_reported=True,
            entering_rz_m=coordinates[entering].tolist(),
            leaving_rz_m=coordinates[leaving].tolist(),
        )
        result["trips"].append(
            dict(
                index=index,
                initial_unknown_t=float(arrays["initial"][size]),
                returned_unknown_t=float(arrays["state"][size]),
                next_state_matching=[k for k, v in distances.items() if v < 1e-12],
                next_state_distances_wb=distances,
                shadow_transition=shadow_transition,
                initial_boundary_current=boundary_current(
                    f"initial-read-{index}", trip["initial"]
                ),
                returned_boundary_current=boundary_current(
                    f"returned-read-{index}", trip["returned"]
                ),
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
    audit(args.input / "inactive-field-suppressed")
    audit(args.input / "inactive-field-reference")


if __name__ == "__main__":
    main()
