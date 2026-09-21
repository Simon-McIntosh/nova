"""Publish small terminal receipts while retaining full measurements externally."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def compact(source: Path) -> dict:
    data = json.loads(source.read_text())
    result = {
        key: data[key]
        for key in (
            "revision",
            "quadrature_sha256",
            "backend",
            "devices",
            "case",
            "requested_cells",
            "realised_cells",
            "target_current_a",
            "map_checks",
            "positive_controls",
            "wall_seconds",
        )
        if key in data
    }
    result["raw_receipt"] = str(source.resolve())
    result["raw_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["completed"] = "terminal" in data
    if "terminal" in data:
        result["terminal"] = {
            key: value for key, value in data["terminal"].items() if key != "solver"
        }
        solver = data["terminal"]["solver"]
        result["terminal"]["termination"] = solver["termination"]
        result["per_trip_residual_history"] = solver["per_trip_residual_history"]
    result["trips"] = [
        {
            key: trip[key]
            for key in (
                "trip",
                "residual",
                "converged",
                "axis_error_m",
                "state_digest",
                "analytic_max_abs_error_wb",
                "nonzero_current_cells",
            )
            if key in trip
        }
        for trip in data["trips"]
    ]
    cell = data["trips"][0]["record"]["cells"]
    result["outside_centroid_fraction"] = sum(
        current
        for current, flux in zip(
            cell["cell_current_a"], cell["psi_norm_centroid"], strict=True
        )
        if flux > 1
    ) / sum(cell["cell_current_a"])
    result["characteristic_pitch_m"] = median(cell["polygon_area_m2"]) ** 0.5
    if "timing" in data:
        result["timing"] = {
            key: value
            for key, value in data["timing"].items()
            if key != "compiler_events"
        }
        result["timing"]["compiler_event_count"] = len(
            data["timing"]["compiler_events"]
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    index = []
    for source in sorted(arguments.archive.glob("rows-*/record/cells-*/receipt.json")):
        destination = root / source.relative_to(arguments.archive)
        result = compact(source)
        text = json.dumps(result, indent=2) + "\n"
        assert len(text.splitlines()) < 2000
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        index.append(
            {
                "compact_receipt": str(destination.relative_to(root)),
                "raw_receipt": result["raw_receipt"],
                "raw_sha256": result["raw_sha256"],
                "completed": result["completed"],
                "authoritative": source.parts[-4] == "rows-titan-adjoint",
                "terminal_residual": result.get("terminal", {}).get("residual"),
            }
        )
    (root / "measurement-index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
