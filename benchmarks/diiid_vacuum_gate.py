"""Score DIII-D coil-only vacuum flux outside the shipped LCFS.

The registered bar is the starter kit's documented ``R² ≈ 0.94`` worked use,
made operational here as ``R² >= 0.94``.  It is emitted before any parquet is
opened.  Scoring removes one additive flux gauge per frame and uses the pinned
corpus convention; it fits no sign, current, turn count, amplitude, spatial
mode or physics parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import shapely

from benchmarks.diiid_corpus_conventions import (
    CORPUS_COCOS,
    nova_total_flux_to_corpus,
)
from nova.imas.diiid_description import (
    STARTER_KIT_VACUUM_BAR_SOURCE,
    STARTER_KIT_VACUUM_R2_BAR,
    DiiidDescriptionRegistry,
    vacuum_psi,
    vacuum_response,
)

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")


def _row(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "run the gate with `uv run --with pyarrow python ...`"
        ) from error
    table = parquet.read_table(path)
    return {name: table[name][0].as_py() for name in table.column_names}


def _exterior(row: dict[str, Any], frame: int) -> np.ndarray:
    count = int(row["efit_lcfs_n"][frame])
    boundary = np.column_stack(
        [row["efit_lcfs_r"][frame][:count], row["efit_lcfs_z"][frame][:count]]
    )
    polygon = shapely.Polygon(boundary)
    radius, height = np.meshgrid(row["efit_grid_R"], row["efit_grid_Z"])
    return ~shapely.contains_xy(polygon, radius, height)


def _aligned_pairs(row: dict[str, Any], prediction: np.ndarray):
    true_values: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    truth = np.asarray(row["efit_psirz"], dtype=float)
    for frame in range(len(truth)):
        mask = (
            _exterior(row, frame)
            & np.isfinite(truth[frame])
            & np.isfinite(prediction[frame])
        )
        actual = truth[frame][mask]
        predicted = prediction[frame][mask]
        predicted = predicted + np.mean(actual - predicted)
        true_values.append(actual)
        predicted_values.append(predicted)
    return np.concatenate(true_values), np.concatenate(predicted_values)


def _centered_pairs(row: dict[str, Any], prediction: np.ndarray):
    """Return gauge-free shape vectors without fitting their amplitude."""

    true_values: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    truth = np.asarray(row["efit_psirz"], dtype=float)
    for frame in range(len(truth)):
        mask = (
            _exterior(row, frame)
            & np.isfinite(truth[frame])
            & np.isfinite(prediction[frame])
        )
        actual = truth[frame][mask]
        predicted = prediction[frame][mask]
        true_values.append(actual - np.mean(actual))
        predicted_values.append(predicted - np.mean(predicted))
    return np.concatenate(true_values), np.concatenate(predicted_values)


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def score(paths: list[Path]) -> dict[str, Any]:
    registry = DiiidDescriptionRegistry()
    rows = [_row(path) for path in paths]
    descriptions = [
        registry.ingest(row, source_row=path.name)
        for row, path in zip(rows, paths, strict=True)
    ]
    digests = {description.physical_digest for description in descriptions}
    if len(digests) != 1:
        raise RuntimeError(
            f"selected cohort contains {len(digests)} geometry configurations"
        )
    description = descriptions[0]
    response = vacuum_response(
        description, rows[0]["efit_grid_R"], rows[0]["efit_grid_Z"]
    )
    predictions = [
        nova_total_flux_to_corpus(vacuum_psi(row, description, response))
        for row in rows
    ]
    pairs = [
        _aligned_pairs(row, prediction)
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    pooled = (
        np.concatenate([pair[0] for pair in pairs]),
        np.concatenate([pair[1] for pair in pairs]),
    )
    per_shot = [
        _r2(*_aligned_pairs(row, prediction))
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    pooled_r2 = _r2(*pooled)
    centered = [
        _centered_pairs(row, prediction)
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    centered_actual = np.concatenate([pair[0] for pair in centered])
    centered_prediction = np.concatenate([pair[1] for pair in centered])
    diagnostic_scale = float(
        np.dot(centered_prediction, centered_actual)
        / np.dot(centered_prediction, centered_prediction)
    )
    diagnostic_correlation_r2 = float(
        np.corrcoef(centered_actual, centered_prediction)[0, 1] ** 2
    )
    return {
        "bar_source": STARTER_KIT_VACUUM_BAR_SOURCE,
        "registered_minimum_r2": STARTER_KIT_VACUUM_R2_BAR,
        "passes": pooled_r2 >= STARTER_KIT_VACUUM_R2_BAR,
        "pooled_r2": pooled_r2,
        "diagnostic_best_scale": diagnostic_scale,
        "diagnostic_correlation_r2": diagnostic_correlation_r2,
        "diagnostic_note": (
            "scale and correlation are diagnosis only; neither participates in the gate"
        ),
        "corpus_cocos": CORPUS_COCOS,
        "shots": len(paths),
        "frames": sum(len(row["efit_times"]) for row in rows),
        "geometry_configurations": len(digests),
        "geometry_digest": description.physical_digest,
        "provenance_complete": description.provenance_complete,
        "per_shot_r2": {
            "minimum": float(np.min(per_shot)),
            "q25": float(np.quantile(per_shot, 0.25)),
            "median": float(np.median(per_shot)),
            "q75": float(np.quantile(per_shot, 0.75)),
            "maximum": float(np.max(per_shot)),
            "mean": float(np.mean(per_shot)),
        },
        "shot_files": [path.name for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.data.glob("*.parquet"))[: args.shots]
    if len(paths) < 20:
        raise SystemExit("vacuum gate requires at least twenty train shots")
    preregistration = {
        "bar_source": STARTER_KIT_VACUUM_BAR_SOURCE,
        "registered_minimum_r2": STARTER_KIT_VACUUM_R2_BAR,
    }
    print("PREREGISTERED " + json.dumps(preregistration, sort_keys=True), flush=True)
    result = score(paths)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.write_text(encoded)
    raise SystemExit(0 if result["passes"] else 1)


if __name__ == "__main__":
    main()
