"""Drift statistics from the per-slice reconstructions of both codes."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

STUDY = Path(os.environ.get("DRIFT_STUDY", str(Path(__file__).resolve().parent)))
BLOCK = 50


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def series(rows: list[dict]) -> dict:
    return dict(
        axis=np.array([row["axis"] for row in rows], dtype=float),
        flux=np.array([row["flux_boundary"] for row in rows], dtype=float),
        q95=np.abs(np.array([row["q95"] for row in rows], dtype=float)),
        current=np.array([row["plasma_current"] for row in rows], dtype=float),
    )


def lag_one(value: np.ndarray) -> float:
    """Slice-to-slice correlation of one series."""
    value = np.asarray(value, dtype=float)
    if value.ndim == 2:
        return float(np.mean([lag_one(column) for column in value.T]))
    if value.size < 3 or np.std(value) == 0.0:
        return float("nan")
    return float(np.corrcoef(value[:-1], value[1:])[0, 1])


def error(series_value: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.sum((series_value - truth) ** 2, axis=1)))


def block_error(value: np.ndarray, truth: np.ndarray, block: int = BLOCK) -> float:
    value = np.asarray(value, dtype=float)
    truth = np.asarray(truth, dtype=float)
    count = value.shape[0] // block
    if count == 0:
        return float("nan")
    means = np.array(
        [
            value[index * block : (index + 1) * block].mean(axis=0)
            for index in range(count)
        ]
    )
    residual = means - truth
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=-1))))


def summarise(rows: list[dict], truth: dict, jacobian: dict) -> dict:
    values = series(rows)
    axis_truth = np.array(truth["axis"], dtype=float)
    flux_truth = float(truth["flux_boundary"]) / float(truth["flux_span"])
    q95_truth = abs(float(truth["q95"]))
    flux_fraction = values["flux"] / float(truth["flux_span"])
    axis_error = np.sqrt(np.mean(np.sum((values["axis"] - axis_truth) ** 2, axis=1)))
    record = dict(
        slices=len(rows),
        axis_rms=float(axis_error),
        flux_rms=float(np.sqrt(np.mean((flux_fraction - flux_truth) ** 2))),
        q95_rms=float(np.sqrt(np.mean((values["q95"] - q95_truth) ** 2))),
        axis_lag1=lag_one(values["axis"]),
        flux_lag1=lag_one(flux_fraction),
        q95_lag1=lag_one(values["q95"]),
        axis_lag1_r=lag_one(values["axis"][:, 0]),
        axis_lag1_z=lag_one(values["axis"][:, 1]),
        axis_block50=block_error(values["axis"], axis_truth),
        flux_block50=block_error(flux_fraction, np.array([flux_truth])),
        q95_block50=block_error(values["q95"], np.array([q95_truth])),
        axis_mean=values["axis"].mean(axis=0).tolist(),
        q95_mean=float(values["q95"].mean()),
        current_rms=float(
            np.sqrt(np.mean((values["current"] - float(truth["plasma_current"])) ** 2))
        )
        if truth.get("plasma_current")
        else 0.0,
        converged=float(np.mean([row.get("converged", True) for row in rows])),
        axis_covariance_ratio=None,
        q95_covariance_ratio=None,
        axis_stated_sigma=None,
        q95_stated_sigma=None,
    )
    if "covariance" in rows[0]:
        covariance = np.array(
            [np.asarray(row["covariance"], dtype=float).reshape(2, 2) for row in rows]
        ).mean(axis=0)
        jac_axis = np.array(jacobian["axis"], dtype=float)
        design = np.vstack([jac_axis, jac_axis])
        axis_covariance = design @ covariance @ design.T
        var_axis = float(np.trace(axis_covariance))
        jac_q95 = np.full(2, float(jacobian["q95"]))
        var_q95 = float(jac_q95 @ covariance @ jac_q95)
        block_axis = block_error(values["axis"], axis_truth)
        block_q95 = block_error(values["q95"], np.array([q95_truth]))
        record["axis_covariance_ratio"] = float((block_axis**2) / (var_axis / BLOCK))
        record["q95_covariance_ratio"] = float((block_q95**2) / (var_q95 / BLOCK))
        record["axis_stated_sigma"] = float(np.sqrt(var_axis))
        record["q95_stated_sigma"] = float(np.sqrt(var_q95))
    return record


def thresholds(table: tuple[dict, dict]) -> dict:
    """Clause (2): the lag-one control both sequences must satisfy."""
    white, drift = table
    return dict(
        white_lag1_pass=bool(
            abs(white["axis_lag1"]) <= 0.15
            and abs(white["flux_lag1"]) <= 0.15
            and abs(white["q95_lag1"]) <= 0.15
        ),
        drift_lag1_pass=bool(
            drift["axis_lag1"] > 0.5
            and drift["flux_lag1"] > 0.5
            and drift["q95_lag1"] > 0.5
        ),
    )


def grouped(paths: list[Path]) -> dict[str, list[dict]]:
    """Merge shard files of one run into a single slice-ordered series."""
    runs: dict[str, list[dict]] = {}
    for path in paths:
        stem = re.sub(r"-shard[0-9]+$", "", path.stem)
        runs.setdefault(stem, []).extend(load(path))
    for rows in runs.values():
        rows.sort(key=lambda row: row["slice"])
    return runs


def crossing(points: dict[float, dict], key: str, limit: float) -> float | None:
    for factor in sorted(points):
        if points[factor][key] > limit:
            return factor
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*")
    args = parser.parse_args()
    truth = json.loads((STUDY / "generating.json").read_text())
    jacobian_file = STUDY / "jacobian.json"
    jacobian = (
        json.loads(jacobian_file.read_text())
        if jacobian_file.exists()
        else dict(axis=[0.0, 0.0], q95=0.0)
    )
    summary = {}
    for code, prefix in (("nova", "nova-"), ("efit", "efit-")):
        paths = [path for path in sorted(STUDY.glob(prefix + args.pattern + ".jsonl"))]
        for stem, rows in grouped(paths).items():
            if not rows:
                continue
            summary[stem] = summarise(rows, truth, jacobian)

    reference = {
        key: value
        for key, value in summary.items()
        if key.split("-")[2:3] == ["200"]
        or re.search(r"-[0-9]+-0\.0100(-shard[0-9]+)?$", key)
    }

    def pick(code: str, sequence: str) -> dict | None:
        """The series that reconstructs best for this code and sequence, then longest.

        Two reference configurations exist for the efit arms: every magnetics row,
        and the 24 flux loops alone; the row set whose reconstruction is closest to
        the generating state is the one the lag-one control is stated on.
        """
        candidates = []
        for key in summary:
            match = re.match(r"%s-%s-([0-9]+)-0\.0100" % (code, sequence), key)
            if match:
                value = summary[key]
                candidates.append((value["axis_rms"], -int(match.group(1)), key))
        return summary[min(candidates)[2]] if candidates else None

    control = {}
    for code in ("nova", "efit"):
        streams = {sequence: pick(code, sequence) for sequence in ("white", "drift")}
        if streams["white"] is None or streams["drift"] is None:
            control[code] = dict(white=streams["white"], drift=streams["drift"])
            continue
        control[code] = thresholds((streams["white"], streams["drift"]))
        control[code]["white"] = streams["white"]
        control[code]["drift"] = streams["drift"]

    sizes = {"nova": {}, "efit": {}}
    for key, value in summary.items():
        match = re.match(r"nova-sweep-50-([0-9.]+)$", key)
        if match:
            sizes["nova"][float(match.group(1))] = value
        match = re.match(r"efit-white-50-([0-9.]+)$", key)
        if match:
            sizes["efit"][float(match.group(1))] = value
    crossing_table = {
        code: dict(
            axis_m=crossing(points, "axis_rms", 0.05),
            q95_limit=crossing(points, "q95_rms", 0.2),
        )
        for code, points in sizes.items()
    }

    spectrum = {"nova": {}, "efit": {}}
    for key, value in summary.items():
        match = re.match(r"nova-spectrum-50-index([0-9.]+)-corner([0-9.]+)$", key)
        if match:
            spectrum["nova"][(float(match.group(1)), float(match.group(2)))] = value
        match = re.match(
            r"efit-drift-50-([0-9.]+)-index([0-9.]+)-corner([0-9.]+)$", key
        )
        if match:
            spectrum["efit"][(float(match.group(2)), float(match.group(3)))] = value

    payload = dict(
        reference=reference,
        control=control,
        sizes={
            code: {str(k): v for k, v in points.items()}
            for code, points in sizes.items()
        },
        crossing=crossing_table,
        spectrum={
            code: {
                "index=%.1f corner=%.4f" % point: value
                for point, value in points.items()
            }
            for code, points in spectrum.items()
        },
    )
    (STUDY / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: payload[key] for key in ("crossing",)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
