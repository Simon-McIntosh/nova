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
    """Per-slice series, holding a quantity the record does not carry as NaN.

    A slice whose reconstruction stored no boundary-derived value is written
    with ``None``; an unwritten DD float reads back beyond 1e30 in magnitude,
    which is held absent rather than read as a measurement.
    """

    def column(key, absolute=False):
        values = np.array(
            [float("nan") if row.get(key) is None else row[key] for row in rows],
            dtype=float,
        )
        values[np.abs(values) > 1e30] = np.nan
        return np.abs(values) if absolute else values

    return dict(
        axis=column("axis"),
        flux=column("flux_boundary"),
        q95=column("q95", absolute=True),
        current=column("plasma_current"),
    )


def lag_one(value: np.ndarray) -> float:
    """Slice-to-slice correlation over consecutive finite pairs."""
    value = np.asarray(value, dtype=float)
    if value.ndim == 2:
        finite = np.isfinite(value).all(axis=1)
        value = value[finite]
        if value.shape[0] < 3:
            return float("nan")
        return float(
            np.mean(
                [
                    np.corrcoef(value[:-1, column], value[1:, column])[0, 1]
                    for column in range(value.shape[1])
                ]
            )
        )
    value = value[np.isfinite(value)]
    if value.size < 3 or np.std(value) == 0.0:
        return float("nan")
    return float(np.corrcoef(value[:-1], value[1:])[0, 1])


def rms_error(value: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error over the slices that carry a value."""
    value = np.asarray(value, dtype=float)
    truth = np.asarray(truth, dtype=float).ravel()
    if value.ndim == 1:
        residual = value - float(truth[0])
        finite = np.isfinite(residual)
        if not finite.any():
            return float("nan")
        return float(np.sqrt(np.mean(residual[finite] ** 2)))
    residual = value - truth
    finite = np.isfinite(residual).all(axis=1)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(residual[finite] ** 2, axis=1))))


def block_error(value: np.ndarray, truth: np.ndarray, block: int = BLOCK) -> float:
    """Root-mean-square of each block mean's error, absent slices left out."""
    value = np.asarray(value, dtype=float)
    truth = np.asarray(truth, dtype=float).ravel()
    if value.shape[0] < block:
        return float("nan")
    errors = []
    for start in range(0, value.shape[0] - block + 1, block):
        chunk = value[start : start + block]
        if chunk.ndim == 1:
            finite = np.isfinite(chunk)
            if finite.any():
                errors.append(float(np.abs(chunk[finite].mean() - truth[0])))
        else:
            finite = np.isfinite(chunk).all(axis=1)
            if finite.any():
                errors.append(float(np.linalg.norm(chunk[finite].mean(axis=0) - truth)))
    if not errors:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(errors))))


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
        # The covariance ratio needs the coefficient jacobian, which the
        # probe-row study skips: two more solves on the card buy a quantity this
        # measure does not read.  Without it the design derivative is zero and
        # the ratio is undefined rather than infinite.
        if var_axis > 0.0:
            record["axis_covariance_ratio"] = float(
                (block_axis**2) / (var_axis / BLOCK)
            )
            record["axis_stated_sigma"] = float(np.sqrt(var_axis))
        if var_q95 > 0.0:
            record["q95_covariance_ratio"] = float((block_q95**2) / (var_q95 / BLOCK))
            record["q95_stated_sigma"] = float(np.sqrt(var_q95))
    return record


def reduce_arm(rows: list[dict], truth: dict, block: int = BLOCK) -> dict:
    """Reduce one arm's per-slice rows to the sequence measures.

    The one reduction both the sweep and this module's own summary read, so a
    NaN-aware rule or a new column cannot land in one of them alone.
    """
    values = series(rows)
    span = float(truth["flux_span"])
    axis_truth = [float(value) for value in truth["axis"]]
    flux = values["flux"] / span
    flux_truth = [float(truth["flux_boundary"]) / span]
    q95_truth = float(truth["q95"])
    axis_rms = rms_error(values["axis"], axis_truth)
    flux_rms = rms_error(flux, flux_truth)
    q95_rms = rms_error(values["q95"], q95_truth)
    axis_mean = block_error(values["axis"], axis_truth, block)
    flux_mean = block_error(flux, flux_truth, block)
    q95_mean = block_error(values["q95"], q95_truth, block)
    return dict(
        slices=len(rows),
        slices_with_boundary=int(np.sum(np.isfinite(values["q95"]))),
        converged=float(np.mean([bool(row.get("converged", True)) for row in rows])),
        axis_rms=axis_rms,
        axis_mean_50=axis_mean,
        axis_mean_ratio=axis_mean / axis_rms,
        flux_rms=flux_rms,
        flux_mean_50=flux_mean,
        flux_mean_ratio=flux_mean / flux_rms,
        q95_rms=q95_rms,
        q95_mean_50=q95_mean,
        q95_mean_ratio=q95_mean / q95_rms,
        axis_lag1=lag_one(values["axis"]),
        flux_lag1=lag_one(flux),
        q95_lag1=lag_one(values["q95"]),
        axis_mean=values["axis"].mean(axis=0).tolist(),
        q95_mean=float(np.nanmean(values["q95"])),
    )


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


def crossing_point(points: dict, key: str, limit: float) -> dict:
    """The first swept value whose measure exceeds the limit, and its neighbour."""
    ordered = sorted(points.values(), key=lambda row: row["factor"])
    for position, row in enumerate(ordered):
        if row[key] > limit:
            previous = ordered[position - 1] if position else None
            return dict(
                factor=row["factor"],
                value=row[key],
                previous_factor=None if previous is None else previous["factor"],
                previous_value=None if previous is None else previous[key],
            )
    return dict(factor=None, value=None, previous_factor=None, previous_value=None)


def crossing(points: dict, key: str, limit: float) -> float | None:
    """The first swept value whose measure exceeds the limit."""
    return crossing_point(points, key, limit)["factor"]


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
