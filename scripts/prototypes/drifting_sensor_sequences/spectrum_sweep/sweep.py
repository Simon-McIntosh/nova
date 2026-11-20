"""Sweep the noise size, the spectral index and the corner frequency.

Plan section 7's clauses (3) and (4) need one quantity from
:mod:`probe_rows.study` at many points of the coloured-noise model: every arm is
one IMAS pulse of ``SLICES`` magnetics rows drawn from the held generating
state, written by the study's own pulse writer and reconstructed by imas-efit.
This module only chooses the arms, launches them, and reduces the per-slice
rows each one leaves behind; the pulse, the sensor geometry, the noise shaping
and the imas-efit invocation are the study's.  Going through the study rather
than a copy of it is what keeps a sweep point comparable with the reference
sequence that module already measured.

Units.  A slice is one second apart in the study's pulse (``SLICE_TIME +
arange``), so a corner frequency of ``f`` Hz is ``f * SLICES`` cycles across the
sequence; the study takes its corner as a count of cycles and this module
converts.  The density is flat below the corner and falls as ``f**-index``
above it, as :func:`probe_rows.study.noise_columns` builds it.  Every arm shares
one white draw per sensor, so a size arm is that draw scaled and a spectrum arm
is that draw shaped: the sweep points are nested realisations, not independent
ones.

Reported measure.  For an arm's per-slice series against the generating state:

``*_rms``        root-mean-square error over every slice;
``*_mean_50``    the same error taken on the mean of each 50-slice block;
``*_mean_ratio`` ``*_mean_50 / *_rms`` -- ``1/sqrt(50) = 0.1414`` under
                 independent per-slice noise, and 1.0 when the slices carry one
                 common offset, so the column says how far the spread of the
                 same draws understates the error of the mean;
``*_lag1``       lag-one correlation of the reconstructed series.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SLICES = 100
SLICE_SECONDS = 1.0
REFERENCE_FACTOR = 0.01
REFERENCE_INDEX = 1.0
# The upstream 200-slice reference run shapes four corner cycles across its two
# hundred slices, i.e. 0.02 Hz over the same one-second slices.
REFERENCE_CORNER_HZ = 0.02
SEED = 20261118
BLOCK = 50
SIZE_FACTORS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
INDEX_VALUES = (0.0, 0.5, 1.0, 2.0)
CORNER_HZ = (0.002, 0.02, 0.2)
PROBE_ROWS = Path(__file__).resolve().parents[1] / "probe_rows"
STUDY = Path(os.environ.get("DRIFT_STUDY", str(PROBE_ROWS)))
NOVA_SOURCE = Path(
    os.environ.get(
        "DRIFT_NOVA_SOURCE",
        "/home/ITER/mcintos/.config/reckon/crew/runs"
        "/r-20261119T065220776269-drifting-sensor-probe-rows/study",
    )
)


def import_study():
    """Import the study and its reducer from their own module files."""
    if str(PROBE_ROWS) not in sys.path:
        sys.path.insert(0, str(PROBE_ROWS))
    import analyse
    import study

    return study, analyse


def arms():
    """The sweep's arms, each a point in one dimension of the noise model."""
    points = []
    for factor in SIZE_FACTORS:
        points.append(
            {
                "name": "size-%.4f" % factor,
                "sequence": "drift",
                "factor": factor,
                "index": REFERENCE_INDEX,
                "corner_hz": REFERENCE_CORNER_HZ,
            }
        )
    for index in INDEX_VALUES:
        if index == REFERENCE_INDEX:
            continue
        points.append(
            {
                "name": "index-%.1f" % index,
                "sequence": "drift",
                "factor": REFERENCE_FACTOR,
                "index": index,
                "corner_hz": REFERENCE_CORNER_HZ,
            }
        )
    for corner in CORNER_HZ:
        if corner == REFERENCE_CORNER_HZ:
            continue
        points.append(
            {
                "name": "corner-%.3f" % corner,
                "sequence": "drift",
                "factor": REFERENCE_FACTOR,
                "index": REFERENCE_INDEX,
                "corner_hz": corner,
            }
        )
    points.append(
        {
            "name": "white-reference",
            "sequence": "white",
            "factor": REFERENCE_FACTOR,
            "index": None,
            "corner_hz": None,
        }
    )
    return points


def label(arm):
    """The study's own stem for this arm, so its jsonl is the one it writes."""
    stem = "%s-%d-%.4f" % (arm["sequence"], SLICES, arm["factor"])
    if arm["index"] is not None:
        stem += "-index%.1f-corner%.4f" % (arm["index"], arm["corner_hz"] * SLICES)
    return stem


def command(arm, python):
    """The study invocation for one arm."""
    parts = [
        python,
        str(PROBE_ROWS / "study.py"),
        "efit",
        "--slices",
        str(SLICES),
        "--sequence",
        arm["sequence"],
        "--factor",
        "%.6g" % arm["factor"],
    ]
    if arm["index"] is not None:
        parts += [
            "--index",
            "%.6g" % arm["index"],
            "--corner",
            "%.6g" % (arm["corner_hz"] * SLICES),
        ]
    return parts


def complete(arm):
    path = STUDY / (label(arm) + ".jsonl")
    if not path.exists():
        return False
    return len(path.read_text().splitlines()) >= SLICES


def launch(arm, python, logdir):
    """Run one arm in its own process and record how it ended."""
    if complete(arm):
        return {"arm": arm["name"], "label": label(arm), "status": "reused"}
    log = logdir / (label(arm) + ".log")
    environment = dict(os.environ)
    environment["DRIFT_STUDY"] = str(STUDY)
    with log.open("w") as stream:
        done = subprocess.run(
            command(arm, python),
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=str(STUDY),
        )
    return {
        "arm": arm["name"],
        "label": label(arm),
        "status": "exit=%d" % done.returncode,
        "log": str(log),
    }


def reduce_arm(rows, truth, analyse):
    """Reduce one arm's per-slice rows to the reported measures."""
    import numpy as np

    values = analyse.series(rows)
    axis_truth = np.asarray(truth["axis"], dtype=float)
    span = float(truth["flux_span"])
    flux = values["flux"] / span
    flux_truth = float(truth["flux_boundary"]) / span
    q95_truth = abs(float(truth["q95"]))
    axis_error = float(
        np.sqrt(np.mean(np.sum((values["axis"] / 1.0 - axis_truth) ** 2, axis=1)))
    )
    flux_error = float(np.sqrt(np.mean((flux - flux_truth) ** 2)))
    q95_error = float(np.sqrt(np.mean((values["q95"] - q95_truth) ** 2)))
    axis_mean = float(analyse.block_error(values["axis"], axis_truth, BLOCK))
    flux_mean = float(analyse.block_error(flux, np.array([flux_truth]), BLOCK))
    q95_mean = float(analyse.block_error(values["q95"], np.array([q95_truth]), BLOCK))
    return {
        "slices": len(rows),
        "converged": float(np.mean([bool(row.get("converged", True)) for row in rows])),
        "axis_rms": axis_error,
        "axis_mean_50": axis_mean,
        "axis_mean_ratio": axis_mean / axis_error,
        "flux_rms": flux_error,
        "flux_mean_50": flux_mean,
        "flux_mean_ratio": flux_mean / flux_error,
        "q95_rms": q95_error,
        "q95_mean_50": q95_mean,
        "q95_mean_ratio": q95_mean / q95_error,
        "axis_lag1": float(analyse.lag_one(values["axis"])),
        "flux_lag1": float(analyse.lag_one(flux)),
        "q95_lag1": float(analyse.lag_one(values["q95"])),
        "axis_mean": values["axis"].mean(axis=0).tolist(),
        "q95_mean": float(values["q95"].mean()),
    }


def generator_control(study, truth):
    """Index zero must reproduce the white draw's own lag-one correlation.

    Measured on the generator alone over 2000 slices, so the sampling spread of
    a 100-slice arm cannot stand in for a shaping defect.
    """
    import numpy as np

    sigma = np.r_[
        np.full(24, float(truth["sigma_loop"])),
        np.full(48, float(truth["sigma_probe"])),
    ]
    size = 2000
    cycles = REFERENCE_CORNER_HZ * size
    white = study.noise_columns("white", size, sigma, SEED, REFERENCE_FACTOR)
    zero = study.noise_columns(
        "drift",
        size,
        sigma,
        SEED,
        REFERENCE_FACTOR,
        corner_cycles=cycles,
        index=0.0,
    )
    flat = study.noise_columns(
        "drift",
        size,
        sigma,
        SEED,
        REFERENCE_FACTOR,
        corner_cycles=cycles,
        index=REFERENCE_INDEX,
    )

    def lag(value):
        return float(np.corrcoef(value[:-1], value[1:])[0, 1])

    sensors = flat.shape[1]
    white_lag = np.array([lag(white[:, sensor]) for sensor in range(sensors)])
    zero_lag = np.array([lag(zero[:, sensor]) for sensor in range(sensors)])
    flat_lag = np.array([lag(flat[:, sensor]) for sensor in range(sensors)])
    return {
        "slices": size,
        "sensors": int(sensors),
        "white_lag1": float(white_lag.mean()),
        "index0_lag1": float(zero_lag.mean()),
        "index1_lag1": float(flat_lag.mean()),
        "index0_minus_white": float(zero_lag.mean() - white_lag.mean()),
        "zero_lag1_max_abs": float(np.max(np.abs(zero_lag))),
        "marginal_size_white": float(white.std()),
        "marginal_size_index0": float(zero.std()),
        "index0_equals_white": bool(np.allclose(zero, white)),
    }


def nova_column(truth, analyse):
    """The nova fit at the reference point, on the slices the upstream node ran.

    One fit at this mesh costs 13 to 16 minutes on a 16 GB P100, so every sweep
    point other than the reference one is unmeasured for nova; the upstream
    node's own receipts are read here rather than recomputed.
    """
    measured = {}
    for sequence in ("white", "drift"):
        path = NOVA_SOURCE / ("nova-%s-3-0.0100.jsonl" % sequence)
        rows = analyse.load(path) if path.exists() else []
        measured[sequence] = {
            "source": str(path),
            "source_exists": path.exists(),
            "slices": [int(row["slice"]) for row in rows],
            "wall_seconds": [float(row["wall_seconds"]) for row in rows],
            "converged": [bool(row["converged"]) for row in rows],
            "metrics": reduce_arm(rows, truth, analyse) if len(rows) > 1 else None,
        }
    return measured


def crossing_of(points, key, limit):
    """The first swept factor whose measure exceeds the limit, and its neighbour."""
    ordered = sorted(points.values(), key=lambda row: row["factor"])
    for position, row in enumerate(ordered):
        if row[key] > limit:
            previous = ordered[position - 1] if position else None
            return {
                "factor": row["factor"],
                "value": row[key],
                "previous_factor": None if previous is None else previous["factor"],
                "previous_value": None if previous is None else previous[key],
            }
    return {
        "factor": None,
        "value": None,
        "previous_factor": None,
        "previous_value": None,
    }


def analysis(study, analyse):
    """Reduce every completed arm and the controls into one payload."""
    truth = json.loads((STUDY / "generating.json").read_text())
    table = {}
    for arm in arms():
        path = STUDY / (label(arm) + ".jsonl")
        rows = analyse.load(path) if path.exists() else []
        if len(rows) < SLICES:
            continue
        record = {
            "sequence": arm["sequence"],
            "factor": arm["factor"],
            "index": arm["index"],
            "corner_hz": arm["corner_hz"],
            "corner_cycles": None
            if arm["corner_hz"] is None
            else arm["corner_hz"] * SLICES,
        }
        record.update(reduce_arm(rows, truth, analyse))
        table[arm["name"]] = record
    sizes = {name: value for name, value in table.items() if name.startswith("size-")}
    payload = {
        "reference": {
            "factor": REFERENCE_FACTOR,
            "index": REFERENCE_INDEX,
            "corner_hz": REFERENCE_CORNER_HZ,
            "slices": SLICES,
            "slice_seconds": SLICE_SECONDS,
            "block": BLOCK,
            "size_factors": list(SIZE_FACTORS),
            "index_values": list(INDEX_VALUES),
            "corner_hz_values": list(CORNER_HZ),
        },
        "arms": table,
        "crossing": {
            "axis_0p05_m": crossing_of(sizes, "axis_mean_50", 0.05),
            "q95_0p2": crossing_of(sizes, "q95_mean_50", 0.2),
        },
        "generator_control": generator_control(study, truth),
        "nova": nova_column(truth, analyse),
        "truth": truth,
    }
    (STUDY / "spectrum-summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("arms")
    run = sub.add_parser("run")
    run.add_argument("--workers", type=int, default=5)
    run.add_argument("--python", default=sys.executable)
    sub.add_parser("analyse")
    arguments = parser.parse_args(argv)
    if arguments.command == "arms":
        for arm in arms():
            print(
                json.dumps(
                    {
                        "arm": arm["name"],
                        "label": label(arm),
                        "argv": command(arm, sys.executable),
                    }
                )
            )
        return 0
    if arguments.command == "analyse":
        study, analyse = import_study()
        analysis(study, analyse)
        return 0
    logdir = STUDY / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = [pool.submit(launch, arm, arguments.python, logdir) for arm in arms()]
        results = [future.result() for future in futures]
    for result in results:
        print(json.dumps(result), flush=True)
    failed = [
        result for result in results if result["status"] not in ("reused", "exit=0")
    ]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
