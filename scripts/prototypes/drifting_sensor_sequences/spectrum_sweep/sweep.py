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
    path = STUDY / ("efit-" + label(arm) + ".jsonl")
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


def salvage(arm):
    """Read a finished imas-efit output slice by slice, absent boundary included.

    imas-efit stores an equilibrium for every requested slice, but above a
    noise size some of those states carry no boundary outline at all
    (``boundary.outline`` has shape (0, 2) although the axis is present).  The
    study's reader refuses such a state, so the per-slice pass here records the
    axis and leaves the boundary-derived quantities absent, which is what the
    stored output holds rather than a reconstruction failure.
    """
    import imas
    from nova.imas.mast_efit_referee import q95_from_flux_map, read_equilibrium_state

    study, _analyse = import_study()
    truth = json.loads((STUDY / "generating.json").read_text())
    output = STUDY / ("efit-" + label(arm))
    wall = str(study.wall_npz(STUDY / "conf" / "machine-wall.npz"))
    entry = imas.DBEntry("imas:hdf5?path=" + str(output), "r", dd_version="4.1.0")
    equilibrium = entry.get("equilibrium")
    rows = []
    for index in range(SLICES):
        frame = equilibrium.time_slice[index]
        axis = [
            float(frame.global_quantities.magnetic_axis.r),
            float(frame.global_quantities.magnetic_axis.z),
        ]
        record = {"slice": index, "axis": axis}
        try:
            state = read_equilibrium_state(output, time_index=index, wall_source=wall)
            q95 = float(q95_from_flux_map(state, vacuum_b0=truth["b0"]))
        except Exception as error:
            record.update(
                flux_axis=None,
                flux_boundary=None,
                flux_span=None,
                plasma_current=None,
                q95=None,
                q95_stored=None,
                converged=False,
                unavailable=str(error).rsplit(":", 1)[-1].strip(),
            )
        else:
            record.update(
                flux_axis=float(state.flux_axis),
                flux_boundary=float(state.flux_boundary),
                flux_span=float(state.flux_boundary - state.flux_axis),
                plasma_current=float(state.plasma_current_a),
                q95=q95,
                q95_stored=float(state.q95),
                converged=True,
            )
        rows.append(record)
    entry.close()
    path = STUDY / ("efit-" + label(arm) + ".jsonl")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {
        "arm": arm["name"],
        "label": label(arm),
        "rows": len(rows),
        "unavailable": int(sum(1 for row in rows if row.get("q95") is None)),
        "path": str(path),
    }


def repair(todo):
    """Salvage every arm whose jsonl is shorter than the sequence."""
    done = []
    for arm in todo:
        if complete(arm):
            continue
        done.append(salvage(arm))
        # salvaged rows are short of a boundary on some slices, so the file is
        # complete by slice count but not by the study's own reader
        done[-1]["status"] = "salvaged"
    return done


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
            "metrics": analyse.reduce_arm(rows, truth) if len(rows) > 1 else None,
        }
    return measured


def analysis(study, analyse):
    """Reduce every completed arm and the controls into one payload."""
    truth = json.loads((STUDY / "generating.json").read_text())
    table = {}
    for arm in arms():
        path = STUDY / ("efit-" + label(arm) + ".jsonl")
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
        record.update(analyse.reduce_arm(rows, truth))
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
            "axis_0p05_m": analyse.crossing(sizes, "axis_rms", 0.05),
            "q95_0p2": analyse.crossing(sizes, "q95_rms", 0.2),
        },
        "generator_control": generator_control(study, truth),
        "nova": nova_column(truth, analyse),
        "truth": truth,
    }
    return payload


CHECKS = (
    "arms",
    "slices",
    "generator-control",
    "white-control",
    "crossing",
    "nova",
)


def validate(payload):
    """Return one note per check that the payload fails.

    The sweep's own numbers decide whether the gate is green: a control that
    stopped reproducing the white draw, an arm that never completed, or a sweep
    whose crossing vanished is a failed measurement rather than a table to
    publish, so the analyse step exits nonzero and the payload is not written.
    """
    failures = []
    table = payload["arms"]
    expected = {arm["name"] for arm in arms()}
    missing = sorted(expected - set(table))
    if missing:
        failures.append("arms: missing " + ", ".join(missing))
    short = sorted(name for name, row in table.items() if row["slices"] != SLICES)
    if short:
        failures.append("slices: " + ", ".join(short) + " not at %d" % SLICES)
    for name, row in table.items():
        factor = row["factor"]
        if factor is None or factor > 0.1:
            continue
        if row["slices_with_boundary"] < SLICES:
            failures.append(
                "slices: %s carries a boundary on only %d of %d"
                % (name, row["slices_with_boundary"], SLICES)
            )
    for name, row in table.items():
        if row["slices_with_boundary"] < SLICES // 5:
            failures.append(
                "slices: %s lost more than four fifths of its states" % name
            )
    control = payload["generator_control"]
    if abs(control["index0_minus_white"]) > 0.01:
        failures.append(
            "generator-control: index 0 minus white exceeds 0.01 (%.5f)"
            % control["index0_minus_white"]
        )
    if control["zero_lag1_max_abs"] > 0.15:
        failures.append(
            "generator-control: per-sensor index-0 lag-one exceeds 0.15 (%.4f)"
            % control["zero_lag1_max_abs"]
        )
    if control["index1_lag1"] < 0.5:
        failures.append(
            "generator-control: index-1 lag-one fell below 0.5 (%.4f)"
            % control["index1_lag1"]
        )
    if control["slices"] < 1000:
        failures.append("generator-control: %d slices is too few" % control["slices"])
    white = table.get("white-reference")
    if white is None:
        failures.append("white-control: the white arm was not reduced")
    else:
        for stream in ("axis", "flux", "q95"):
            value = white[stream + "_lag1"]
            if not abs(value) <= 0.15:
                failures.append(
                    "white-control: %s lag-one %s outside +/-0.15" % (stream, value)
                )
    for key, limit in (("axis_0p05_m", 0.05), ("q95_0p2", 0.2)):
        crossing = payload["crossing"][key]
        if crossing["factor"] is None:
            failures.append("crossing: %s never exceeds %.2f" % (key, limit))
        elif crossing["value"] <= limit:
            failures.append("crossing: %s reports a value under its limit" % key)
    for sequence, row in payload["nova"].items():
        if not row["source_exists"]:
            failures.append("nova: no receipt file for the %s sequence" % sequence)
            continue
        if row["metrics"] is None:
            failures.append("nova: the %s sequence reduced nothing" % sequence)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("arms")
    run = sub.add_parser("run")
    run.add_argument("--workers", type=int, default=5)
    run.add_argument("--python", default=sys.executable)
    sub.add_parser("analyse")
    sub.add_parser("salvage")
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
    if arguments.command == "salvage":
        for result in repair(arms()):
            print(json.dumps(result), flush=True)
        return 0
    if arguments.command == "analyse":
        study, analyse = import_study()
        payload = analysis(study, analyse)
        print(json.dumps(payload, indent=2))
        failures = validate(payload)
        for note in failures:
            print("CHECK-FAILED " + note, flush=True)
        if failures:
            print("%d check(s) failed; payload not written" % len(failures))
            return 1
        (STUDY / "spectrum-summary.json").write_text(json.dumps(payload, indent=2))
        print("CHECKS-PASSED %d" % len(CHECKS), flush=True)
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
