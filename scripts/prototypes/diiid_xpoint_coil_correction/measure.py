"""Solve one DIII-D row with a fitted conductor correction on its corpus columns.

The fitted conductor error is applied where the solve reads the measured
currents: the corpus ``magnetics_*`` columns. The build seam and the current
adapter both interpolate those same columns, so a shift written into the row is
what the operator is built from and what the adapter installs, and the adapter's
own agreement check sees one vector rather than two. Applying the same number to
the build seam alone is retained as ``--build-seam-only``: it is the control
that requires the adapter to refuse a current the solve would otherwise never
consume.

Every arm writes beside its solve:

``current-inputs.json``   the measured, equivalent-error and applied vectors as
                          ampere-turns per conductor.
``solve-currents.json``   the conductor order and current vector the adapter
                          installed into the solve, which is the record the
                          solved-current check reads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_xpoint_held_nodes"))

SPEC = importlib.util.spec_from_file_location(
    "held_measure", ROOT / "scripts/prototypes/diiid_xpoint_held_nodes/measure.py"
)
held_measure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = held_measure
SPEC.loader.exec_module(held_measure)

import numpy as np  # noqa: E402

from benchmarks import diiid_forward_gs_match as bench  # noqa: E402

import conductor_correction  # noqa: E402


def conduct_columns(row):
    description = bench.dataset_machine_description(
        row, source_row=str(row.get("_source_path", "corpus row"))
    )
    return {item.name: item for item in description.physical.conductors}


def measured_current(row, conductors, names, time_ms):
    """The ampere-turns per conductor, read from the row as it stands."""

    values = []
    for name in names:
        conductor = conductors[name]
        values.append(
            conductor_correction.current_at(
                row["magnetics_time"],
                row[conductor.input_column],
                time_ms,
                conductor.turns.applied_multiplier,
            )
        )
    return values


def column_shifts(conductors, correction, names):
    """The per-column shift in the column's own kiloamp unit."""

    shifts = {}
    for name in names:
        amount = float(correction.get(name, 0.0))
        if amount == 0.0:
            continue
        conductor = conductors[name]
        column = conductor.input_column
        shifts[column] = shifts.get(column, 0.0) + conductor_correction.column_shift(
            amount, conductor.turns.applied_multiplier
        )
    return shifts


def install_corpus_correction(correction, output, frame, names):
    """Shift the corpus columns, then record measured and applied vectors.

    The measured vector is read from the row before the shift and the applied
    vector from the row the solve consumes, so the receipt shows what the shift
    did rather than what it was asked to do.
    """

    original = bench._read

    def corrected_read(path, columns):
        row = original(path, columns)
        if not correction:
            return row
        row["_source_path"] = str(path)
        conductors = conduct_columns(row)
        time_ms = float(row["efit_times"][frame])
        measured = measured_current(row, conductors, names, time_ms)
        for column, shift in column_shifts(conductors, correction, names).items():
            row[column] = np.asarray(row[column], dtype=float) + shift
        applied = measured_current(row, conductors, names, time_ms)
        error = conductor_correction.equivalent_error(names, correction)
        output.mkdir(parents=True, exist_ok=True)
        (output / "current-inputs.json").write_text(
            json.dumps(
                {
                    "time_ms": time_ms,
                    "names": list(names),
                    "measured_a_turn": measured,
                    "correction_a_turn": error,
                    "applied_a_turn": applied,
                    "relative_correction": conductor_correction.relative_correction(
                        measured, error
                    ),
                    "seam": "corpus-columns",
                },
                indent=2,
            )
            + "\n"
        )
        return row

    bench._read = corrected_read


def install_build_seam_correction(correction, output):
    """Apply the correction only to the build seam, never to the adapter's read.

    The adapter installs a resolution built from the unmodified columns, so it
    and the operator's build-time current disagree and the adapter must refuse.
    """

    original = bench._source_current

    def applied(row, description, names, time_ms):
        measured = original(row, description, names, time_ms)
        error = conductor_correction.equivalent_error(names, correction)
        currents = conductor_correction.applied_currents(names, measured, correction)
        output.mkdir(parents=True, exist_ok=True)
        (output / "current-inputs.json").write_text(
            json.dumps(
                {
                    "time_ms": float(time_ms),
                    "names": list(names),
                    "measured_a_turn": [float(item) for item in measured],
                    "correction_a_turn": error,
                    "applied_a_turn": currents,
                    "relative_correction": conductor_correction.relative_correction(
                        measured, error
                    ),
                    "seam": "build",
                },
                indent=2,
            )
            + "\n"
        )
        return currents

    bench._source_current = applied


def record_solved_currents(output):
    """Record the conductor order and current vector the adapter installed."""

    original = bench.complete_profile_current_adapter

    def recorded(*args, **kwargs):
        adapter = original(*args, **kwargs)
        current = np.asarray(adapter.resolution.current(()), dtype=float)
        names = list(adapter.resolution.names)
        shipped_current = dict(kwargs["shipped_current_a"])
        shipped = [name for name in names if name in shipped_current]
        output.mkdir(parents=True, exist_ok=True)
        (output / "solve-currents.json").write_text(
            json.dumps(
                {
                    "names": names,
                    "current_a_turn": current.tolist(),
                    "shipped_names": shipped,
                    "shipped_current_a_turn": [
                        float(current[names.index(name)]) for name in shipped
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        return adapter

    bench.complete_profile_current_adapter = recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    parser.add_argument(
        "--correction",
        type=Path,
        help="JSON mapping conductor name to an equivalent error in ampere-turns",
    )
    parser.add_argument(
        "--build-seam-only",
        action="store_true",
        help="apply the correction at the build seam and require the adapter's refusal",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert Path(bench.__file__).resolve().is_relative_to(ROOT), bench.__file__
    correction = json.loads(args.correction.read_text()) if args.correction else {}
    if correction and args.build_seam_only:
        install_build_seam_correction(correction, args.output)
    elif correction:
        install_corpus_correction(
            correction, args.output, args.frame, bench.POLOIDAL_CONDUCTORS
        )
    record_solved_currents(args.output)
    held_measure.core.install_analysis = held_measure.install_analysis
    sys.argv = [
        sys.argv[0],
        "--tree",
        str(args.tree),
        "--revision",
        args.revision,
        "--frame",
        str(args.frame),
        "--branch",
        args.branch,
        "--output",
        str(args.output),
        "--trace-helper",
        str(args.trace_helper),
    ]
    held_measure.core.main()


if __name__ == "__main__":
    main()
