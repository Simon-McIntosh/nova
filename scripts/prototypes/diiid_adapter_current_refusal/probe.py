"""Show whether a build-seam conductor current reaches the adapter or is replaced.

Builds frame 137D through the benchmark's own builder twice: once unpatched, and
once with ``bench._source_current`` shifted on the F1A conductor, which is the
seam the coil-correction driver used. It then reports what
``complete_profile_current_adapter`` does with the two build-time vectors.

The run is the reproduction, so it is rerun unchanged at the head: a build whose
seam current differs is either replaced silently (the defect) or refused with a
message naming the conductor (the repair).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

FRAME = 137
SEAM_SHIFT_A_TURN = 54_058.0
SHIFTED_CONDUCTOR = "F1A"


def _corpus_row():
    from benchmarks import diiid_forward_gs_match as bench

    paths = sorted(bench.DEFAULT_DATA.glob("*.parquet"))
    selected = bench.select_frames(
        paths, bench.EXECUTION_FRAME_COUNT, bench.polarity_population()
    )
    match = [item for item in selected if item.frame == FRAME]
    if len(match) != 1:
        raise RuntimeError(f"frame {FRAME} did not select exactly one corpus row")
    columns = (
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
    )
    return bench, bench._read(match[0].path, columns)


def _shipped(bench, row):
    description = bench.dataset_machine_description(
        row, source_row=str(row.get("_source_path", "corpus row"))
    ).physical
    time_ms = float(row["efit_times"][FRAME])
    currents = bench.shipped_current_at(
        row, description, bench.POLOIDAL_CONDUCTORS, time_ms
    )
    return currents, time_ms


def _verdict(label, profile, shipped, module):
    names = tuple(module.POLOIDAL_CONDUCTORS)
    built = np.asarray(profile.operator.external_current, dtype=float)
    reference = np.asarray([shipped[name] for name in names], dtype=float)
    discrepancy = np.abs(built - reference)
    index = names.index(SHIFTED_CONDUCTOR)
    record = {
        "arm": label,
        "build_vector_size": int(built.size),
        "shipped_vector_size": int(reference.size),
        "max_abs_difference_a": float(np.max(discrepancy)),
        "exact_entries": int(np.count_nonzero(discrepancy == 0.0)),
        "f1a_build_a": float(built[index]),
        "f1a_shipped_a": float(shipped[SHIFTED_CONDUCTOR]),
    }
    try:
        adapter = module.complete_profile_current_adapter(
            profile,
            shipped_names=names,
            shipped_current_a=shipped,
            use_circuit=True,
        )
    except ValueError as error:
        record["adapter"] = "REFUSED"
        record["message"] = str(error)
        return record, None
    installed = np.asarray(adapter.resolution.current(()), dtype=float)
    installed_index = tuple(adapter.resolution.names).index(SHIFTED_CONDUCTOR)
    record["adapter"] = "REPLACED"
    record["installed_f1a_a"] = float(installed[installed_index])
    record["installed_vs_shipped_max_abs_a"] = float(
        np.max(np.abs(installed[: len(reference)] - reference))
    )
    return record, adapter


def _module(path: str):
    """Load the adapter module from a named source file, or from the tree."""

    if path == "tree":
        from nova.imas import diiid_current

        return diiid_current
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nova.imas.diiid_current_probe", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the adapter module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default="tree",
        help="adapter module source to exercise: 'tree' or a file path",
    )
    options = parser.parse_args(argv)
    diiid_current = _module(options.module)

    bench, row = _corpus_row()
    shipped, time_ms = _shipped(bench, row)
    profile, _seed, _label, _wall, _reliable, _statement = bench.build_profile(
        row, FRAME
    )
    records = [_verdict("unpatched", profile, shipped, diiid_current)[0]]

    names = tuple(diiid_current.POLOIDAL_CONDUCTORS)
    vector = np.asarray(profile.operator.external_current, dtype=float).copy()
    vector[names.index(SHIFTED_CONDUCTOR)] += SEAM_SHIFT_A_TURN
    patched = replace(
        profile, operator=replace(profile.operator, external_current=vector)
    )
    records.append(_verdict("patched", patched, shipped, diiid_current)[0])

    print(f"frame {FRAME} time_ms {time_ms:.6f} shift {SEAM_SHIFT_A_TURN} A")
    print(json.dumps(records, indent=2))
    if records[0]["adapter"] != "REPLACED":
        print("UNPATCHED-BUILD-DID-NOT-PASS")
        return 2
    if records[1]["adapter"] == "REFUSED":
        print("PATCHED-BUILD-REFUSED")
        return 0
    print("PATCHED-BUILD-REPLACED-SILENTLY")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())