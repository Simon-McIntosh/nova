"""Compare frame 137D across every in-repo adapter caller and across the adapter.

Phase ``callers`` builds frame 137D once and runs each caller's own
adapter-accepting entry on it, so a caller the new refusal breaks is named.
Entries whose driver runs a whole route repertoire or renders a figure are
exercised through the same construction they use, and the record says so under
``kind``.

Phase ``solve`` solves frame 137D twice through the benchmark's public frame
solve: once with the adapter in this tree, once with the adapter loaded from a
named base-revision file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

FRAME = 137

def _write(output: Path, name: str, record: dict) -> None:
    path = output / (name + ".json")
    path.write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(name.upper() + " " + json.dumps(record, default=str), flush=True)

def _digest(array) -> str:
    values = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def _module(path: str):
    if path == "tree":
        from nova.imas import diiid_current

        return diiid_current
    import importlib.util

    spec = importlib.util.spec_from_file_location("nova.imas.diiid_current_arm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load " + path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def _corpus_row(bench):
    paths = sorted(bench.DEFAULT_DATA.glob("*.parquet"))
    selected = bench.select_frames(
        paths, bench.EXECUTION_FRAME_COUNT, bench.polarity_population()
    )
    match = [item for item in selected if item.frame == FRAME]
    if len(match) != 1:
        raise RuntimeError("frame " + str(FRAME) + " is not one corpus row")
    columns = (
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
    )
    path = match[0].path
    row = bench._read(path, columns)
    row["_source_path"] = str(path)
    return row, path

def _shipped(bench, row):
    description = bench.dataset_machine_description(
        row, source_row=str(row.get("_source_path", "corpus row"))
    ).physical
    time_ms = float(row["efit_times"][FRAME])
    currents = bench.shipped_current_at(
        row, description, bench.POLOIDAL_CONDUCTORS, time_ms
    )
    return currents, time_ms

def _completed(bench, profile, shipped):
    """Install the resolution the way every caller does and size the result."""

    from nova.imas import diiid_current

    adapter = diiid_current.complete_profile_current_adapter(
        profile,
        shipped_names=bench.POLOIDAL_CONDUCTORS,
        shipped_current_a=shipped,
        use_circuit=True,
    )
    current = np.asarray(adapter.resolution.current(()), dtype=float)
    return {
        "complete_count": int(current.size),
        "current_digest": _digest(current),
    }

def _run(output, results, name, kind, function):
    record = {"caller": name, "kind": kind}
    try:
        detail = function()
    except Exception as error:
        record["status"] = "error"
        record["error"] = type(error).__name__ + ": " + str(error)
    else:
        record["status"] = "built"
        if isinstance(detail, dict):
            record.update(detail)
    results.append((name, record["status"]))
    _write(output, name, record)
    return record

def _caller_entries(bench, row, path, profile, shipped, time_ms):
    """Return (name, kind, callable) for every in-repo adapter call site."""

    entries = []

    def bench_wall_slice():
        return _completed(bench, profile, shipped)

    def pinned_entry():
        from benchmarks import diiid_current_pinned_forward as pinned

        _profile, _shipped_map, current, _receipt = pinned.description_driven_currents(
            row, profile, time_ms
        )
        return {"complete_count": int(np.asarray(current).size)}

    entries.append(("wall-topology", "slice", bench_wall_slice))
    entries.append(("pinned", "entry", pinned_entry))

    def repaired_entry():
        from benchmarks import diiid_repaired_solve_remeasure as remeasure

        _profile, current, _target, _time_ms, _seed = remeasure._prepare_frame(
            row, FRAME
        )
        return {"complete_count": int(np.asarray(current).size)}

    entries.append(("repaired", "entry", repaired_entry))

    def cold_entry():
        from benchmarks import diiid_constrained_cold_start as cold

        prepared = cold.prepare_frame(path, FRAME)
        return {"complete_count": int(np.asarray(prepared.current).size)}

    def diverted_entry():
        from benchmarks import diiid_diverted_root_full_currents as diverted

        _profile, arms, _receipt = diverted.description_driven_currents(
            row, profile, time_ms
        )
        return {"arm_sizes": [int(np.asarray(arm).size) for arm in arms]}

    def circuit_driven_slice():
        return _completed(bench, profile, shipped)

    def strict_entry():
        from benchmarks import strict_exit_incidence as strict

        _member, gate_row = strict._build_diiid_member_row(
            {"shot": path.name, "frame": FRAME},
            number=1,
            machine_cache=bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
        )
        return {"gate_row_keys": len(gate_row)}

    def map_fidelity_entry():
        import importlib.util as loader

        spec = loader.spec_from_file_location(
            "diiid_map_fidelity_measure",
            ROOT / "scripts" / "diiid_map_fidelity" / "measure.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the map-fidelity driver")
        module = loader.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        prepared = module.prepare_diiid(FRAME)
        return {"complete_count": int(np.asarray(prepared["current"]).size)}

    def reachability_slice():
        return _completed(bench, profile, shipped)

    entries.append(("cold-start", "entry", cold_entry))
    entries.append(("diverted-root", "entry", diverted_entry))
    entries.append(("circuit-driven", "slice", circuit_driven_slice))
    entries.append(("strict-exit", "entry", strict_entry))
    entries.append(("map-fidelity", "entry", map_fidelity_entry))
    entries.append(("reachability", "slice", reachability_slice))
    return entries

def _solve_arm(bench, row, adapter):
    """Solve frame 137D with the named adapter module installed in the benchmark."""

    original = bench.complete_profile_current_adapter
    bench.complete_profile_current_adapter = adapter.complete_profile_current_adapter
    try:
        result, fields = bench.solve_frame(
            row, FRAME, bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
        )
    finally:
        bench.complete_profile_current_adapter = original
    flux = np.asarray(fields["predicted"], dtype=float)
    return {
        "converged": bool(result.converged),
        "finite": bool(result.finite),
        "relative_residual": float(result.fixed_point_relative_residual),
        "flux_digest": _digest(flux),
        "flux_shape": [int(value) for value in flux.shape],
        "flux_sum": float(np.sum(flux)),
    }

def _callers(output: Path) -> list[tuple[str, str]]:
    from benchmarks import diiid_forward_gs_match as bench

    row, path = _corpus_row(bench)
    profile, _seed, *_rest = bench.build_profile(
        row, FRAME, bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
    )
    shipped, time_ms = _shipped(bench, row)
    results = []
    for name, kind, function in _caller_entries(
        bench, row, path, profile, shipped, time_ms
    ):
        _run(output, results, name, kind, function)
    return results

def _solve(output: Path, base_module: str) -> int:
    from benchmarks import diiid_forward_gs_match as bench

    row, _path = _corpus_row(bench)
    head = _module("tree")
    base = _module(base_module)
    head_record = _solve_arm(bench, row, head)
    _write(output, "solve-head", head_record)
    base_record = _solve_arm(bench, row, base)
    _write(output, "solve-base", base_record)
    identical = head_record["flux_digest"] == base_record["flux_digest"]
    comparison = {
        "flux_identical": bool(identical),
        "head_digest": head_record["flux_digest"],
        "base_digest": base_record["flux_digest"],
        "identical_inputs": True,
    }
    _write(output, "solve-comparison", comparison)
    return 0 if identical else 1

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-module", default="tree")
    parser.add_argument(
        "--phase", choices=("callers", "solve", "both"), default="both"
    )
    options = parser.parse_args(argv)
    options.output.mkdir(parents=True, exist_ok=True)
    status = 0
    if options.phase in ("callers", "both"):
        results = _callers(options.output)
        failed = [name for name, state in results if state != "built"]
        print(
            "CALLERS-SUMMARY "
            + json.dumps({"built": len(results) - len(failed), "failed": failed})
        )
        if failed:
            status = 1
    if options.phase in ("solve", "both"):
        status = _solve(options.output, options.base_module) or status
    return status


if __name__ == "__main__":
    raise SystemExit(main())