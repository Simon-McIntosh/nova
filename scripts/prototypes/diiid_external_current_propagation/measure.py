"""Measure which conductor state the DIII-D forward solve actually reads.

The coil-correction arms came back bit-identical to the measured arms although
the correction moved the operator's external field. This driver isolates the
candidate explanations by solving frame 137D on arms that differ only in where
the conductor current is installed:

``base``      the corpus row exactly as read, on the registered pseudo-wall;
``physical``  the same row on the physical wall, to identify which wall the
              banked cohort used;
``seam``      the correction driver's seam: :func:`bench._source_current`
              returns a uniformly shifted vector, everything else untouched;
``row``       every shipped poloidal conductor channel in the corpus row is
              scaled, so both current readers see the change;
``warm``      two solves on ONE adapted profile: solve the base conductor
              vector, then the scaled one, so the accelerated history program
              and the request compilation cache are already warm.

Each arm writes its terminal grid flux, lower null and receipt as it lands, so
an expired allocation loses one arm rather than the run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

FRAME = 137
SEAM_SHIFT_A_TURN = 54058.0
ROW_SCALE = 1.1
PSEUDO_WALL_EXPANSION = 0.02


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


def _machine(bench, row):
    from nova.imas.diiid_description import dataset_machine_description

    return dataset_machine_description(
        row, source_row=str(row.get("_source_path", "corpus row"))
    )


def _shipped(bench, row):
    time_ms = float(row["efit_times"][FRAME])
    shipped = bench.shipped_current_at(
        row, _machine(bench, row).physical, bench.POLOIDAL_CONDUCTORS, time_ms
    )
    return shipped, time_ms


def _shipped_scaled(bench, row):
    """Scale every shipped poloidal conductor channel in the corpus row."""

    import numpy as np

    scaled = dict(row)
    for name in bench.POLOIDAL_CONDUCTORS:
        column = f"magnetics_{name}"
        scaled[column] = np.asarray(row[column], dtype=float) * ROW_SCALE
    return scaled


def _pre_adapter_external(bench, row, expansion):
    """Return the external-field sup of the operator before the current adapter."""
    import numpy as np

    profile, *_rest = bench.build_profile(row, FRAME, expansion)
    current = np.asarray(
        bench._source_current(
            row,
            _machine(bench, row).physical,
            bench.POLOIDAL_CONDUCTORS,
            float(row["efit_times"][FRAME]),
        ),
        dtype=float,
    )
    return _external_sup(profile.operator, current)


def _adapted(bench, row, expansion):
    """Build the profile the cohort solves, with its 24-column operator."""

    import numpy as np

    from nova.imas.diiid_current import complete_profile_current_adapter

    profile, seed, label, wall, _reliable, _statement = bench.build_profile(
        row, FRAME, expansion
    )
    shipped, time_ms = _shipped(bench, row)
    adapter = complete_profile_current_adapter(
        profile,
        shipped_names=bench.POLOIDAL_CONDUCTORS,
        shipped_current_a=shipped,
        use_circuit=True,
    )
    return {
        "profile": adapter.profile,
        "seed": seed,
        "label": label,
        "wall": wall,
        "time_ms": time_ms,
        "shipped": shipped,
        "complete": np.asarray(adapter.resolution.current(()), dtype=float),
    }


def _external_sup(operator, current):
    import numpy as np

    return float(np.max(np.abs(np.asarray(operator.external(current)))))


def _solve_detail(bench, built, complete, row):
    """Return the equilibrium and the termination the route reported."""

    (
        equilibrium,
        termination,
        _selection,
        _iterations,
        _identity,
        converged,
        _,
    ) = bench._solve_registered(
        built["profile"],
        built["seed"],
        row,
        FRAME,
        complete,
        bench._target_current(row, built["time_ms"]),
    )
    equilibrium.flux.block_until_ready()
    return equilibrium, termination, converged


def _record(
    output,
    name,
    built,
    equilibrium,
    row,
    complete,
    external_pre,
    external_solved,
    termination=None,
    converged=None,
):
    import numpy as np

    from benchmarks.diiid_state_of_play_figures import boundary_gradient_minimum

    profile = built["profile"]
    lattice = profile.lattice
    flux = np.asarray(equilibrium.flux, dtype=np.float64)
    grid_flux = flux[: lattice.node_count].reshape(lattice.shape)
    topology = equilibrium.topology
    lower_x = np.asarray(topology.x_point, dtype=float)
    axis = np.asarray(topology.axis, dtype=float)
    from benchmarks import diiid_forward_gs_match as bench

    full_radius, full_height = bench.canonical_axes(row)
    count = int(row["efit_lcfs_n"][FRAME])
    boundary = np.c_[
        np.asarray(row["efit_lcfs_r"][FRAME][:count], dtype=float),
        np.asarray(row["efit_lcfs_z"][FRAME][:count], dtype=float),
    ]
    efit_x = np.asarray(
        boundary_gradient_minimum(
            full_radius,
            full_height,
            np.asarray(row["efit_psirz"][FRAME], dtype=float),
            boundary,
        ),
        dtype=float,
    )
    offset_cm = 100.0 * (lower_x - efit_x)
    record = {
        "arm": name,
        "frame": FRAME,
        "residual": float(equilibrium.fixed_point.residual),
        "termination": str(termination),
        "converged": None if converged is None else bool(converged),
        "diverted": bool(topology.diverted),
        "lower_x_rz_m": lower_x.tolist(),
        "efit_lower_x_rz_m": efit_x.tolist(),
        "lower_x_offset_cm": offset_cm.tolist(),
        "lower_x_distance_cm": float(np.linalg.norm(offset_cm)),
        "axis_rz_m": axis.tolist(),
        "current_vector_sup_a_turn": float(np.max(np.abs(complete))),
        "external_pre_adapter_sup_wb": float(external_pre),
        "external_solved_sup_wb": float(external_solved),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"arm-{name}.npz",
        flux=grid_flux,
        radius=np.asarray(lattice.radius, dtype=np.float64),
        height=np.asarray(lattice.height, dtype=np.float64),
        wall=np.asarray(built["wall"], dtype=np.float64),
        lower_x=lower_x,
        axis=axis,
        efit_lower_x=efit_x,
    )
    (output / f"arm-{name}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print("ARM " + json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default="base,physical,seam,row,warm")
    parser.add_argument("--revision", default="")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    import numpy as np

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64

    bench, row = _corpus_row()
    arms = [item for item in args.arms.split(",") if item]
    print("REVISION=" + args.revision, flush=True)
    print(
        "COMPILATION_CACHE_ROOT="
        + os.environ.get("NOVA_FORWARD_COMPILATION_CACHE_ROOT", "<unset>"),
        flush=True,
    )
    print("PLATFORMS=" + ",".join(str(device) for device in jax.devices()), flush=True)
    records = {}

    def run(name, built, row_used, external_pre):
        equilibrium, termination, converged = _solve_detail(
            bench, built, built["complete"], row_used
        )
        records[name] = _record(
            args.output,
            name,
            built,
            equilibrium,
            row_used,
            built["complete"],
            external_pre,
            _external_sup(built["profile"].operator, built["complete"]),
            termination,
            converged,
        )

    if "cohort" in arms:
        # Instrument control: the cohort's own entry point on the same frame.
        # It must reproduce the banked 137D state; if it does not, the arms
        # below measure this driver rather than the solve.
        result, _fields = bench.solve_frame(row, FRAME, PSEUDO_WALL_EXPANSION)
        records["cohort"] = {
            "arm": "cohort",
            "frame": FRAME,
            "residual": float(result.fixed_point_relative_residual),
            "termination": str(result.solver_termination),
            "converged": bool(result.converged),
            "lower_x_distance_cm": (
                None
                if result.metrics.polished_saddle_to_nearest_efit_x_m is None
                else 100.0 * float(result.metrics.polished_saddle_to_nearest_efit_x_m)
            ),
            "axis_rz_m": list(result.display_map.lattice_saddle_rz_m)
            if result.display_map is not None
            else None,
            "note": "cohort entry point; no terminal flux retained by FrameResult",
        }
        (args.output / "arm-cohort.json").write_text(
            json.dumps(records["cohort"], indent=2) + "\n", encoding="utf-8"
        )
        print("ARM " + json.dumps(records["cohort"]), flush=True)

    if "base" in arms:
        pre = _pre_adapter_external(bench, row, PSEUDO_WALL_EXPANSION)
        run("base", _adapted(bench, row, PSEUDO_WALL_EXPANSION), row, pre)

    if "physical" in arms:
        pre = _pre_adapter_external(bench, row, None)
        run("physical", _adapted(bench, row, None), row, pre)

    if "seam" in arms:
        original = bench._source_current

        def shifted(row_arg, description, names, time_ms):
            return original(row_arg, description, names, time_ms) + SEAM_SHIFT_A_TURN

        bench._source_current = shifted
        try:
            pre = _pre_adapter_external(bench, row, PSEUDO_WALL_EXPANSION)
            built = _adapted(bench, row, PSEUDO_WALL_EXPANSION)
        finally:
            bench._source_current = original
        run("seam", built, row, pre)

    if "row" in arms or "warm" in arms:
        scaled_row = _shipped_scaled(bench, row)

    if "row" in arms:
        pre = _pre_adapter_external(bench, scaled_row, PSEUDO_WALL_EXPANSION)
        run("row", _adapted(bench, scaled_row, PSEUDO_WALL_EXPANSION), row, pre)

    if "warm" in arms:
        from nova.imas.diiid_current import resolve_diiid_currents

        built = _adapted(bench, row, PSEUDO_WALL_EXPANSION)
        scaled_shipped, _time = _shipped(bench, scaled_row)
        scaled_complete = np.asarray(
            resolve_diiid_currents(bench.POLOIDAL_CONDUCTORS, scaled_shipped).current(
                ()
            ),
            dtype=float,
        )
        pre = _external_sup(built["profile"].operator, built["complete"])
        solved = _external_sup(built["profile"].operator, scaled_complete)
        first, first_termination, first_converged = _solve_detail(
            bench, built, built["complete"], row
        )
        records["warm-base"] = _record(
            args.output,
            "warm-base",
            built,
            first,
            row,
            built["complete"],
            pre,
            pre,
            first_termination,
            first_converged,
        )
        second, second_termination, second_converged = _solve_detail(
            bench, built, scaled_complete, row
        )
        records["warm-scaled"] = _record(
            args.output,
            "warm-scaled",
            built,
            second,
            row,
            scaled_complete,
            pre,
            solved,
            second_termination,
            second_converged,
        )

    (args.output / "records.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print("ARMS=" + ",".join(records), flush=True)


if __name__ == "__main__":
    main()
