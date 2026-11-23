"""Run the fit inside the equilibrium iteration over drifting-sensor sequences.

The nova column of the drifting-sensor study is the held-map pass of
:class:`nova.equilibrium.flux_iteration.FluxIterationFit`, not the nested
first-order class: the nested form costs 805 to 1014 s per slice and cannot be
batched, while the held-map pass converges the known ITER state in one pass.

Rows, noise, the plasma-current draw and the sensor set are the study's own
(``study.py``), so both codes reconstruct identical rows slice for slice. The
generating state and its noise-free rows are prepared once; every shard then
reuses the persisted generating flux as its starting map, which is the map the
nested class starts from, so a shard pays the machine load and the sensor
geometry but never a second generating solve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import study  # noqa: E402

STUDY = study.STUDY


def cell_rows(case: dict):
    """Read a per-cell plasma current vector into the measured row order."""
    module = case["module"]
    biot = case["magnetic_source"].biot
    sensors = case["sensors"]

    def read(currents):
        biot.sloc["Ic"] = np.asarray(currents, dtype=np.float64)
        return np.asarray(module._physical_sensor_rows(biot, sensors), dtype=np.float64)

    return read


def reference_q95(case: dict) -> float:
    """The generating state's q95 from the one shared calculator."""
    generating = case["generated"].terminal_state
    return study.q95_of(
        study.nova_flux_map(case, generating),
        generating.topology.axis,
        generating.topology.axis_flux,
        generating.topology.boundary_flux,
        case["reference"]["b0"],
    )


def prepare(args) -> int:
    """Solve the generating state once and persist its rows, reference and seed."""
    case = study.build_case(args.cells, reuse_rows=args.reuse_rows)
    study.write_prepare(case)
    reference = case["reference"]
    reference["q95"] = reference_q95(case)
    (STUDY / "generating.json").write_text(json.dumps(reference, indent=2))
    generated = case["generated"].terminal_state
    np.savez(
        STUDY / "generated.npz",
        flux=np.asarray(generated.flux, dtype=float),
        cell_current=np.asarray(generated.cell_current, dtype=float),
    )
    print(
        "PREPARED q95=%.6f axis=%s span=%.6f"
        % (
            reference["q95"],
            np.round(np.asarray(reference["axis"], float), 6).tolist(),
            reference["flux_span"],
        ),
        flush=True,
    )
    return 0


def load_case(cells: int = 500) -> dict:
    """The generating state's geometry, its persisted rows and its seed map."""
    case = study.geometry_case(cells)
    rows = np.load(STUDY / "rows.npz")
    generated = np.load(STUDY / "generated.npz")
    loops = np.asarray(case["sensors"].flux_loop, dtype=bool)
    plasma = np.asarray(rows["plasma"], dtype=float)
    case.update(
        reference=json.loads((STUDY / "generating.json").read_text()),
        seed_flux=np.asarray(generated["flux"], dtype=float),
        total=np.asarray(rows["total"], dtype=float),
        vacuum=np.asarray(rows["vacuum"], dtype=float),
        plasma=plasma,
        sigma_loop=float(np.sqrt(np.mean(plasma[loops] ** 2))),
        sigma_probe=float(np.sqrt(np.mean(plasma[~loops] ** 2))),
    )
    return case


def record_of(case: dict, fitted, index: int) -> dict:
    """One slice's reconstruction, in the columns the reduction reads."""
    topology = fitted.terminal_state.topology
    if topology is None:
        raise RuntimeError(
            "the terminal map carries no topology: %s" % fitted.topology_error
        )
    flux_map = study.nova_flux_map(case, fitted.terminal_state)
    axis = np.asarray(topology.axis, dtype=float)
    q95 = study.q95_of(
        flux_map,
        axis,
        topology.axis_flux,
        topology.boundary_flux,
        case["reference"]["b0"],
    )
    x_points = np.asarray(topology.x_point, dtype=float).reshape(-1, 2)
    return dict(
        slice=index,
        axis=axis.tolist(),
        flux_axis=float(topology.axis_flux),
        flux_boundary=float(topology.boundary_flux),
        flux_span=float(topology.boundary_flux - topology.axis_flux),
        plasma_current=float(np.sum(fitted.terminal_state.cell_current)),
        q95=q95,
        x_points=[
            [float(point[0]), float(point[1])]
            for point in x_points
            if np.all(np.isfinite(point))
        ],
        coefficients=np.asarray(fitted.coefficients, dtype=float).ravel().tolist(),
        passes=len(fitted.passes),
        flux_evaluations=int(fitted.flux_evaluations),
        terminal_misfit=float(fitted.terminal_misfit),
        last_flux_update=float(fitted.last_flux_update),
        stopped_improving_pass=fitted.stopped_improving_pass,
        stalled_pass=fitted.stalled_pass,
        rejected_pass=fitted.rejected_pass,
        axis_lost_pass=fitted.axis_lost_pass,
        topology_error=fitted.topology_error,
        converged=bool(fitted.converged),
        wall_seconds=float(fitted.wall_seconds),
        device=str(fitted.device),
    )


def fit_sequence(args) -> int:
    """Fit every slice of the shard's sequences with the held-map pass."""
    from nova.equilibrium.flux_iteration import FluxIterationFit

    study.serial_pools()
    case = load_case(args.cells)
    reference = case["reference"]
    sigma = np.r_[
        np.full(24, case["sigma_loop"]),
        np.full(48, case["sigma_probe"]),
    ]
    sigma_ip = args.factor * abs(float(reference["plasma_current"]))
    # The row set the fit reads. ``loops`` fits the 24 flux loops alone, which
    # is the set imas-efit's arm reconstructs from, so the two codes are
    # compared on one information set rather than two.
    mask = np.ones(case["sensors"].number, bool)
    if args.rows == "loops":
        mask = np.asarray(case["sensors"].flux_loop, dtype=bool)
    fitter = FluxIterationFit(
        case["profile"],
        case["sensors"],
        cell_rows(case),
        coefficient_basis=case["basis"],
        prescribed_current=getattr(case["request"], "prescribed_current", None),
    )
    suffix = "" if args.shards == 1 else "-shard%d" % args.shard
    tag = "" if args.rows == "all" else "-" + args.rows
    slots = (
        range(args.shard, args.slices, args.shards)
        if args.shards > 1
        else range(args.slices)
    )
    for sequence in filter(None, (args.sequences or "").split(",")):
        noise = study.noise_columns(
            sequence, args.slices, sigma, args.seed, args.factor
        )
        rng = np.random.default_rng(args.seed + 991)
        current = reference["plasma_current"] + sigma_ip * rng.standard_normal(
            args.slices
        )
        path = STUDY / (
            "nova-%s-%d-%.4f%s%s.jsonl"
            % (sequence, args.slices, args.factor, tag, suffix)
        )
        grids: list[np.ndarray] = []
        indices: list[int] = []
        with path.open("w") as stream:
            for index in slots:
                measured = case["total"] + noise[index]
                measurement = study.fitted_measurement(
                    case, measured, sigma, current[index], mask=mask
                )
                function = study.guard("nova-%s-%.4f" % (sequence, args.factor), index)
                with function:
                    fitted = fitter.fit(
                        measurement,
                        case["seed_flux"],
                        plasma_current_scale=sigma_ip,
                        max_passes=args.max_passes,
                        flux_tolerance=args.flux_tolerance,
                        step_rule=args.step_rule,
                    )
                    record = record_of(case, fitted, index)
                    record["plasma_current_measured"] = float(current[index])
                    record["rows"] = args.rows
                    grids.append(
                        np.asarray(
                            study.nova_flux_map(case, fitted.terminal_state).flux,
                            dtype=float,
                        )
                    )
                    indices.append(index)
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    study.land("nova-%s-%.4f" % (sequence, args.factor), record)
        maps = STUDY / "maps"
        maps.mkdir(parents=True, exist_ok=True)
        np.savez(
            maps
            / (
                "nova-%s-%d-%.4f%s%s.npz"
                % (sequence, args.slices, args.factor, tag, suffix)
            ),
            radius=study.GRID_RADIUS,
            height=study.GRID_HEIGHT,
            flux=np.asarray(grids, dtype=float),
            slice=np.asarray(indices, dtype=int),
        )
        print("NOVA_ITERATION -> %s slices=%d" % (path, len(indices)), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--cells", type=int, default=500)
    prep.add_argument("--reuse-rows", action="store_true")
    prep.add_argument("--sequences", default="")
    prep.set_defaults(handler=prepare)
    fit = commands.add_parser("fit")
    fit.add_argument("--slices", type=int, default=200)
    fit.add_argument("--factor", type=float, default=study.REFERENCE_FACTOR)
    fit.add_argument("--sequences", default="white,drift")
    fit.add_argument("--seed", type=int, default=20261118)
    fit.add_argument("--cells", type=int, default=500)
    fit.add_argument("--rows", default="all", choices=("all", "loops"))
    fit.add_argument("--shard", type=int, default=0)
    fit.add_argument("--shards", type=int, default=1)
    fit.add_argument("--max-passes", type=int, default=48)
    fit.add_argument("--flux-tolerance", type=float, default=1.0e-10)
    fit.add_argument("--step-rule", default="coupled")
    fit.set_defaults(handler=fit_sequence)
    arguments = parser.parse_args(argv)
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())