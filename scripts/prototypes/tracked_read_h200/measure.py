"""Measure the tracked read's warm wall, its launches by stage and its accuracy.

The diverted 550-cell operator bank is loaded through ``support_read.read_bank``
from the shared store the fast route consumes, so no source jet is re-derived
here. Every row is the minimum of warm repeats of a compiled step, so no compile
is inside the timed region.

The launch table splits the steady call into the stationary solve
(``fixed_points.solve``), the remainder the tracked read pays on every call, and
the rebuild branch that only a cell or ownership change pays.

Accuracy is judged at the four certificate states — limited and diverted, trial
and converged — against a direct Newton solve on the analytic psi, which shares
no code with any read.

The header names the device the run is on, and every row is that device's
measurement; a wall measured on another device is never reported as this one's.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_points, support_read

ROOT = Path(__file__).resolve().parents[3]
# Appended, never prepended: a base arm runs this file from the head tree with
# PYTHONPATH naming an extracted base tree, and every module it imports must
# come from that tree rather than from the head tree the script lives in.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.prototypes.tracked_read.measure import (  # noqa: E402
    build_kind,
    cache_for,
    emit,
    lowered_counts,
    newton_reference,
    timed_call,
)


def header(args):
    try:
        revision = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[3]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        revision = "archive:" + os.environ.get("BASE_SHA", "unknown")
    devices = [str(d) for d in jax.local_devices()]
    device = jax.local_devices()[0] if jax.local_devices() else None
    emit(
        "HEADER",
        revision=revision,
        arm=args.arm,
        tree=str(ROOT),
        module=support_read.__file__,
        fixed_points_module=fixed_points.__file__,
        cwd=str(Path.cwd().resolve()),
        backend=jax.default_backend(),
        devices=devices,
        device_device_kind=getattr(device, "device_kind", None),
        cells=args.cells,
        repeats=args.repeats,
        digest=args.digest,
    )


def loaded_bank(case, store, digest):
    """Bind a published bank to this case, or build one when none is published.

    A published bank carries one certificate and one cell count, so a case the
    store does not hold is assembled from its own operands rather than bound to
    a bank of another state.
    """
    if digest is None:
        operands = support_read.SupportOperands.from_rows(
            case["geometry"],
            case["matrix"],
            case["exterior"],
            case["nodes"],
            case["sample_index"],
            bucket=len(np.asarray(case["nodes"])),
        )
        jax.block_until_ready(operands.rows)
        emit(
            "BANK_BUILD",
            kind=case["kind"],
            nodes=int(np.asarray(operands.active_nodes)),
            bucket=operands.bucket,
        )
        return operands
    mesh = fixed_points.build_mesh(
        np.asarray(case["nodes"]), np.asarray(case["sample_index"]), case["geometry"]
    )
    operands = support_read.read_bank(store, digest, case["geometry"], mesh)
    jax.block_until_ready(operands.rows)
    emit(
        "BANK_LOAD",
        kind=case["kind"],
        digest=digest,
        nodes=int(np.asarray(operands.active_nodes)),
        bucket=operands.bucket,
        bytes=int(np.asarray(operands.rows).nbytes),
    )
    return operands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("head", "base"), default="head")
    parser.add_argument("--cells", type=int, default=550)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--digest",
        default="sha256:6ebe02e02ae9ed8646077b3116596fa08ff0f18cb7f17d2566f91f779b4c17be",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-tree", type=Path, default=None)
    parser.add_argument("--accuracy", action="store_true")
    args = parser.parse_args()

    header(args)
    policy_radius = 0.054806712567833996
    diverted = build_kind("diverted", args.cells, policy_radius)
    operands = loaded_bank(diverted, args.store, args.digest)
    cache = cache_for(operands)
    moments = diverted["states"]["converged"]
    seeds = jnp.asarray(diverted["roots"])
    previous = seeds
    convention = diverted["convention"]
    policy = diverted["policy"]
    pitch = jnp.median(operands.geometry.pitch)
    rows = []

    def read_call(op, cache_, moments_, seeds_, previous_):
        return support_read.read(
            op, cache_, moments_, seeds_, previous_, convention, policy
        )

    if args.arm == "base":
        warm, _ = timed_call(
            read_call, operands, cache, moments, seeds, previous, repeats=args.repeats
        )
        launches, instructions = lowered_counts(
            read_call, operands, cache, moments, seeds, previous
        )
        rows.append(
            dict(
                arm="base",
                call="read",
                warm_ms=1e3 * warm,
                launches=launches,
                hlo_instructions=instructions,
                copy_bytes=0,
            )
        )
        emit("STAGE_ROW", **rows[-1])
    else:
        first = read_call(operands, cache, moments, seeds, previous)
        jax.block_until_ready(first)
        tracking = support_read.tracking_of(first)

        def steady_call(op, cache_, moments_, seeds_, previous_, track):
            return support_read.read(
                op,
                cache_,
                moments_,
                seeds_,
                previous_,
                convention,
                policy,
                tracking=track,
            )

        def solve_call(cache_, moments_, seeds_, previous_):
            return fixed_points.solve(
                cache_,
                jnp.stack(tuple(moments_), axis=-1),
                seeds_,
                previous_,
                pitch,
                -1.0,
                iterations=12,
            )

        warm, steady = timed_call(
            steady_call,
            operands,
            cache,
            moments,
            seeds,
            previous,
            tracking,
            repeats=args.repeats,
        )
        if bool(steady.rebuilt):
            raise RuntimeError("the steady arm rebuilt; the track did not hold")
        total_launches, total_instructions = lowered_counts(
            steady_call, operands, cache, moments, seeds, previous, tracking
        )
        solve_warm, _ = timed_call(
            solve_call, cache, moments, seeds, previous, repeats=args.repeats
        )
        solve_launches, solve_instructions = lowered_counts(
            solve_call, cache, moments, seeds, previous
        )
        rebuild_warm, _ = timed_call(
            read_call, operands, cache, moments, seeds, previous, repeats=args.repeats
        )
        rebuild_launches, rebuild_instructions = lowered_counts(
            read_call, operands, cache, moments, seeds, previous
        )
        leaves = jax.tree_util.tree_leaves(steady)
        on_device = all(isinstance(leaf, jax.Array) for leaf in leaves)
        rows.extend(
            [
                dict(
                    arm="head",
                    call="steady",
                    warm_ms=1e3 * warm,
                    launches=total_launches,
                    hlo_instructions=total_instructions,
                    copy_bytes=0,
                    device_resident=bool(on_device),
                    rebuilt=bool(steady.rebuilt),
                    valid=bool(steady.valid),
                ),
                dict(
                    arm="head",
                    call="stationary-solve",
                    warm_ms=1e3 * solve_warm,
                    launches=solve_launches,
                    hlo_instructions=solve_instructions,
                    copy_bytes=0,
                ),
                dict(
                    arm="head",
                    call="rebuild",
                    warm_ms=1e3 * rebuild_warm,
                    launches=rebuild_launches,
                    hlo_instructions=rebuild_instructions,
                    copy_bytes=0,
                ),
            ]
        )
        for row in rows:
            emit("STAGE_ROW", **row)

    if args.arm == "head" and args.accuracy:
        limited = build_kind("limited", args.cells, policy_radius)
        for case in (limited, diverted):
            case_digest = args.digest if case["kind"] == "diverted" else None
            case_operands = loaded_bank(case, args.store, case_digest)
            case_cache = cache_for(case_operands)
            case_seeds = jnp.asarray(case["roots"])
            references = [
                newton_reference(case["analytic"], seed) for seed in case_seeds
            ]
            minor_radius = float(
                np.max(
                    np.linalg.norm(
                        np.asarray(case["geometry"].sample_points).reshape(-1, 2)
                        - references[0][0],
                        axis=-1,
                    )
                )
            )
            from tests.test_tracked_read_accuracy import _psi, analytic_boundary

            boundary_flux_reference = analytic_boundary(
                case, [reference[0] for reference in references]
            )[1]
            span = abs(
                boundary_flux_reference - _psi(case["analytic"], references[0][0])
            )
            for state_name, state_moments in case["states"].items():
                reading = support_read.read(
                    case_operands,
                    case_cache,
                    state_moments,
                    case_seeds,
                    case_seeds,
                    case["convention"],
                    case["policy"],
                    iterations=24,
                )
                jax.block_until_ready(reading)
                position = np.asarray(reading.stationary.position)
                axis_error = float(np.linalg.norm(position[0] - references[0][0]))
                saddle_error = (
                    None
                    if len(references) < 2
                    else float(np.linalg.norm(position[1] - references[1][0]))
                )
                boundary = float(reading.reading.boundary_flux)
                boundary_error = abs(boundary - boundary_flux_reference) / span
                emit(
                    "ACCURACY_ROW",
                    kind=case["kind"],
                    state=state_name,
                    valid=bool(reading.valid),
                    qualified=bool(reading.reading.qualified),
                    status=np.asarray(reading.stationary.status).tolist(),
                    iterations=np.asarray(reading.stationary.iterations).tolist(),
                    axis_error_m=axis_error,
                    axis_budget_m=5e-4 * minor_radius,
                    axis_within_budget=bool(axis_error <= 5e-4 * minor_radius),
                    saddle_error_m=saddle_error,
                    saddle_within_budget=(
                        None
                        if saddle_error is None
                        else bool(saddle_error <= 5e-4 * minor_radius)
                    ),
                    boundary_flux_error_relative=boundary_error,
                    boundary_within_budget=bool(boundary_error <= 1e-4),
                )

    if args.arm == "head" and args.base_tree is not None:
        base_log = args.output.with_suffix(".base.log")
        environment = dict(
            os.environ,
            PYTHONPATH=str(args.base_tree),
            JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cuda"),
        )
        command = [
            "/home/ITER/mcintos/Code/nova/.venv/bin/python",
            str(Path(__file__).resolve()),
            "--arm",
            "base",
            "--cells",
            str(args.cells),
            "--repeats",
            str(args.repeats),
            "--store",
            str(args.store),
            "--digest",
            args.digest,
            "--output",
            str(args.output.with_suffix(".base.json")),
        ]
        base = subprocess.run(
            command,
            cwd=str(args.base_tree),
            env=environment,
            capture_output=True,
            text=True,
        )
        base_log.write_text(base.stdout + base.stderr)
        emit("BASE_ARM", exit_status=base.returncode, log=str(base_log))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print("RUN_COMPLETE " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
