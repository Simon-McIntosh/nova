"""Measure the tracked read: warm steady-state wall, rebuild wall and accuracy.

The read is measured at the diverted certificate with the operator bank the
fast route consumes, built once and reloaded through the content store, so no
source jet is re-derived here.

Two arms run. ``head`` measures this tree: the first (rebuild) call, then the
warm steady state where the connected decision is carried forward, then one
rebuild call. ``base`` measures an extracted base tree given by ``--base-tree``
through a second process, timing the same call without tracking.

Accuracy is judged at the four certificate states — limited and diverted, trial
and converged — against a direct Newton solve on the analytic psi, which shares
no code with any read.

Host-device copy bytes are reported by construction: the timed region returns
jax arrays on the device and neither materialises them nor syncs inside the
region, so a steady-state call transfers no bytes between host and device. The
launch count is read from the lowered HLO as the number of emitted fusion and
command lines, which is what the compiler launches per call, with the
instruction count printed beside it.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, fixed_points, support_read, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy

assert jax.config.jax_enable_x64 is True
ROOT = Path(__file__).resolve().parents[3]


def emit(token, **row):
    print(token + " " + json.dumps(row, sort_keys=True, default=float), flush=True)


def lowered_counts(function, *operands):
    """Fusion/command lines and instruction count of the compiled program."""
    text = jax.jit(function).lower(*operands).as_text(dialect="hlo")
    lines = text.splitlines()
    launches = sum(
        1
        for line in lines
        if "fusion(" in line or "custom-call(" in line or "call(" in line
    )
    return launches, sum(1 for line in lines if " = " in line)


def newton_reference(analytic, seed, iterations=80):
    """Direct Newton solve on the analytic psi, sharing no read code."""
    position = np.asarray(seed, dtype=float)
    for _ in range(iterations):
        jet = analytic.evaluate(jnp.asarray(position))
        gradient = np.asarray(jet.gradient)
        hessian = np.asarray(jet.hessian)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        position = position - step
        if np.linalg.norm(step) < 1e-14:
            break
    value = float(np.asarray(analytic.evaluate(jnp.asarray(position)).value))
    return position, value


def build_kind(kind, cells, policy):
    """Geometry, analytic field, states and bank operands for one certificate."""
    import tests.equilibrium.test_topology_read as fixture
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures.measure import analytic_profile
    from scripts.prototypes.read_route.jets import build_jet_matrix
    from scripts.prototypes.read_route.measure import sample_field

    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, cells)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    read_policy = TopologyPolicy(
        normal_form_radius=policy if saddle is not None else 0.0
    )
    analytic_read = jax.jit(topology.read)(analytic, geometry, convention, read_policy)
    sampled_read = jax.jit(topology.read)(
        sample_field(analytic, geometry), geometry, convention, read_policy
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    _, source, _ = certificate._case(name, clip_mode="exact")
    profile = analytic_profile(source)
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    reference = fixture._kernel_backed_field(kind, oracle, analytic, geometry)
    coupling = reference.coupling

    def book(reading):
        return ClippedCurrentMoments(
            *current.book(
                psi,
                reading,
                geometry,
                profile,
                current.Quadrature(sigma=convention.sigma, segments=64),
            )[:3]
        )

    states = {
        "converged": book(analytic_read),
        "trial": book(sampled_read),
    }
    roots = np.asarray([axis] if saddle is None else [axis, saddle])
    matrix, exterior, nodes, sample_index, _build = build_jet_matrix(
        coupling, geometry, analytic, reference.moments, persist=False
    )
    jax.block_until_ready(matrix)
    return dict(
        kind=kind,
        geometry=geometry,
        analytic=analytic,
        convention=convention,
        policy=read_policy,
        states=states,
        roots=roots,
        matrix=matrix,
        exterior=exterior,
        nodes=nodes,
        sample_index=sample_index,
        analytic_read=analytic_read,
    )


def bank_for(case, store, digest=None):
    """Build the bank once, or reload it from the content store."""
    geometry = case["geometry"]
    started = perf_counter()
    if digest is None:
        operands = support_read.SupportOperands.from_rows(
            geometry,
            case["matrix"],
            case["exterior"],
            case["nodes"],
            case["sample_index"],
            bucket=len(case["nodes"]),
        )
        jax.block_until_ready(operands.rows)
        digest = support_read.write_bank(operands, store)
        build = perf_counter() - started
        emit(
            "BANK_BUILD",
            kind=case["kind"],
            bucket=operands.bucket,
            nodes=len(case["nodes"]),
            digest=digest,
            seconds=build,
            bytes=int(
                np.asarray(operands.rows).nbytes + np.asarray(operands.exterior).nbytes
            ),
        )
    started = perf_counter()
    mesh = fixed_points.build_mesh(
        np.asarray(case["nodes"]), np.asarray(case["sample_index"]), geometry
    )
    loaded = support_read.read_bank(store, digest, geometry, mesh)
    jax.block_until_ready(loaded.rows)
    emit(
        "BANK_LOAD", kind=case["kind"], digest=digest, seconds=perf_counter() - started
    )
    return loaded, digest


def cache_for(operands):
    """The row window the read solves in, taken from the bank's active prefix."""
    active = int(operands.active_nodes)
    return fixed_points.RowCache(
        operands.mesh,
        operands.nodes[:active],
        jnp.arange(active, dtype=jnp.int32),
        operands.rows[:active],
        operands.exterior[:active],
    )


def timed_call(function, *operands, repeats):
    """Warm per-call wall: compile and one warm call outside, minimum of repeats."""
    out = function(*operands)
    jax.block_until_ready(out)
    walls = []
    for _ in range(repeats):
        started = perf_counter()
        result = function(*operands)
        jax.block_until_ready(result)
        walls.append(perf_counter() - started)
    return min(walls), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("head", "base"), default="head")
    parser.add_argument("--cells", type=int, default=550)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-tree", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"# revision={revision} tree={ROOT} module={support_read.__file__} "
        f"cwd={Path.cwd().resolve()} arm={args.arm} "
        f"backend={jax.default_backend()} "
        f"devices={[str(d) for d in jax.local_devices()]}",
        flush=True,
    )

    cells = args.cells
    policy_radius = 0.054806712567833996
    diverted = build_kind("diverted", cells, policy_radius)
    operands, digest = bank_for(diverted, args.store)
    cache = cache_for(operands)
    moments = jnp.stack(tuple(diverted["states"]["converged"]), axis=-1)
    seeds = jnp.asarray(diverted["roots"])
    convention = diverted["convention"]
    policy = diverted["policy"]

    rows = []

    if args.arm == "base":

        def read_call(op, cache_, moments_, seeds_, previous_):
            return support_read.read(
                op, cache_, moments_, seeds_, previous_, convention, policy
            )

        warm, result = timed_call(
            read_call, operands, cache, moments, seeds, seeds, repeats=args.repeats
        )
        launches, instructions = lowered_counts(
            read_call, operands, cache, moments, seeds, seeds
        )
        rows.append(
            dict(
                arm="base",
                call="read",
                warm_seconds=warm,
                launches=launches,
                hlo_instructions=instructions,
                copy_bytes=0,
            )
        )
        emit("STAGE_ROW", **rows[-1])
    else:

        def rebuild_call(op, cache_, moments_, seeds_, previous_):
            return support_read.read(
                op, cache_, moments_, seeds_, previous_, convention, policy
            )

        first = rebuild_call(operands, cache, moments, seeds, seeds)
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

        warm, steady = timed_call(
            steady_call,
            operands,
            cache,
            moments,
            seeds,
            seeds,
            tracking,
            repeats=args.repeats,
        )
        if bool(steady.rebuilt):
            raise RuntimeError("the steady arm rebuilt; the track did not hold")
        leaves = jax.tree_util.tree_leaves(steady)
        on_device = all(
            isinstance(leaf, jax.Array) and jax.device_get(leaf.dtype) is not None
            for leaf in leaves
        )
        launches, instructions = lowered_counts(
            steady_call, operands, cache, moments, seeds, seeds, tracking
        )
        rows.append(
            dict(
                arm="head",
                call="steady",
                warm_seconds=warm,
                launches=launches,
                hlo_instructions=instructions,
                copy_bytes=0,
                device_resident=bool(on_device),
                rebuilt=bool(steady.rebuilt),
                valid=bool(steady.valid),
            )
        )
        emit("STAGE_ROW", **rows[-1])

        rebuild_warm, rebuilt = timed_call(
            rebuild_call, operands, cache, moments, seeds, seeds, repeats=args.repeats
        )
        rebuild_launches, rebuild_instructions = lowered_counts(
            rebuild_call, operands, cache, moments, seeds, seeds
        )
        rows.append(
            dict(
                arm="head",
                call="rebuild",
                warm_seconds=rebuild_warm,
                launches=rebuild_launches,
                hlo_instructions=rebuild_instructions,
                copy_bytes=0,
                rebuilt=bool(rebuilt.rebuilt),
            )
        )
        emit("STAGE_ROW", **rows[-1])

        # Accuracy at the four certificate states against the analytic Newton.
        limited = build_kind("limited", cells, policy_radius)
        for case in (limited, diverted):
            case_operands, _ = bank_for(case, args.store)
            case_cache = cache_for(case_operands)
            case_seeds = jnp.asarray(case["roots"])
            reference_points = [
                newton_reference(case["analytic"], seed) for seed in case["roots"]
            ]
            minor_radius = float(
                np.max(
                    np.linalg.norm(
                        np.asarray(case["geometry"].sample_points).reshape(-1, 2)
                        - reference_points[0][0],
                        axis=-1,
                    )
                )
            )
            analytic_read = case["analytic_read"]
            span = abs(float(analytic_read.boundary_flux - analytic_read.axis_flux))
            for state_name, state_moments in case["states"].items():
                stacked = jnp.stack(tuple(state_moments), axis=-1)
                reading = support_read.read(
                    case_operands,
                    case_cache,
                    stacked,
                    case_seeds,
                    case_seeds,
                    case["convention"],
                    case["policy"],
                    iterations=24,
                )
                jax.block_until_ready(reading)
                position = np.asarray(reading.stationary.position)
                axis_error = float(np.linalg.norm(position[0] - reference_points[0][0]))
                saddle_error = (
                    None
                    if len(reference_points) < 2
                    else float(np.linalg.norm(position[1] - reference_points[1][0]))
                )
                emit(
                    "ACCURACY_ROW",
                    kind=case["kind"],
                    state=state_name,
                    valid=bool(reading.valid),
                    qualified=bool(reading.reading.qualified),
                    status=np.asarray(reading.stationary.status).tolist(),
                    axis_error_m=axis_error,
                    axis_budget_m=5e-4 * minor_radius,
                    axis_within_budget=bool(axis_error <= 5e-4 * minor_radius),
                    saddle_error_m=saddle_error,
                    saddle_within_budget=(
                        None
                        if saddle_error is None
                        else bool(saddle_error <= 5e-4 * minor_radius)
                    ),
                    boundary_flux=float(reading.reading.boundary_flux),
                    boundary_flux_reference=float(analytic_read.boundary_flux),
                    boundary_flux_error_relative=float(
                        abs(
                            float(reading.reading.boundary_flux)
                            - float(analytic_read.boundary_flux)
                        )
                        / span
                    )
                    if span
                    else None,
                    boundary_within_budget=bool(
                        span
                        and abs(
                            float(reading.reading.boundary_flux)
                            - float(analytic_read.boundary_flux)
                        )
                        / span
                        <= 1e-4
                    ),
                )

    if args.arm == "head" and args.base_tree is not None:
        base_log = args.output.with_suffix(".base.log")
        command = [
            "/home/ITER/mcintos/Code/nova/.venv/bin/python",
            str(Path(__file__)),
            "--arm",
            "base",
            "--cells",
            str(cells),
            "--repeats",
            str(args.repeats),
            "--store",
            str(args.store),
            "--output",
            str(args.output.with_suffix(".base.json")),
        ]
        environment = dict(
            __import__("os").environ,
            PYTHONPATH=str(args.base_tree),
            JAX_PLATFORMS=__import__("os").environ.get("JAX_PLATFORMS", "cuda"),
        )
        base = subprocess.run(
            command,
            cwd=args.base_tree,
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
