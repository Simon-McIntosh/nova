"""Attribute the present-route forward-solve producer's memory to named owners.

Runs the DIII-D forward-gs producer through one frame's profile build and one
solve on the present route. Each phase prints a PHASE-BEGIN line and flushes it
before the work starts, so a phase that dies silently is still named, then a
PHASE-END line with its wall, host RSS before and after, and peak RSS. The
solve is split into trace-and-lower, compile, first execute and warm execute so
memory is attributed to the stage that holds it rather than to "the solve".

The ledger JSON is rewritten atomically at every phase boundary, and once more
on SIGTERM, so a run stopped at its time limit or killed for its memory leaves
every phase, census and analysis it had reached on disk instead of nothing. The
file names the phase that was still in flight and carries an explicit
``completed`` flag, so a partial ledger is distinguishable from a whole one
without reading the log.

Every number carries the rung and device it was measured on.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
RSS_FIELDS = ("VmRSS", "VmHWM", "VmPeak")


def host_rss() -> dict[str, int]:
    values: dict[str, int] = {}
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            name, _, rest = line.partition(":")
            if name in RSS_FIELDS:
                values[name] = int(rest.split()[0]) * 1024
    values["ru_maxrss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return values


def jax_memory() -> dict:
    try:
        import jax
    except Exception as error:
        return {"import": f"unavailable: {type(error).__name__}: {error}"}
    memory: dict = {"backend": jax.default_backend()}
    memory["devices"] = [str(device) for device in jax.local_devices()]
    try:
        pass
    except Exception:
        pass
    try:
        stats = jax.local_devices()[0].memory_stats()
        memory["device_memory_stats"] = (
            None if stats is None else {k: int(v) for k, v in stats.items()}
        )
    except Exception as error:
        memory["device_memory_stats"] = f"unavailable: {type(error).__name__}: {error}"
    try:
        live = jax.live_arrays()
        memory["live_array_bytes"] = int(sum(a.size * a.dtype.itemsize for a in live))
        memory["live_array_count"] = len(live)
    except Exception as error:
        memory["live_array_bytes"] = f"unavailable: {type(error).__name__}: {error}"
    return memory


def _owner_names(target: object) -> list[str]:
    names: list[str] = []
    for referrer in gc.get_referrers(target):
        attributes = getattr(referrer, "__dict__", None)
        if isinstance(attributes, dict):
            for key, value in list(attributes.items()):
                if value is target:
                    names.append(f"{type(referrer).__name__}.{key}")
                elif isinstance(value, (list, tuple)):
                    for index, item in enumerate(value):
                        if item is target:
                            names.append(f"{type(referrer).__name__}.{key}[{index}]")
        elif isinstance(referrer, dict):
            for key, value in list(referrer.items()):
                if value is target and isinstance(key, str):
                    names.append(f"cell:{key}")
        if len(names) >= 3:
            break
    return names


def largest_arrays(limit: int = 0, floor: int = 1 << 1) -> dict:
    import jax
    import numpy as np

    seen: dict[int, dict] = {}
    for obj in gc.get_objects():
        if id(obj) in seen or not isinstance(obj, (np.ndarray, jax.Array)):
            continue
        try:
            nbytes = int(obj.size * obj.dtype.itemsize)
        except (TypeError, ValueError, AttributeError):
            continue
        if nbytes < floor:
            continue
        seen[id(obj)] = {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "bytes": nbytes,
            "kind": type(obj).__name__,
            "owners": _owner_names(obj),
        }
    ordered = sorted(seen.values(), key=lambda row: row["bytes"], reverse=True)
    return {
        "top": ordered[:limit],
        "total_enumerated_bytes": sum(row["bytes"] for row in ordered),
        "enumerated_arrays": len(ordered),
        "floor_bytes": floor,
    }


def persist(state: dict) -> Path | None:
    """Write the ledger as it stands, atomically, so a stopped run keeps it."""
    directory = state.get("output_path")
    if directory is None:
        return None
    target = directory / f"ledger-{state['file_arm']}.json"
    scratch = directory / f".ledger-{state['file_arm']}.part.json"
    phases = state["phases"]
    if phases:
        peak = [row for row in phases if row.get("rss_after")]
        if peak:
            peak_row = max(peak, key=lambda row: row["rss_after"].get("ru_maxrss", 0))
            state["peak_rss"] = peak_row["phase"]
            state["peak_rss_bytes"] = peak_row["rss_after"].get("ru_maxrss")
    state["phases_completed"] = [row["phase"] for row in phases]
    scratch.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n")
    scratch.replace(target)
    print(
        f"LEDGER {target} phases={len(phases)} completed={state.get('completed')}",
        flush=True,
    )
    return target


def census(state: dict, label: str, limit: int) -> None:
    """Record the largest live arrays, naming the census before it starts."""
    state["census_inflight"] = label
    persist(state)
    state["arrays"][label] = largest_arrays(limit)
    state["census_inflight"] = None
    persist(state)


def begin(state: dict, name: str) -> dict:
    marker = {"phase": name, "rss_before": host_rss(), "jax_before": jax_memory()}
    print(f"PHASE-BEGIN {name} rss={marker['rss_before'].get('VmRSS')}", flush=True)
    state["inflight"] = marker
    persist(state)
    return marker


def end(state: dict, name: str, started: float, extra: dict | None = None) -> None:
    row = {
        "phase": name,
        "seconds": perf_counter() - started,
        "rss_before": state.get("inflight", {}).get("rss_before"),
        "rss_after": host_rss(),
        "jax": jax_memory(),
    }
    if extra:
        row.update(extra)
    state["phases"].append(row)
    state["inflight"] = None
    print(
        f"PHASE-END {name} seconds={row['seconds']:.3f} "
        f"rss_before={row['rss_before'].get('VmRSS') if row['rss_before'] else None} "
        f"rss_after={row['rss_after'].get('VmRSS')} "
        f"peak={row['rss_after'].get('ru_maxrss')}",
        flush=True,
    )
    persist(state)


def phase(state: dict, name: str):
    """Context manager measuring one phase."""

    class _Phase:
        def __enter__(self):
            self.started = perf_counter()
            self.marker = begin(state, name)
            return self

        def __exit__(self, kind, value, trace):
            end(state, name, self.started)
            return False

    return _Phase()


def import_peak(python: str, code: str, env: dict) -> dict:
    started = perf_counter()
    result = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT)
    )
    payload: dict = {}
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        payload = {"parse": "failed", "stdout": result.stdout.strip()[-400:]}
    payload.update(
        {
            "returncode": result.returncode,
            "seconds": perf_counter() - started,
            "stderr_tail": result.stderr.strip()[-800:],
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-artifact-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="gpu", choices=("cpu", "gpu", "both"))
    parser.add_argument("--array-limit", type=int, default=30)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    state: dict = {
        "arm": arguments.arm,
        "output_path": arguments.output.resolve(),
        "file_arm": arguments.arm,
        "completed": False,
        "inflight": None,
        "phases": [],
        "arrays": {},
        "banner": {
            "revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "tree": str(ROOT),
            "cwd": str(Path.cwd().resolve()),
            "python": sys.executable,
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.uname().nodename,
        },
    }
    print(f"BANNER {json.dumps(state['banner'], sort_keys=True)}", flush=True)
    if arguments.arm == "both":
        child_env = dict(os.environ)
        child_env["JAX_PLATFORMS"] = "cpu"
        started_child = perf_counter()
        child = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--machine-artifact-cache",
                str(arguments.machine_artifact_cache),
                "--output",
                str(arguments.output),
                "--arm",
                "cpu",
            ],
            env=child_env,
            cwd=str(ROOT),
        )
        print(
            f"CPU_ARM returncode={child.returncode} "
            f"seconds={perf_counter() - started_child}",
            flush=True,
        )
        arguments.arm = "gpu"
    state["file_arm"] = arguments.arm

    def persist_on_stop(signum, frame):
        state["stop_signal"] = signal.Signals(signum).name
        persist(state)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, persist_on_stop)
    persist(state)

    from nova.jax.config import configure_dtypes

    with phase(state, f"{arguments.arm}:import-nova-jax"):
        configure_dtypes()
        import jax

        assert jax.config.jax_enable_x64 is True
    state["x64"] = True

    with phase(state, f"{arguments.arm}:producer-import"):
        import benchmarks.diiid_forward_gs_match as producer

    if arguments.arm == "gpu":
        with phase(state, f"{arguments.arm}:spawn-pool-worker-rss"):
            state["spawn_pool"] = {
                "bare_import": import_peak(
                    sys.executable,
                    "import json,resource\n"
                    "import benchmarks.diiid_forward_gs_match\n"
                    "print(json.dumps({'ru_maxrss_bytes': "
                    "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}))\n",
                    dict(os.environ),
                )
            }

    with phase(state, f"{arguments.arm}:machine-description"):
        data = producer.DEFAULT_DATA
        paths = sorted(Path(data).glob("*.parquet"))
        if not paths:
            raise RuntimeError(f"no parquet rows under {data}")
        row = producer._read(
            paths[0],
            producer._LABEL_COLUMNS
            + producer._GEOMETRY_COLUMNS
            + producer._CURRENT_COLUMNS
            + producer._PLASMA_CURRENT_COLUMNS,
        )
        frame = producer._eligible_frame(row)
        if frame is None:
            raise RuntimeError(f"{paths[0].name} carries no eligible diverted frame")
        machine = producer.dataset_machine_description(
            row, source_row=str(row.get("_source_path", "corpus row"))
        )
        state["row"] = {"file": paths[0].name, "frame": frame}

    with phase(state, f"{arguments.arm}:operand-build"):
        radius_full, height_full = producer.canonical_axes(row)
        stride = producer.REGISTERED_GRID_STRIDE
        radius = radius_full[::stride]
        height = height_full[::stride]
        couplings = producer._couplings(
            row,
            machine,
            radius,
            height,
            None,
            arguments.machine_artifact_cache,
            producer.DEFAULT_MACHINE_ARTIFACT_DIGEST,
        )
        names, wall, inside = couplings[0], couplings[1], couplings[2]
        source_to_grid, source_to_wall = couplings[3], couplings[4]
        plasma_to_grid, plasma_to_wall = couplings[5], couplings[6]
        operand_bytes = {
            "source_to_grid": int(source_to_grid.nbytes),
            "source_to_wall": int(source_to_wall.nbytes),
            "plasma_to_grid": int(plasma_to_grid.nbytes),
            "plasma_to_wall": int(plasma_to_wall.nbytes),
            "wall_coordinate": int(wall.nbytes),
            "inside_material": int(inside.nbytes),
        }
        state["operands"] = {
            "grid_nodes": int(plasma_to_grid.shape[0]),
            "wall_rows": int(plasma_to_wall.shape[0]),
            "conductor_sources": int(source_to_grid.shape[1]),
            "bytes": operand_bytes,
            "total_bytes": sum(operand_bytes.values()),
        }

    with phase(state, f"{arguments.arm}:profile-build"):
        built = producer._build_profile(
            row,
            frame,
            machine_artifact_cache=arguments.machine_artifact_cache,
            machine_artifact_digest=producer.DEFAULT_MACHINE_ARTIFACT_DIGEST,
        )
    state["profile"] = {
        "lattice_nodes": int(built.label.size),
        "lattice_shape": list(built.label.shape),
        "seed_bytes": int(built.seed.nbytes),
    }
    census(state, "profile-build", arguments.array_limit)

    profile, seed = built.profile, built.seed
    current = producer._source_current(
        row, machine.physical, names, float(row["efit_times"][frame])
    )
    target = float(producer._target_current(row, float(row["efit_times"][frame])))
    request = producer._registered_solve_request(
        profile,
        seed,
        producer.jnp.asarray(current),
        target,
        carrier_identity=f"ledger:{paths[0].name}:{frame}",
    )

    map_program = jax.jit(
        lambda flux: profile.flux_map(producer.jnp.asarray(current))(flux)
    )

    with phase(state, f"{arguments.arm}:solve-trace-and-lower"):
        lowered = map_program.lower(producer.jnp.asarray(seed))
        text = lowered.as_text()
        state["hlo"] = {
            "text_bytes": len(text.encode()),
            "text_lines": text.count("\n"),
            "constant_literals": text.count("constant("),
        }
        print(f"HLO_BYTES {len(text.encode())}", flush=True)

    with phase(state, f"{arguments.arm}:solve-compile"):
        compiled = lowered.compile()
        try:
            analysis = compiled.memory_analysis()
            state["memory_analysis"] = {
                name: int(getattr(analysis, name))
                for name in dir(analysis)
                if not name.startswith("_") and isinstance(getattr(analysis, name), int)
            }
        except Exception as error:
            state["memory_analysis"] = f"unavailable: {error}"
        for name in ("serialized_executable", "as_bytes"):
            try:
                payload = getattr(compiled, name)
                payload = payload() if callable(payload) else payload
                state["serialized_executable_bytes"] = len(bytes(payload))
                break
            except Exception:
                continue
        if "serialized_executable_bytes" not in state:
            state["serialized_executable_bytes"] = "unavailable"

    with phase(state, f"{arguments.arm}:solve-first-execute"):
        first = compiled(producer.jnp.asarray(seed))
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), first)

    with phase(state, f"{arguments.arm}:solve-warm-execute"):
        warm = compiled(producer.jnp.asarray(seed))
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), warm)
    census(state, "after-warm-execute", arguments.array_limit)

    if not os.environ.get("NOVA_LEDGER_SKIP_PRESENT_ROUTE"):
        with phase(state, f"{arguments.arm}:present-route-solve"):
            receipt = profile.solve(request)
            receipt.equilibrium.flux.block_until_ready()
            state["solve"] = {
                "qualified": bool(receipt.qualified),
                "residual": float(receipt.equilibrium.fixed_point.residual),
                "newton_steps": int(
                    getattr(
                        receipt.equilibrium.fixed_point, "active_set_iterations", -1
                    )
                ),
            }
    census(state, "solved", arguments.array_limit)

    from nova.equilibrium import support_read

    with phase(state, f"{arguments.arm}:bank-attempt"):
        attempt: dict = {"builder": "support_read.SupportOperands.from_rows"}
        try:
            geometry = profile.operator.grid
            bank = support_read.SupportOperands.from_rows(
                geometry,
                plasma_to_grid,
                plasma_to_wall[:, :1],
                bucket=len(plasma_to_grid),
            )
            attempt["result"] = {
                "built": True,
                "bucket": bank.bucket,
                "rows_bytes": int(bank.rows.size * bank.rows.dtype.itemsize),
                "exterior_bytes": int(
                    bank.exterior.size * bank.exterior.dtype.itemsize
                ),
            }
        except Exception as error:
            attempt["result"] = {
                "built": False,
                "exception": type(error).__name__,
                "message": str(error)[:800],
            }
        state["bank_attempt"] = attempt

    state["completed"] = True
    state["stop_signal"] = None
    persist(state)
    print(f"PEAK {state['peak_rss']} {state['peak_rss_bytes']}", flush=True)


if __name__ == "__main__":
    main()
