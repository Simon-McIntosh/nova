"""Attribute the present-route forward-solve producer's memory to named owners.

Runs the DIII-D forward-gs producer through one frame's profile build and one
solve on the present route, recording host RSS by phase, the largest live
arrays with their owning attributes, the compiled map program's memory
analysis, and the operand-cache files read. It then attempts the §1 operator
bank for the same machine shape and records the bank's bytes beside the
present route, or the refusal that stops the attempt and the code it comes
labelled with the rung and device it was measured on.

The ledger is written as JSON and a Markdown table under the output directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
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
    except Exception as error:  # pragma: no cover
        return {"import": f"unavailable: {type(error).__name__}: {error}"}
    memory: dict = {"backend": jax.default_backend()}
    memory["devices"] = [str(device) for device in jax.local_devices()]
    try:
        stats = jax.local_devices()[0].memory_stats()
        memory["device_memory_stats"] = (
            None if stats is None else {k: int(v) for k, v in stats.items()}
        )
    except Exception as error:  # pragma: no cover - backend dependent
        memory["device_memory_stats"] = f"unavailable: {type(error).__name__}: {error}"
    try:
        live = jax.live_arrays()
        memory["live_array_bytes"] = int(sum(a.size * a.dtype.itemsize for a in live))
        memory["live_array_count"] = len(live)
    except Exception as error:  # pragma: no cover
        memory["live_array_bytes"] = f"unavailable: {type(error).__name__}: {error}"
    return memory


def _owner_names(target: object) -> list[str]:
    """Return candidate ``object.attribute`` names that reference ``target``."""
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


def largest_arrays(limit: int = 30, floor: int = 1 << 1) -> dict:
    """Return the largest live arrays with shape, dtype, bytes and owners."""
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


def compile_memory(profile, seed) -> dict:
    """Record the map program's memory analysis, constant bytes and size."""
    import jax.numpy as jnp

    record: dict = {}
    program = jax.jit(lambda flux: profile.flux_map(jnp.asarray(flux)))
    try:
        compiled = program.lower(jnp.asarray(seed)).compile()
    except Exception as error:
        return {"compiled": f"unavailable: {type(error).__name__}: {error}"}
    record["compiled"] = True
    try:
        analysis = compiled.memory_analysis()
        record["memory_analysis"] = {
            name: int(getattr(analysis, name))
            for name in dir(analysis)
            if not name.startswith("_")
            and isinstance(getattr(analysis, name), int)
        }
    except Exception as error:
        record["memory_analysis"] = f"unavailable: {error}"
    try:
        text = compiled.as_text()
        record["hlo_text_bytes"] = len(text.encode())
        record["hlo_constant_literals"] = text.count("constant(")
    except Exception as error:
        record["as_text"] = f"unavailable: {error}"
    for name in ("serialized_executable", "as_bytes"):
        try:
            payload = getattr(compiled, name)
            payload = payload() if callable(payload) else payload
            record["serialized_executable_bytes"] = len(bytes(payload))
            break
        except Exception:
            continue
    return record


def cache_inventory(root: Path) -> dict:
    rows = []
    if root and root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                rows.append(
                    {
                        "path": str(path),
                        "bytes": stat.st_size,
                        "mtime": stat.st_mtime,
                    }
                )
    rows.sort(key=lambda row: row["bytes"], reverse=True)
    return {
        "root": str(root),
        "exists": bool(root and root.exists()),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "largest": rows[:50],
    }


def import_peak(python: str, module: str, env: dict) -> dict:
    """Peak RSS of a fresh interpreter that imports ``module`` and nothing else."""
    code = (
        "import json, resource\n"
        f"import {module}\n"
        "print(json.dumps({'ru_maxrss_bytes': "
        "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}))\n"
    )
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
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--machine-artifact-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-limit", type=int, default=1)
    parser.add_argument("--configurable", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    state: dict = {
        "phases": [],
        "arrays": {},
        "banner": {
            "revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "tree": str(ROOT),
            "cwd": str(Path.cwd().resolve()),
            "python": os.environ.get("NOVA_LEDGER_PYTHON", sys.executable),
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
    }

    from nova.jax.config import configure_dtypes

    started = perf_counter()
    configure_dtypes()
    import jax

    assert jax.config.jax_enable_x64 is True
    state["banner"]["x64"] = True
    mark(state, "python-import-nova-jax", started)

    import benchmarks.diiid_forward_gs_match as producer

    state["banner"]["producer_module"] = producer.__file__
    state["import_peak"] = {
        "bare_python": import_peak(
            state["banner"]["python"], "benchmarks.diiid_forward_gs_match",
            dict(os.environ),
        ),
    }
    state["banner"]["module_cwd"] = str(Path.cwd().resolve())

    # Machine description and frame selection.
    started = perf_counter()
    data = arguments.data or producer.DEFAULT_DATA
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
    mark(
        state,
        "machine-description",
        started,
        {"row": paths[0].name, "frame": frame, "physical": type(machine.physical).__name__},
    )

    # Coupling operators: the present route's operand build.
    started = perf_counter()
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
    names, wall, inside, source_to_grid, source_to_wall, plasma_to_grid, plasma_to_wall = (
        couplings[0],
        couplings[1],
        couplings[2],
        couplings[3],
        couplings[4],
        couplings[5],
        couplings[6],
    )
    operand_bytes = {
        "source_to_grid": source_to_grid.nbytes,
        "source_to_wall": source_to_wall.nbytes,
        "plasma_to_grid": plasma_to_grid.nbytes,
        "plasma_to_wall": plasma_to_wall.nbytes,
        "wall_coordinate": wall.nbytes,
        "inside_material": inside.nbytes,
    }
    mark(
        state,
        "operand-build",
        started,
        {
            "grid_nodes": int(plasma_to_grid.shape[0]),
            "wall_rows": int(plasma_to_wall.shape[0]),
            "conductor_sources": int(source_to_grid.shape[1]),
            "operand_bytes": operand_bytes,
            "operand_total_bytes": sum(operand_bytes.values()),
        },
    )
    state["arrays"]["operand-build"] = largest_arrays()
    state["operand_cache"] = cache_inventory(arguments.machine_artifact_cache)

    # Profile build.
    started = perf_counter()
    built = producer._build_profile(
        row,
        frame,
        machine_artifact_cache=arguments.machine_artifact_cache,
        machine_artifact_digest=producer.DEFAULT_MACHINE_ARTIFACT_DIGEST,
    )
    mark(
        state,
        "profile-build",
        started,
        {
            "lattice_nodes": int(built.label.size),
            "lattice_shape": list(built.label.shape),
            "seed_bytes": int(built.seed.nbytes),
        },
    )
    state["arrays"]["profile-build"] = largest_arrays()

    # One solve on the present route, with the compiled map program measured.
    profile = built.profile
    seed = built.seed
    target = float(producer._target_current(row, float(row["efit_times"][frame])))
    from nova.equilibrium.solve_request import ExplicitSolveSeed

    current = producer._source_current(
        row, machine.physical, names, float(row["efit_times"][frame])
    )
    request = producer._registered_solve_request(
        profile,
        seed,
        producer.jnp.asarray(current),
        target,
        carrier_identity=f"ledger:{paths[0].name}:{frame}",
    )
    started = perf_counter()
    receipt = profile.solve(request)
    receipt.equilibrium.flux.block_until_ready()
    mark(
        state,
        "compile-and-first-solve",
        started,
        {
            "qualified": bool(receipt.qualified),
            "residual": float(receipt.equilibrium.fixed_point.residual),
            "newton_steps": int(getattr(receipt.equilibrium.fixed_point, "active_set_iterations", -1)),
        },
    )
    state["program"] = compile_memory(profile, seed)
    state["arrays"]["solved"] = largest_arrays()

    # The §1 operator bank for the same shape, or its refusal.
    from nova.equilibrium import support_read

    attempt: dict = {"builder": "support_read.SupportOperands.from_rows"}
    try:
        geometry = profile.operator.grid
        operands = support_read.SupportOperands.from_rows(
            geometry, plasma_to_grid, plasma_to_wall[:, :1]
        )
        attempt["result"] = {"built": True, "bucket": operands.bucket}
    except Exception as error:
        attempt["result"] = {
            "built": False,
            "exception": type(error).__name__,
            "message": str(error),
        }
    state["bank_attempt"] = attempt

    peak = max(row["rss"]["ru_maxrss"] for row in state["phases"])
    state["peak_rss"] = peak
    state["peak_rss_phase"] = max(
        state["phases"], key=lambda row: row["rss"]["ru_maxrss"]
    )["phase"]
    (arguments.output / "ledger.json").write_text(
        json.dumps(state, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"LEDGER {arguments.output / 'ledger.json'}", flush=True)
    print(f"PEAK {peak}", flush=True)


def mark(state: dict, name: str, started: float, extra: dict | None = None) -> None:
    row = {
        "phase": name,
        "seconds": perf_counter() - started,
        "rss": host_rss(),
        "jax": jax_memory(),
    }
    if extra:
        row.update(extra)
    state["phases"].append(row)
    print(
        f"PHASE {name} seconds={row['seconds']:.3f} "
        f"rss={row['rss'].get('VmRSS')} peak={row['rss'].get('ru_maxrss')}",
        flush=True,
    )


if __name__ == "__main__":
    main()