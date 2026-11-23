#!/usr/bin/env python3
"""Measure host memory, compile wall time and executable size of the whole-cell
forward solve program across the certificate weak row.

Each rung compiles --- never executes --- the program the public certificate
solve would run for the weak restricted case at one cell count: the per-mesh
accelerated history program, whose initial flux and fixture exterior are traced
arguments and whose route, topology class and target current are static.  A cold
child process records the compiler's peak resident set plus the allocation's
cgroup high-water mark, the lowering and backend-compile wall times separately,
the serialized executable size, the HLO instruction counts, and the compiler's
memory analysis.  A second fresh child process retrieves the same executable
from the persistent compilation cache and records the retrieval wall time.  A
parent drives the three rungs as sequential subprocesses and persists a part
receipt as each one lands, so an out-of-memory kill loses one rung rather than
the run; the aggregate stage re-reads the parts on the login node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.solovev_certificate import _certificate_compile_problem
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
)

CASE_NAME = "weak-rotation-reactor-static"
RUNG_CELLS = (300, 1000, 2500)
CACHE_MAXIMUM_BYTES = 1 << 40
SCHEMA = "nova.forward-solve-compile-host-memory"


def _source_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _cgroup_peak() -> int | None:
    """Return this allocation's resident-memory high-water mark when exposed."""
    try:
        rows = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        relative = next(row.split("::", 1)[1] for row in rows if "::" in row)
        peak = Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.peak"
        return int(peak.read_text(encoding="utf-8").strip())
    except FileNotFoundError, PermissionError, StopIteration, ValueError:
        return None


def _rusage_peak() -> int:
    """Return this process's peak resident set in bytes (Linux ru_maxrss)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _peak_probe() -> dict[str, int | None]:
    return {
        "ru_maxrss_bytes": _rusage_peak(),
        "cgroup_peak_bytes": _cgroup_peak(),
    }


def _memory_analysis_fields(analysis: Any) -> dict[str, int]:
    """Flatten the executable's buffer-allocation census to integer fields."""
    fields: dict[str, int] = {}
    for name in dir(analysis):
        if name.startswith("_"):
            continue
        value = getattr(analysis, name)
        if isinstance(value, int):
            fields[name] = int(value)
    return fields


def _count_hlo_instructions(text: str) -> int:
    """Count HLO instructions as the printed module's definition lines."""
    return sum(1 for line in text.splitlines() if " = " in line)


def _solve_program(profile: Any, request: Any) -> Any:
    """Return the compiled program ``profile.solve(request)`` would execute."""
    return profile._accelerated_history_program(
        request.route,
        requested_class=None,
        target_current=request.target_current,
        **request.policy.kernel_options(),
    )


def _lane() -> dict[str, str | None]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURM_JOB_NODELIST")
        or os.environ.get("SLURMD_NODENAME"),
    }


def _child(stage: str, cells: int, rung_dir: Path, cache_root: Path) -> int:
    """Compile one whole-cell solve program in this fresh process, never run it."""
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        print("COMPILE_HOST_MEMORY_ERROR expected binary64", flush=True)
        return 1
    rung_dir.mkdir(parents=True, exist_ok=True)
    cache = configure_persistent_compilation_cache(
        cache_root,
        minimum_compile_seconds=0.0,
        maximum_bytes=CACHE_MAXIMUM_BYTES,
    )
    start = perf_counter()
    record: dict[str, Any] = {"checkpoints": []}

    def checkpoint(label: str, extra: dict[str, Any]) -> None:
        record["checkpoints"].append(
            {
                "label": label,
                "elapsed_seconds": perf_counter() - start,
                "peak": _peak_probe(),
                **extra,
            }
        )
        _write_json(
            rung_dir / f"{stage}-state.json",
            {
                "schema": SCHEMA,
                "stage": stage,
                "cells": cells,
                "checkpoints": record["checkpoints"],
                "completed": False,
            },
        )

    # a negative count requests a cell number (~-int) while a positive one is a
    # filament linear dimension, so the count must carry the driver's negative sign
    began = perf_counter()
    profile, seed, request, dimensions = _certificate_compile_problem(
        CASE_NAME, -int(cells)
    )
    program = _solve_program(profile, request)
    external = profile.operator.external(request.current, request.prescribed_current)
    checkpoint(
        "construction",
        {"seconds": perf_counter() - began, "dimensions": dimensions},
    )

    began = perf_counter()
    lowered = program.lower(
        jnp.asarray(seed, dtype=jnp.float64),
        external,
        profile.operator,
    )
    lower_seconds = perf_counter() - began
    stablehlo_sha256 = hashlib.sha256(
        lowered.as_text(dialect="stablehlo").encode("utf-8")
    ).hexdigest()
    module = lowered.compiler_ir(dialect="hlo").as_hlo_module()
    hlo_instruction_count = sum(
        len(list(computation.instructions())) for computation in module.computations()
    )
    checkpoint(
        "lowering",
        {
            "seconds": lower_seconds,
            "stablehlo_sha256": stablehlo_sha256,
            "hlo_instruction_count": hlo_instruction_count,
        },
    )

    began = perf_counter()
    compiled = lowered.compile()
    backend_compile_seconds = perf_counter() - began
    checkpoint("backend_compile", {"seconds": backend_compile_seconds})

    began = perf_counter()
    executable_bytes = None
    serialize_error = None
    try:
        executable = compiled.runtime_executable().serialize()
        executable_bytes = len(executable)
    except (MemoryError, RuntimeError, ValueError) as error:
        serialize_error = f"{type(error).__name__}: {error}"
    memory_analysis = _memory_analysis_fields(compiled.memory_analysis())
    checkpoint(
        "serialize",
        {
            "seconds": perf_counter() - began,
            "executable_bytes": executable_bytes,
            "serialize_error": serialize_error,
            "memory_analysis": memory_analysis,
        },
    )

    optimized_instruction_count = None
    try:
        optimized_instruction_count = _count_hlo_instructions(compiled.as_text())
    except MemoryError, RuntimeError, ValueError:
        pass
    checkpoint(
        "optimized_hlo",
        {"optimized_instruction_count": optimized_instruction_count},
    )

    part = {
        "schema": SCHEMA,
        "stage": stage,
        "cells": cells,
        "source_revision": _source_revision(),
        "driver_sha256": _sha256(Path(__file__)),
        "cache_directory": str(cache.directory),
        "lane": _lane(),
        "dimensions": dimensions,
        "elapsed": {
            "construction_seconds": record["checkpoints"][0]["seconds"],
            "lower_seconds": lower_seconds,
            "backend_compile_seconds": backend_compile_seconds,
            "serialize_seconds": record["checkpoints"][3]["seconds"],
            "total_seconds": perf_counter() - start,
        },
        "stablehlo_sha256": stablehlo_sha256,
        "hlo": {
            "stablehlo_instruction_count": hlo_instruction_count,
            "optimized_instruction_count": optimized_instruction_count,
        },
        "executable_bytes": executable_bytes,
        "serialize_error": serialize_error,
        "memory_analysis": memory_analysis,
        "peak": _peak_probe(),
        "checkpoints": record["checkpoints"],
        "completed": True,
    }
    _write_json(rung_dir / f"{stage}.json", part)
    _write_json(
        rung_dir / f"{stage}-state.json",
        {
            "schema": SCHEMA,
            "stage": stage,
            "cells": cells,
            "checkpoints": record["checkpoints"],
            "completed": True,
        },
    )
    print(
        "COMPILE_HOST_MEMORY "
        f"stage={stage} cells={cells} "
        f"lower={lower_seconds:.3f}s compile={backend_compile_seconds:.3f}s "
        f"executable={part['executable_bytes']} "
        f"peak={part['peak']}",
        flush=True,
    )
    print("COMPILE_HOST_MEMORY_EXIT=0", flush=True)
    return 0


def _killed_partish(rung_dir: Path, stage: str) -> dict[str, Any]:
    """Recover the last observed state of a child process that did not finish."""
    state_path = rung_dir / f"{stage}-state.json"
    observed = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        observed = {
            "checkpoints": state.get("checkpoints"),
            "completed": state.get("completed", False),
        }
    return {
        "status": "killed",
        "stage": stage,
        "observed": observed,
        "allocation_cgroup_peak_bytes": _cgroup_peak(),
    }


def _spawn(
    python: str, rung_dir: Path, cache_root: Path, stage: str, cells: int
) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    log = (rung_dir / f"{stage}.log").open("wb")
    try:
        completed = subprocess.run(
            [
                python,
                "-m",
                "benchmarks.compile_host_memory",
                "--stage",
                stage,
                "--cells",
                str(cells),
                "--rung-dir",
                str(rung_dir),
                "--cache-root",
                str(cache_root),
            ],
            cwd=root,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    finally:
        log.close()
    return completed


def _run_rung(work: Path, python: str, cells: int) -> None:
    rung_dir = work / f"rung-{cells}"
    rung_dir.mkdir(parents=True, exist_ok=True)
    # jax's cache copy helper resolves the file's group name, so the writes must
    # land under a group this cluster can name, not the setgid stats' tree
    cache_root = (
        Path.home() / ".cache" / "nova" / "compile-host-memory" / f"cells-{cells}"
    )
    part_path = work / f"part-{cells}.json"
    _write_json(
        part_path,
        {
            "schema": SCHEMA,
            "cells": cells,
            "status": "running",
            "rung_dir": str(rung_dir),
        },
    )
    cold = _spawn(sys.executable, rung_dir, cache_root, "child", cells)
    cold_path = rung_dir / "child.json"
    if cold.returncode == 0 and cold_path.exists():
        cold_part = json.loads(cold_path.read_text(encoding="utf-8"))
        warm = _spawn(sys.executable, rung_dir, cache_root, "warm", cells)
        warm_path = rung_dir / "warm.json"
        if warm.returncode == 0 and warm_path.exists():
            warm_part = json.loads(warm_path.read_text(encoding="utf-8"))
        else:
            warm_part = _killed_partish(rung_dir, "warm")
        status = "complete"
    else:
        cold_part = _killed_partish(rung_dir, "child")
        warm_part = None
        status = "killed"
    _write_json(
        part_path,
        {
            "schema": SCHEMA,
            "cells": cells,
            "status": status,
            "cold": cold_part,
            "warm": warm_part,
        },
    )
    print(
        f"COMPILE_HOST_MEMORY_RUNG cells={cells} status={status} "
        f"cold_rc={cold.returncode}",
        flush=True,
    )


def _aggregate(work: Path) -> dict[str, Any]:
    parts = []
    for cells in RUNG_CELLS:
        path = work / f"part-{cells}.json"
        if path.exists():
            parts.append(json.loads(path.read_text(encoding="utf-8")))
    receipt = {
        "schema": SCHEMA,
        "source_revision": _source_revision(),
        "driver_sha256": _sha256(Path(__file__)),
        "lane": _lane(),
        "rungs": parts,
        "completed": all(part.get("status") == "complete" for part in parts),
    }
    _write_json(work / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "completed": receipt["completed"],
                "rungs": [
                    {"cells": part["cells"], "status": part["status"]} for part in parts
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def _smoke(cells: int) -> int:
    """Construct one rung's problem and program without lowering or compiling."""
    configure_dtypes()
    profile, seed, request, dimensions = _certificate_compile_problem(
        CASE_NAME, -int(cells)
    )
    program = _solve_program(profile, request)
    external = profile.operator.external(request.current, request.prescribed_current)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "stage": "smoke",
                "cells": cells,
                "dimensions": dimensions,
                "seed_shape": [int(value) for value in np.shape(seed)],
                "seed_dtype": str(jnp.asarray(seed).dtype),
                "external_shape": [int(value) for value in np.shape(external)],
                "program_identity": program.__name__,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("COMPILE_HOST_MEMORY_SMOKE_EXIT=0", flush=True)
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="measure whole-cell solve compile host memory"
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--cells", type=int, default=None)
    parser.add_argument("--rung-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--work", type=Path, default=None)
    arguments = parser.parse_args(argv)

    if arguments.stage == "child":
        if arguments.cells is None or arguments.rung_dir is None:
            raise SystemExit("child requires --cells and --rung-dir")
        if arguments.cache_root is None:
            raise SystemExit("child requires --cache-root")
        raise SystemExit(
            _child(
                arguments.stage,
                arguments.cells,
                arguments.rung_dir,
                arguments.cache_root,
            )
        )
    if arguments.stage == "warm":
        if arguments.cells is None or arguments.rung_dir is None:
            raise SystemExit("warm requires --cells and --rung-dir")
        if arguments.cache_root is None:
            raise SystemExit("warm requires --cache-root")
        raise SystemExit(
            _child(
                arguments.stage,
                arguments.cells,
                arguments.rung_dir,
                arguments.cache_root,
            )
        )
    if arguments.stage == "smoke":
        if arguments.cells is None:
            raise SystemExit("smoke requires --cells")
        raise SystemExit(_smoke(arguments.cells))
    if arguments.stage in {"parent", "aggregate"}:
        if arguments.work is None:
            raise SystemExit(f"{arguments.stage} requires --work")
        if arguments.stage == "parent":
            _run_rung(arguments.work, sys.executable, 300)
            _run_rung(arguments.work, sys.executable, 1000)
            _run_rung(arguments.work, sys.executable, 2500)
        _aggregate(arguments.work)
        return
    raise SystemExit(f"unknown stage {arguments.stage!r}")


if __name__ == "__main__":
    main()
