"""Probe native JAX compilation exits around the topology-boundary slice.

Each condition runs in a child process so a native abort remains an observed
exit status instead of terminating the diagnostic driver.  The probe executes
the exact topology test function after either no preparation, unrelated JAX
compilation warm-up, or a bounded virtual-memory constraint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import time

CONDITIONS = ("fresh", "warm-cache", "constrained-memory")
ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "tests" / "test_topology_boundary.py"


def _memory_bytes() -> dict[str, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    virtual_pages, resident_pages, *_ = Path("/proc/self/statm").read_text().split()
    return {
        "virtual": int(virtual_pages) * page_size,
        "resident": int(resident_pages) * page_size,
        "peak_resident": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def _warm_compilation_cache(count: int) -> None:
    import jax
    import jax.numpy as jnp

    compiled = []
    for index in range(count):
        width = 8 + index % 64
        function = jax.jit(lambda values, shift=float(index): jnp.sin(values) + shift)
        function(jnp.arange(width, dtype=jnp.float32)).block_until_ready()
        compiled.append(function)


def _constrain_address_space(headroom_mib: int) -> dict[str, int]:
    memory = _memory_bytes()
    headroom = headroom_mib * 1024 * 1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = memory["virtual"] + headroom
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return {"address_space_limit": limit, "headroom": headroom, **memory}


def _run_child(condition: str, warmup_compilations: int, headroom_mib: int) -> int:
    namespace = runpy.run_path(str(TEST_PATH), run_name="topology_boundary_probe")
    topology = namespace["_raster_nulls"]()[0]
    preparation: dict[str, int] = {}

    if condition == "warm-cache":
        _warm_compilation_cache(warmup_compilations)
        preparation = {
            "warmup_compilations": warmup_compilations,
            **_memory_bytes(),
        }
    elif condition == "constrained-memory":
        preparation = _constrain_address_space(headroom_mib)

    started = time.monotonic()
    namespace["test_x_mask_excludes_cells_beyond_null_heights"](topology)
    print(
        json.dumps(
            {
                "condition": condition,
                "elapsed_seconds": time.monotonic() - started,
                "preparation": preparation,
                "result": "passed",
                "memory_after": _memory_bytes(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _exit_status(returncode: int) -> tuple[int, int | None]:
    if returncode < 0:
        signal = -returncode
        return 128 + signal, signal
    return returncode, None


def _run_parent(
    conditions: tuple[str, ...],
    output_dir: Path,
    warmup_compilations: int,
    headroom_mib: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for condition in conditions:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--condition",
            condition,
            "--warmup-compilations",
            str(warmup_compilations),
            "--memory-headroom-mib",
            str(headroom_mib),
        ]
        environment = os.environ.copy()
        environment["JAX_PLATFORMS"] = "cpu"
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        status, signal = _exit_status(completed.returncode)
        stdout_path = output_dir / f"{condition}.stdout.log"
        stderr_path = output_dir / f"{condition}.stderr.log"
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        records.append(
            {
                "condition": condition,
                "exit_status": status,
                "raw_returncode": completed.returncode,
                "signal": signal,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
        )

    summary = {
        "jax_platform": "cpu",
        "warmup_compilations": warmup_compilations,
        "memory_headroom_mib": headroom_mib,
        "results": records,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument("--warmup-compilations", type=int, default=256)
    parser.add_argument("--memory-headroom-mib", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("topology-compile-logs")
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.child:
        if arguments.condition == "all":
            parser.error("a child process requires one condition")
        return _run_child(
            arguments.condition,
            arguments.warmup_compilations,
            arguments.memory_headroom_mib,
        )

    conditions = CONDITIONS if arguments.condition == "all" else (arguments.condition,)
    return _run_parent(
        conditions,
        arguments.output_dir,
        arguments.warmup_compilations,
        arguments.memory_headroom_mib,
    )


if __name__ == "__main__":
    raise SystemExit(main())
