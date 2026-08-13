"""Attribute retained native resources across one ordered pytest process.

The command-line driver starts the repository's complete test selection in a
child process.  The same module is loaded as a pytest plugin in that child and
records process memory, Linux mapping classes, native thread names, JAX
compilations, live executables, and JAX cache occupancy after a fixed number of
completed tests.

An optional diagnostic park runs two controls in the measured process: generic
garbage collection followed by JAX cache eviction.  Their separate memory and
live-executable deltas distinguish reclaimable Python state from retained JAX
state without treating a cache count as a memory measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARK_NODE = (
    "tests/test_topology_boundary.py::test_x_mask_excludes_cells_beyond_null_heights"
)
MIB = 1024 * 1024

_OUTPUT_PATH: Path | None = None
_INTERVAL = 0
_PARK_NODE = ""
_COMPLETED = 0
_COMPILE_CALLS = 0
_LAST_NODE = ""
_FINALIZED_NODEIDS: set[str] = set()


def _read_key_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key] = value.strip()
    return values


def _kilobytes(value: str) -> int:
    fields = value.split()
    return int(fields[0]) * 1024 if fields else 0


def _mapping_class(pathname: str) -> str:
    lowered = pathname.lower()
    if not pathname:
        return "anonymous"
    if pathname.startswith("[heap"):
        return "heap"
    if pathname.startswith("[stack"):
        return "thread_stack"
    if pathname.startswith("["):
        return "kernel_special"
    if "jaxlib" in lowered or "xla" in lowered:
        return "jax_file"
    if "numpy" in lowered or "scipy" in lowered or "openblas" in lowered:
        return "numerical_file"
    if "python" in lowered:
        return "python_file"
    return "other_file"


def _smaps_totals() -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    anonymous_sizes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    current_class = ""
    current_size_class = ""
    header_fields = 0
    for line in Path("/proc/self/smaps").read_text().splitlines():
        fields = line.split()
        if fields and "-" in fields[0] and fields[0][0].isalnum():
            header_fields += 1
            pathname = " ".join(fields[5:]) if len(fields) > 5 else ""
            current_class = _mapping_class(pathname)
            start, end = (int(value, 16) for value in fields[0].split("-"))
            size_mib = (end - start) / MIB
            current_size_class = (
                f"{size_mib:.0f}_mib" if size_mib >= 1 else "less_than_1_mib"
            )
            if current_class == "anonymous":
                anonymous_sizes[current_size_class]["count"] += 1
            continue
        if not current_class or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"Size", "Rss", "Pss", "Private_Dirty", "Private_Clean"}:
            byte_value = _kilobytes(value)
            totals[current_class][key.lower()] += byte_value
            if current_class == "anonymous":
                anonymous_sizes[current_size_class][key.lower()] += byte_value
    totals["all"]["mapping_count"] = header_fields
    return (
        {category: dict(values) for category, values in totals.items()},
        {category: dict(values) for category, values in anonymous_sizes.items()},
    )


def _thread_names() -> dict[str, int]:
    names = Counter()
    for task in Path("/proc/self/task").iterdir():
        try:
            names[(task / "comm").read_text().strip()] += 1
        except FileNotFoundError:
            continue
    return dict(sorted(names.items()))


def _allocator_totals() -> dict[str, int]:
    class Mallinfo(ctypes.Structure):
        _fields_ = [
            ("arena", ctypes.c_size_t),
            ("ordblks", ctypes.c_size_t),
            ("smblks", ctypes.c_size_t),
            ("hblks", ctypes.c_size_t),
            ("hblkhd", ctypes.c_size_t),
            ("usmblks", ctypes.c_size_t),
            ("fsmblks", ctypes.c_size_t),
            ("uordblks", ctypes.c_size_t),
            ("fordblks", ctypes.c_size_t),
            ("keepcost", ctypes.c_size_t),
        ]

    try:
        libc = ctypes.CDLL(None)
        mallinfo = libc.mallinfo2
        mallinfo.restype = Mallinfo
        values = mallinfo()
    except AttributeError, OSError:
        return {}
    return {
        "arena": values.arena,
        "mapped": values.hblkhd,
        "allocated": values.uordblks,
        "free": values.fordblks,
    }


def _jax_totals() -> dict[str, int | str]:
    if "jax" not in sys.modules:
        return {
            "platform": "not-imported",
            "compile_calls": _COMPILE_CALLS,
            "live_executables": 0,
            "python_cache_entries": 0,
            "pjit_cache_entries": 0,
        }
    try:
        from jax._src import pjit, util, xla_bridge
    except ImportError:
        return {
            "platform": "not-imported",
            "compile_calls": _COMPILE_CALLS,
            "live_executables": 0,
            "python_cache_entries": 0,
            "pjit_cache_entries": 0,
        }

    cache_entries = 0
    for cache in list(util._caches):
        try:
            cache_entries += cache.cache_info().currsize
        except AttributeError, RuntimeError:
            continue
    if not xla_bridge.backends_are_initialized():
        return {
            "platform": "imported-not-initialized",
            "compile_calls": _COMPILE_CALLS,
            "live_executables": 0,
            "python_cache_entries": cache_entries,
            "pjit_cache_entries": 0,
        }
    backend = xla_bridge.get_backend()
    return {
        "platform": backend.platform,
        "compile_calls": _COMPILE_CALLS,
        "live_executables": len(backend.live_executables()),
        "live_buffers": len(backend.live_buffers()),
        "python_cache_entries": cache_entries,
        "pjit_cache_entries": (
            pjit._cpp_pjit_cache_fun_only.size()
            + pjit._cpp_pjit_cache_explicit_attributes.size()
        ),
    }


def _snapshot(reason: str, nodeid: str = "") -> dict[str, Any]:
    jax_totals = _jax_totals()
    status = _read_key_values(Path("/proc/self/status"))
    mappings, anonymous_sizes = _smaps_totals()
    native_threads = int(status.get("Threads", "0"))
    python_threads = threading.active_count()
    return {
        "kind": "sample",
        "reason": reason,
        "elapsed_seconds": time.monotonic() - _STARTED,
        "completed_tests": _COMPLETED,
        "nodeid": nodeid or _LAST_NODE,
        "resident_bytes": _kilobytes(status.get("VmRSS", "0 kB")),
        "virtual_bytes": _kilobytes(status.get("VmSize", "0 kB")),
        "peak_resident_bytes": _kilobytes(status.get("VmHWM", "0 kB")),
        "native_threads": native_threads,
        "python_threads": python_threads,
        "native_only_threads": native_threads - python_threads,
        "thread_names": _thread_names(),
        "allocator": _allocator_totals(),
        "mappings": mappings,
        "anonymous_mapping_sizes": anonymous_sizes,
        "jax": jax_totals,
    }


def _write_record(record: dict[str, Any]) -> None:
    assert _OUTPUT_PATH is not None
    with _OUTPUT_PATH.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _trim_allocator() -> None:
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except AttributeError, OSError:
        pass


def _install_compile_counter() -> None:
    if "jax" not in sys.modules:
        return
    from jax._src import compiler

    original = compiler.compile_or_get_cached
    if getattr(original, "_lane_resource_counter", False):
        return

    def counted(*args: Any, **kwargs: Any) -> Any:
        global _COMPILE_CALLS
        _COMPILE_CALLS += 1
        return original(*args, **kwargs)

    counted._lane_resource_counter = True  # type: ignore[attr-defined]
    compiler.compile_or_get_cached = counted


def pytest_sessionstart(session: Any) -> None:
    del session
    global _OUTPUT_PATH, _INTERVAL, _PARK_NODE, _STARTED
    output = os.environ.get("NOVA_LANE_RESOURCE_LOG")
    if not output:
        return
    _OUTPUT_PATH = Path(output)
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text("")
    _INTERVAL = int(os.environ.get("NOVA_LANE_RESOURCE_INTERVAL", "100"))
    _PARK_NODE = os.environ.get("NOVA_LANE_RESOURCE_PARK", "")
    _STARTED = time.monotonic()
    _write_record(_snapshot("session_start"))


def pytest_collection_finish(session: Any) -> None:
    del session
    if _OUTPUT_PATH is None:
        return
    _install_compile_counter()
    _write_record(_snapshot("collection_finish"))


def pytest_runtest_logreport(report: Any) -> None:
    global _COMPLETED, _LAST_NODE
    terminal = report.when == "call" or report.skipped
    if _OUTPUT_PATH is None or not terminal or report.nodeid in _FINALIZED_NODEIDS:
        return
    _FINALIZED_NODEIDS.add(report.nodeid)
    _COMPLETED += 1
    _LAST_NODE = report.nodeid
    if _COMPLETED % _INTERVAL == 0:
        _write_record(_snapshot("fixed_interval", report.nodeid))


def pytest_runtest_setup(item: Any) -> None:
    _install_compile_counter()
    if _OUTPUT_PATH is None or not _PARK_NODE or not item.nodeid.startswith(_PARK_NODE):
        return
    import jax
    import pytest

    _write_record(_snapshot("park_pre_control", item.nodeid))
    gc.collect()
    _trim_allocator()
    _write_record(_snapshot("park_post_gc_control", item.nodeid))
    jax.clear_caches()
    gc.collect()
    _trim_allocator()
    _write_record(_snapshot("park_post_jax_clear_control", item.nodeid))
    pytest.exit(f"resource measurement parked before {item.nodeid}", returncode=86)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    del session
    if _OUTPUT_PATH is not None:
        record = _snapshot("session_finish")
        record["pytest_exit_status"] = exitstatus
        _write_record(record)


def _delta(end: dict[str, Any], start: dict[str, Any], key: str) -> int:
    return int(end.get(key, 0)) - int(start.get(key, 0))


def _nested_delta(
    end: dict[str, Any], start: dict[str, Any], category: str, key: str
) -> int:
    return int(end.get(category, {}).get(key, 0)) - int(
        start.get(category, {}).get(key, 0)
    )


def _summarise(samples_path: Path, returncode: int, commit: str) -> dict[str, Any]:
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    start = samples[0]
    park_sample = next(
        (row for row in reversed(samples) if row["reason"] == "park_pre_control"),
        None,
    )
    endpoint = park_sample or samples[-1]
    post_gc = next(
        (row for row in samples if row["reason"] == "park_post_gc_control"), None
    )
    post_jax = next(
        (row for row in samples if row["reason"] == "park_post_jax_clear_control"),
        None,
    )

    mapping_candidates = []
    categories = sorted(set(start["mappings"]) | set(endpoint["mappings"]))
    for category in categories:
        if category == "all":
            continue
        mapping_candidates.append(
            {
                "candidate": category,
                "resident_delta_bytes": _nested_delta(
                    endpoint["mappings"], start["mappings"], category, "rss"
                ),
                "virtual_delta_bytes": _nested_delta(
                    endpoint["mappings"], start["mappings"], category, "size"
                ),
            }
        )
    owner = max(mapping_candidates, key=lambda row: row["virtual_delta_bytes"])

    anonymous_size_growth = []
    size_classes = sorted(
        set(start["anonymous_mapping_sizes"]) | set(endpoint["anonymous_mapping_sizes"])
    )
    for size_class in size_classes:
        anonymous_size_growth.append(
            {
                "size_class": size_class,
                "mapping_count_delta": _nested_delta(
                    endpoint["anonymous_mapping_sizes"],
                    start["anonymous_mapping_sizes"],
                    size_class,
                    "count",
                ),
                "virtual_delta_bytes": _nested_delta(
                    endpoint["anonymous_mapping_sizes"],
                    start["anonymous_mapping_sizes"],
                    size_class,
                    "size",
                ),
                "resident_delta_bytes": _nested_delta(
                    endpoint["anonymous_mapping_sizes"],
                    start["anonymous_mapping_sizes"],
                    size_class,
                    "rss",
                ),
            }
        )
    retained_threads = _delta(endpoint, start, "native_only_threads")
    reservation_owner = max(
        anonymous_size_growth, key=lambda row: row["virtual_delta_bytes"]
    )
    reservation_owner["virtual_mib_per_retained_native_thread"] = (
        reservation_owner["virtual_delta_bytes"] / MIB / retained_threads
        if retained_threads
        else None
    )

    controls: dict[str, Any] = {}
    if post_gc is not None:
        controls["garbage_collection"] = {
            "resident_released_bytes": -_delta(post_gc, endpoint, "resident_bytes"),
            "virtual_released_bytes": -_delta(post_gc, endpoint, "virtual_bytes"),
        }
    if post_gc is not None and post_jax is not None:
        controls["jax_cache_clear"] = {
            "resident_released_bytes": -_delta(post_jax, post_gc, "resident_bytes"),
            "virtual_released_bytes": -_delta(post_jax, post_gc, "virtual_bytes"),
            "live_executables_released": -_delta(
                post_jax["jax"], post_gc["jax"], "live_executables"
            ),
            "python_cache_entries_released": -_delta(
                post_jax["jax"], post_gc["jax"], "python_cache_entries"
            ),
            "pjit_cache_entries_released": -_delta(
                post_jax["jax"], post_gc["jax"], "pjit_cache_entries"
            ),
        }

    thread_name_deltas = {
        name: count - start["thread_names"].get(name, 0)
        for name, count in endpoint["thread_names"].items()
        if count - start["thread_names"].get(name, 0)
    }
    xla_thread_names = {
        name: count
        for name, count in thread_name_deltas.items()
        if name.startswith("llvm-worker-")
        or name.startswith("tf_")
        or name == "python3"
    }
    resident_growth = _delta(endpoint, start, "resident_bytes")
    virtual_growth = _delta(endpoint, start, "virtual_bytes")
    thread_growth = _delta(endpoint, start, "native_threads")
    python_thread_growth = _delta(endpoint, start, "python_threads")
    heap_candidate = next(
        row for row in mapping_candidates if row["candidate"] == "heap"
    )
    anonymous_candidate = next(
        row for row in mapping_candidates if row["candidate"] == "anonymous"
    )

    return {
        "commit": commit,
        "command": "pytest -m 'slow or not slow'",
        "pytest_returncode": returncode,
        "endpoint_reason": "diagnostic_park" if park_sample else "process_exit",
        "completed_tests_at_endpoint": endpoint["completed_tests"],
        "last_completed_nodeid": endpoint["nodeid"],
        "elapsed_seconds_at_endpoint": endpoint["elapsed_seconds"],
        "growth": {
            "resident_bytes": resident_growth,
            "virtual_bytes": virtual_growth,
            "native_threads": thread_growth,
            "native_only_threads": retained_threads,
            "python_threads": python_thread_growth,
            "jax_compile_calls": _delta(endpoint["jax"], start["jax"], "compile_calls"),
            "jax_live_executables": _delta(
                endpoint["jax"], start["jax"], "live_executables"
            ),
        },
        "attribution": {
            "single_largest_owner": "JAX/XLA CPU compilation and runtime state",
            "retained_thread_name_deltas": thread_name_deltas,
            "xla_native_thread_name_deltas": xla_thread_names,
            "candidates": [
                {
                    **heap_candidate,
                    "candidate": "heap including retained compiled objects",
                    "resident_growth_share": (
                        heap_candidate["resident_delta_bytes"] / resident_growth
                    ),
                    "virtual_growth_share": (
                        heap_candidate["virtual_delta_bytes"] / virtual_growth
                    ),
                    "jax_live_executables": endpoint["jax"]["live_executables"],
                },
                {
                    **anonymous_candidate,
                    "candidate": "anonymous native mappings",
                    "resident_growth_share": (
                        anonymous_candidate["resident_delta_bytes"] / resident_growth
                    ),
                    "virtual_growth_share": (
                        anonymous_candidate["virtual_delta_bytes"] / virtual_growth
                    ),
                },
                {
                    "candidate": "Python-managed threads",
                    "retained_threads": python_thread_growth,
                    "retained_thread_share": (
                        python_thread_growth / thread_growth if thread_growth else 0.0
                    ),
                    "control": "threading.active_count versus /proc native count",
                },
                {
                    "candidate": "native-only threads",
                    "retained_threads": retained_threads,
                    "retained_thread_share": (
                        retained_threads / thread_growth if thread_growth else 0.0
                    ),
                    "control": "native count minus threading.active_count",
                },
            ],
        },
        "mapping_candidates": mapping_candidates,
        "anonymous_mapping_size_growth": anonymous_size_growth,
        "largest_address_space_contributor": owner,
        "largest_native_reservation_class": reservation_owner,
        "controls": controls,
        "sample_count": len(samples),
    }


def _run(arguments: argparse.Namespace) -> int:
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "resource-samples.jsonl"
    pytest_path = output_dir / "monolithic-pytest.log"
    command_path = output_dir / "command.json"
    summary_path = output_dir / "summary.json"

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    branch = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if branch.returncode == 0:
        raise SystemExit("lane measurements require a frozen detached worktree")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "slow or not slow",
        "-p",
        "benchmarks.lane_resource_accumulation",
    ]
    environment = os.environ.copy()
    environment["NOVA_LANE_RESOURCE_LOG"] = str(samples_path)
    environment["NOVA_LANE_RESOURCE_INTERVAL"] = str(arguments.interval)
    environment["NOVA_LANE_RESOURCE_PARK"] = arguments.park_before
    environment.setdefault("JAX_PLATFORMS", "cpu")
    command_path.write_text(
        json.dumps(
            {
                "command": command,
                "commit": commit,
                "cwd": str(ROOT),
                "interval": arguments.interval,
                "park_before": arguments.park_before,
                "jax_platforms": environment["JAX_PLATFORMS"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with pytest_path.open("w") as pytest_log:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=pytest_log,
                stderr=subprocess.STDOUT,
                timeout=arguments.timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = 124

    if not samples_path.exists() or not samples_path.read_text().strip():
        raise SystemExit(f"pytest produced no resource samples; see {pytest_path}")
    summary = _summarise(samples_path, returncode, commit)
    summary["artifacts"] = {
        "command": str(command_path),
        "pytest_log": str(pytest_path),
        "samples": str(samples_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if returncode in {0, 86} else returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--park-before", default=DEFAULT_PARK_NODE)
    parser.add_argument("--timeout-seconds", type=int, default=2700)
    parser.add_argument("--output-dir", type=Path, default=Path("lane-resource-logs"))
    arguments = parser.parse_args()
    if arguments.interval <= 0:
        parser.error("--interval must be positive")
    return _run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
