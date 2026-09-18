"""Per-program compile accounting for the persistent compilation cache.

JAX reports persistent-cache writes as a monitoring event and reports nothing
for the corresponding reads, so a lane log cannot tell a run that reused a
pre-warmed executable from one that paid the compile again. This module wraps
and records the cache lookup and write, prints a header stating the directory
the lane serves and the miss budget, and prints one row per program. A lane run
whose miss count exceeds the budget prints WALL-CLOCK-UNRELIABLE and fails; a run
that declares itself a pre-warm prints the same rows with no budget enforced,
because compiling those programs is the work it is there to do.

A write is only a write when the persistent cache is enabled. JAX calls
``put_executable_and_time`` on every compile that clears the minimum compile
time and returns without touching the disk when ``jax_compilation_cache_dir`` is
unset, which is the state ``JAX_ENABLE_COMPILATION_CACHE=1`` alone leaves it in.
Recording that call as a miss alone therefore reports a compile cost the run
never persisted, so each write also records whether the cache was enabled and a
run whose writes were not persisted fails with CACHE-NOT-PERSISTED rather than
reporting a populated cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "WALL_CLOCK_UNRELIABLE",
    "CACHE_NOT_PERSISTED",
    "PREWARM_RUN_MARKER",
    "DEFAULT_MISS_BUDGET",
    "PINNED_CACHE_ROOT",
    "CacheLedger",
    "install",
    "ledger",
    "miss_budget",
    "prewarm_run",
    "pin_reference",
    "pinned_cache_directory",
    "verify_pinned_directory",
    "sample_gpu_utilisation",
    "emit_header",
    "emit_receipt",
]

WALL_CLOCK_UNRELIABLE = "WALL-CLOCK-UNRELIABLE"
CACHE_NOT_PERSISTED = "CACHE-NOT-PERSISTED"
PREWARM_RUN_MARKER = "pre-warm-run"
DEFAULT_MISS_BUDGET = 0
MISS_BUDGET_VARIABLE = "NOVA_CACHE_MISS_BUDGET"
PREWARM_RUN_VARIABLE = "NOVA_CACHE_PREWARM_RUN"
REFUSE_TIMING_VARIABLE = "NOVA_CACHE_GUARD_REFUSE_TIMING"
PREWARM_PIN_FILENAME = "prewarm-latest.json"


# The pinned pre-warm directory lives on shared storage the compute nodes reach
# and outside every pruner root: it is not under $HOME/.cache and not inside a
# worktree, so a pre-warm survives until the next merge-time pre-warm replaces
# it.  The root is named by the same variable the drivers resolve through, so
# the pin document and the served directory cannot disagree.
PINNED_CACHE_ROOT = Path(
    os.environ.get(
        "NOVA_COMPILATION_CACHE_ROOT",
        "/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm",
    )
)


def prewarm_run() -> bool:
    """Report whether this process populates the cache rather than reading it.

    A pre-warm exists to fill the cache, so its misses are the deliverable and
    its own budget is not enforceable; a lane run reads a cache that a pre-warm
    already compiled, so there a miss means the measurement is not the one it
    reports and the run fails.
    """
    declared = os.environ.get(PREWARM_RUN_VARIABLE, "").strip().lower()
    return declared not in {"", "0", "false", "no"}


def miss_budget() -> int | None:
    """Return the enforced miss budget, or None where none is enforceable."""
    if prewarm_run():
        return None
    return int(os.environ.get(MISS_BUDGET_VARIABLE, DEFAULT_MISS_BUDGET))


@dataclass
class CacheLedger:
    """Compile seconds and hit or miss outcome keyed by JAX cache key."""

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    installed: bool = False
    unpersisted_misses: int = 0

    def record(
        self,
        key: str,
        program: str,
        seconds: float,
        outcome: str,
        persisted: bool = True,
    ) -> None:
        entry = self.entries.get(key)
        if entry is None:
            entry = {"cache_key": key, "program": program, "hits": 0, "misses": 0}
            self.entries[key] = entry
        if program:
            entry["program"] = program
        if outcome == "hit":
            entry["hits"] += 1
            entry["cached_compile_seconds"] = seconds
        else:
            entry["misses"] += 1
            entry["compile_seconds"] = seconds
            if not persisted:
                entry["unpersisted_misses"] = entry.get("unpersisted_misses", 0) + 1
                self.unpersisted_misses += 1

    def hit_count(self) -> int:
        return sum(entry["hits"] for entry in self.entries.values())

    def miss_count(self) -> int:
        return sum(entry["misses"] for entry in self.entries.values())

    def compile_seconds(self) -> float:
        return sum(
            entry.get("compile_seconds", 0.0)
            for entry in self.entries.values()
            if entry["misses"]
        )

    def rows(self) -> list[dict[str, Any]]:
        return sorted(self.entries.values(), key=lambda row: row["cache_key"])


_LEDGER = CacheLedger()


def ledger() -> CacheLedger:
    """Return the process ledger, installing the wrappers on first use."""
    return install()


def install() -> CacheLedger:
    """Record JAX persistent-cache lookups and writes in the ledger."""
    global _LEDGER
    if _LEDGER.installed:
        return _LEDGER

    from jax._src import compilation_cache

    original_get = compilation_cache.get_executable_and_time
    original_put = compilation_cache.put_executable_and_time

    def get_executable_and_time(cache_key, *args, **kwargs):
        result = original_get(cache_key, *args, **kwargs)
        executable, compile_time = result
        if executable is not None:
            _LEDGER.record(cache_key, "", float(compile_time or 0.0), "hit")
        return result

    def put_executable_and_time(
        cache_key, module_name, executable, backend, compile_time
    ):
        persisted = bool(compilation_cache.is_persistent_cache_enabled())
        _LEDGER.record(cache_key, module_name, float(compile_time), "miss", persisted)
        return original_put(cache_key, module_name, executable, backend, compile_time)

    compilation_cache.get_executable_and_time = get_executable_and_time
    compilation_cache.put_executable_and_time = put_executable_and_time
    _LEDGER.installed = True
    return _LEDGER


def pin_reference() -> dict[str, Any] | None:
    """Read the pinned pre-warm receipt published beside the shared cache root."""
    receipt = PINNED_CACHE_ROOT / PREWARM_PIN_FILENAME
    if not receipt.is_file():
        return None
    return json.loads(receipt.read_text(encoding="utf-8"))


def pinned_cache_directory() -> Path:
    """Return the runtime-versioned directory the pre-warm job compiled into."""
    reference = pin_reference()
    if reference is None:
        return PINNED_CACHE_ROOT
    return Path(reference["directory"])


def served_directory_state() -> dict[str, Any]:
    """Report the directory this process writes executables into, if any.

    The persistent cache is only enabled when a directory is configured, so a
    run whose compilations were never persisted shows up here as a named
    directory holding nothing rather than as a populated cache.
    """
    import jax

    configured = jax.config.jax_compilation_cache_dir
    return {
        "served_directory": str(configured) if configured else "",
        "served_entries": directory_entry_count(configured) if configured else 0,
    }


def directory_entry_count(directory: Path | str) -> int:
    """Return the number of files written under a cache directory."""
    path = Path(directory)
    if not path.is_dir():
        return 0
    return sum(1 for entry in path.rglob("*") if entry.is_file())


def verify_pinned_directory() -> dict[str, Any]:
    """Report the pinned directory's entry count and the pre-warm revision."""
    directory = pinned_cache_directory()
    entries = directory_entry_count(directory)
    bytes_on_disk = 0
    if directory.is_dir():
        for path in directory.rglob("*"):
            if path.is_file():
                bytes_on_disk += path.stat().st_size
    reference = pin_reference() or {}
    return {
        "pinned_cache_root": str(PINNED_CACHE_ROOT),
        "pinned_cache_directory": str(directory),
        "pinned_entries": entries,
        "pinned_bytes": bytes_on_disk,
        "prewarm_present": bool(reference),
        "prewarm_revision": reference.get("source_revision"),
        "prewarm_job_id": reference.get("slurm_job_id"),
    }


def sample_gpu_utilisation(offsets: tuple[int, ...] = (10, 30, 60)) -> None:
    """Print the device inventory, then GPU utilisation at the requested seconds.

    The samples are taken while the target program runs, so a reading above zero
    is the target's own use of the device and is not masked by a dispatch issued
    from here. A zero reading only says something about the target when the
    allocation's device is known to be present, so the inventory row is printed
    first: a device listed there and idle is a run that stayed on the host,
    while a device missing or erroring there is an allocation that never reached
    one.
    """
    inventory = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    utilisation = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    print("GPU_DEVICE_INVENTORY=%s" % _gpu_reading(inventory), flush=True)
    started = time.monotonic()
    for offset in offsets:
        remaining = offset - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        print(
            "GPU_UTILISATION_AT_%dS=%s" % (offset, _gpu_reading(utilisation)),
            flush=True,
        )


def _gpu_reading(command: list[str]) -> str:
    """Return one nvidia-smi reading, or the failure name if it is unavailable."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        return "unavailable:%s" % type(error).__name__
    return completed.stdout.strip().replace("\n", " ").replace(" ", "_")


def emit_header(cache_directory: Path | str, version_key: str = "") -> None:
    """Print the cache contract this run is measured under."""
    import jax

    reference = pin_reference()
    print("CACHE_GUARD_DIRECTORY=%s" % pinned_cache_directory(), flush=True)
    print("CACHE_GUARD_ROOT=%s" % PINNED_CACHE_ROOT, flush=True)
    print("CACHE_GUARD_SERVED=%s" % cache_directory, flush=True)
    print(
        "CACHE_GUARD_CACHE_ENABLED=%s"
        % (jax.config.jax_compilation_cache_dir or "none"),
        flush=True,
    )
    print("CACHE_GUARD_VERSION_KEY=%s" % version_key, flush=True)
    budget = miss_budget()
    print(
        "CACHE_GUARD_MISS_BUDGET=%s" % ("not-enforced" if budget is None else budget),
        flush=True,
    )
    print(
        "CACHE_GUARD_PREWARM=%s"
        % (json.dumps(reference, sort_keys=True) if reference else "none"),
        flush=True,
    )


def emit_receipt(version_key: str = "", revision: str = "unknown") -> dict[str, Any]:
    """Print the per-program receipt and return it as a mapping."""
    budget = miss_budget()
    misses = _LEDGER.miss_count()
    receipt = {
        "cache_directory": str(pinned_cache_directory()),
        "version_key": version_key,
        "source_revision": revision,
        "miss_budget": budget,
        "enforced": budget is not None,
        "hits": _LEDGER.hit_count(),
        "misses": misses,
        "marker": "",
        "rows": _LEDGER.rows(),
    }
    receipt["compile_seconds"] = round(_LEDGER.compile_seconds(), 3)
    if budget is None:
        marker = "%s misses=%d budget=not-enforced" % (PREWARM_RUN_MARKER, misses)
    elif misses > budget:
        marker = "%s misses=%d budget=%d" % (WALL_CLOCK_UNRELIABLE, misses, budget)
    else:
        marker = "wall-clock-reliable misses=%d budget=%d" % (misses, budget)
    unpersisted = _LEDGER.unpersisted_misses
    receipt["unpersisted_misses"] = unpersisted
    receipt.update(served_directory_state())
    if unpersisted:
        marker += " unpersisted=%d" % unpersisted
    receipt["marker"] = marker
    print("CACHE_GUARD_MARKER=%s" % marker, flush=True)
    print("CACHE_GUARD_RECEIPT=%s" % json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def pytest_sessionstart(session, **_kwargs):
    """Install the ledger and print the cache contract before any test runs."""
    install()
    emit_header(pinned_cache_directory())


def pytest_sessionfinish(session, **_kwargs):
    """Emit the receipt; refuse a result the run cannot support."""
    receipt = emit_receipt(
        version_key=os.environ.get("NOVA_PREWARM_VERSION_KEY", ""),
        revision=os.environ.get("H200_LANE_EXPECTED_REVISION", "unknown"),
    )
    if receipt["unpersisted_misses"]:
        print(
            "CACHE_GUARD_PERSISTENCE=%s misses=%d"
            % (CACHE_NOT_PERSISTED, receipt["unpersisted_misses"]),
            flush=True,
        )
        session.exitstatus = 1
        return
    if not receipt["enforced"]:
        return
    refuse = os.environ.get(REFUSE_TIMING_VARIABLE, "1") not in {"0", "false", "no"}
    if receipt["misses"] > receipt["miss_budget"] and refuse:
        session.exitstatus = 1
