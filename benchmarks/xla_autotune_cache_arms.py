#!/usr/bin/env python3
"""Measure XLA's per-fusion autotuner and its persistent cache on one certificate row.

Four arms run the same certificate row (``tests/test_current_book_fragments.py
::test_analytic_certificate_moments``), each against a fresh cache root under
node-local ``/tmp``:

  a  the JAX default, cold: the persistent compile cache and XLA's per-fusion
     autotune cache are both enabled, and the root starts empty.
  b  the default again with the compile entries removed and the autotune
     subtree kept, so the row recompiles while the autotuner reads its cached
     emitter choice.  Comparing (b)'s cold compile wall against (a)'s shows what
     the autotune cache saves on a compile miss.
  c  ``JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none``: the compile cache stays on
     and XLA's autotune subtree is not created.
  d  ``XLA_FLAGS=--xla_gpu_experimental_enable_fusion_autotuner=false``: the
     per-fusion autotuner does not run at all.

Comparing (d) against (c) states the fusion-autotuner decision: what the
autotuner costs in compile wall against what it moves in steady-state solve
wall.

Receipt glossary: ``cold_compile_seconds`` is the sum of the cache guard
ledger's per-program ``compile_seconds`` on the first run; ``programs`` carries
one row per program with its compile seconds and hit/miss counts;
``cold_total_seconds`` is the wall of the whole first pytest run (interpreter
import, fixtures, compile and solve together); ``steady_seconds`` is the wall of
each repeat run after the first; ``files`` separates the compile entries
(``<key>-cache`` files outside the autotune subtree) from the autotune entries
(files under ``xla_gpu_per_fusion_autotune_cache_dir``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SHARED_PYTHON = Path("/home/ITER/mcintos/Code/nova/.venv/bin/python")
CACHE_GUARD_DIRECTORY = ROOT / "scripts/h200_test_lane"
TARGET_PATH = "tests/test_current_book_fragments.py"
AUTOTUNE_DIR_NAME = "xla_gpu_per_fusion_autotune_cache_dir"
COLOURS = {"compile": "#3366cc", "solve": "#8aa04a"}
ARMS = ("a", "b", "c", "d")
ARM_LABELS = {
    "a": "default cold",
    "b": "compile cleared / autotune kept",
    "c": "autotune cache off",
    "d": "fusion autotuner off",
}


def row_selection(kind: str, cells: int) -> str:
    """Return the pytest -k expression that selects exactly one certificate row.

    ``-k`` matches substrings; its grammar has no ``*`` glob, so a wildcard is
    rejected as a usage error before collection and the row never runs.
    """
    return f"certificate_moments and {kind} and {cells}"


def arm_environment(arm: str, root: Path) -> dict[str, str]:
    """Return the process environment overrides for one arm's measurement runs."""
    environment = {
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT), str(CACHE_GUARD_DIRECTORY), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "JAX_PLATFORMS": "cuda,cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_COMPILATION_CACHE_DIR": str(root),
        "NOVA_COMPILATION_CACHE_ROOT": str(root),
        # The arms miss and hit in different proportions by design; the guard
        # still reports both, it simply must not refuse the run.
        "NOVA_CACHE_MISS_BUDGET": "1000000",
        "NOVA_CACHE_GUARD_REFUSE_TIMING": "0",
    }
    if arm == "c":
        environment["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = "none"
    if arm == "d":
        environment["XLA_FLAGS"] = (
            "--xla_gpu_experimental_enable_fusion_autotuner=false"
        )
    return environment


def is_autotune_path(parts: tuple[str, ...]) -> bool:
    """Report whether a cache-relative path lies in an autotune subtree."""
    return any(
        AUTOTUNE_DIR_NAME in part or "autotune" in part or "gemm" in part
        for part in parts
    )


def inventory(root: Path) -> dict[str, int]:
    """Count files under a cache root, separating compile from autotune entries."""
    counts = {"total": 0, "compile_entries": 0, "autotune_entries": 0, "other": 0}
    if not root.is_dir():
        return counts
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        counts["total"] += 1
        parts = path.relative_to(root).parts
        if is_autotune_path(parts):
            counts["autotune_entries"] += 1
        elif path.name.endswith("-cache"):
            counts["compile_entries"] += 1
        else:
            counts["other"] += 1
    return counts


def cache_guard_receipt_text(text: str) -> dict:
    """Return the last CACHE_GUARD_RECEIPT object carried in run output."""
    receipt: dict = {}
    for line in text.splitlines():
        marker = "CACHE_GUARD_RECEIPT="
        if line.startswith(marker):
            try:
                receipt = json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                continue
    return receipt


def cache_guard_receipt(log_path: Path) -> dict:
    """Return the last CACHE_GUARD_RECEIPT object written to a run log."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return cache_guard_receipt_text(text)


_DROPPED_LOG_PREFIXES = ("CACHE_GUARD_PREWARM=",)


def trim_log(text: str) -> str:
    """Return run output with the pre-warm inventory dropped.

    The cache guard prints one `CACHE_GUARD_PREWARM=` line carrying the whole
    pre-warm inventory, which reaches a couple of megabytes and buries the rows
    a reader needs.  Only that prefix is dropped: the receipt line grows with
    the number of programs and is the record the arm is read from, so a length
    threshold must never select it.
    """
    kept = [
        line for line in text.splitlines() if not line.startswith(_DROPPED_LOG_PREFIXES)
    ]
    return "\n".join(kept) + "\n"


def run_target_once(
    arm: str, root: Path, results: Path, log_path: Path, selection: str
) -> dict:
    """Run the certificate row once and return its wall, status and log path."""
    results.mkdir(parents=True, exist_ok=True)
    command = [
        str(SHARED_PYTHON),
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "cache_guard",
        TARGET_PATH,
        "-k",
        selection,
        "--timeout",
        "1800",
        "-q",
    ]
    environment = dict(os.environ)
    environment.update(arm_environment(arm, root))
    environment["NOVA_BOOK_RESULTS"] = str(results)
    started = perf_counter()
    completed = subprocess.run(
        command, cwd=str(ROOT), env=environment, capture_output=True, text=True
    )
    wall = perf_counter() - started
    text = completed.stdout + completed.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(trim_log(text), encoding="utf-8")
    return {
        "wall_seconds": wall,
        "exit_status": completed.returncode,
        "log": str(log_path),
        "text": text,
    }


def measure_arm(
    arm: str, root: Path, workdir: Path, repeats: int, selection: str
) -> dict:
    """Run one arm's cold pass plus its repeats against a prepared cache root."""
    root.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    before = inventory(root)
    cold = run_target_once(
        arm, root, workdir / "results", workdir / "cold.log", selection
    )
    cold_receipt = cache_guard_receipt_text(cold["text"])
    after_cold = inventory(root)
    steady = []
    for index in range(repeats):
        step = run_target_once(
            arm,
            root,
            workdir / "results",
            workdir / f"repeat-{index + 1}.log",
            selection,
        )
        steady.append(
            {"wall_seconds": step["wall_seconds"], "exit_status": step["exit_status"]}
        )
    programs = [
        {
            "program": row.get("program", ""),
            "cache_key": row.get("cache_key", ""),
            "compile_seconds": row.get("compile_seconds", 0.0),
            "hits": row.get("hits", 0),
            "misses": row.get("misses", 0),
        }
        for row in cold_receipt.get("rows", [])
    ]
    return {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "root": str(root),
        "cold_total_seconds": cold["wall_seconds"],
        "cold_exit_status": cold["exit_status"],
        "cold_compile_seconds": cold_receipt.get("compile_seconds", 0.0),
        "cold_hits": cold_receipt.get("hits", 0),
        "cold_misses": cold_receipt.get("misses", 0),
        "cold_marker": cold_receipt.get("marker", ""),
        "programs": programs,
        "steady_seconds": [row["wall_seconds"] for row in steady],
        "steady_exit_status": [row["exit_status"] for row in steady],
        "files_before": before,
        "files_after_cold": after_cold,
        "cold_log": cold["log"],
    }


def seed_autotune_subtree(source: Path, destination: Path) -> int:
    """Copy only the autotune subtrees of one cache root into another."""
    copied = 0
    if not source.is_dir():
        return copied
    for subtree in source.rglob(AUTOTUNE_DIR_NAME):
        if subtree.is_dir():
            shutil.copytree(
                subtree, destination / subtree.relative_to(source), dirs_exist_ok=True
            )
            copied += sum(1 for item in subtree.rglob("*") if item.is_file())
    return copied


def drive(
    base: Path, repeats: int, kind: str, cells: int, logs: Path | None = None
) -> dict:
    """Run the four arms in order, each against its own prepared cache root.

    ``base`` holds the four per-arm cache roots and is node-local scratch.
    ``logs`` holds the per-arm run logs and is a durable directory, so a reader
    can see why an arm produced what it did after the job's scratch is gone.
    """
    base.mkdir(parents=True, exist_ok=True)
    selection = row_selection(kind, cells)
    logs = base / "logs" if logs is None else logs
    roots = {arm: base / f"arm-{arm}" for arm in ARMS}
    for root in roots.values():
        if root.exists():
            shutil.rmtree(root)
    receipts = {}
    for arm in ARMS:
        workdir = logs / f"work-{arm}"
        workdir.mkdir(parents=True, exist_ok=True)
        if arm == "b":
            seed_autotune_subtree(roots["a"], roots["b"])
            # Arm (b) is the default arm with the compile entries removed and
            # the autotune subtree kept: seed the subtree, then measure as the
            # default so the row recompiles against the warm autotuner.
        receipts[arm] = measure_arm(arm, roots[arm], workdir, repeats, selection)
    return {"row": {"kind": kind, "cells": cells}, "repeats": repeats, "arms": receipts}


def figure_from_receipt(receipt: dict, figure_directory: Path) -> tuple[Path, Path]:
    """Draw compile wall and steady-state solve wall per arm as one SVG and PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = [arm for arm in ARMS if arm in receipt["arms"]]
    compile_wall = [receipt["arms"][arm]["cold_compile_seconds"] for arm in arms]
    solve_wall = [
        float(np.median(receipt["arms"][arm]["steady_seconds"]))
        if receipt["arms"][arm]["steady_seconds"]
        else 0.0
        for arm in arms
    ]
    positions = np.arange(len(arms))
    width = 0.36
    figure, axes = plt.subplots(figsize=(14, 6))
    axes.bar(
        positions - width / 2,
        compile_wall,
        width,
        color=COLOURS["compile"],
        label="cold compile wall",
        zorder=2,
    )
    axes.bar(
        positions + width / 2,
        solve_wall,
        width,
        color=COLOURS["solve"],
        label="steady-state solve wall",
        zorder=2,
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(
        [f"{arm}: {ARM_LABELS[arm]}" for arm in arms], rotation=0, fontsize=12
    )
    axes.set_ylabel("wall [s]", fontsize=16)
    axes.tick_params(labelsize=14)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    axes.grid(False)
    axes.set_axisbelow(True)
    figure.legend(loc="upper center", frameon=False, ncols=2, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure_directory.mkdir(parents=True, exist_ok=True)
    svg_path = figure_directory / "xla-autotune-arms-walls.svg"
    png_path = figure_directory / "xla-autotune-arms-walls.png"
    figure.savefig(svg_path)
    figure.savefig(png_path, dpi=100)
    plt.close(figure)
    return svg_path, png_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="cache-root parent under /tmp"
    )
    parser.add_argument("--receipt", type=Path, help="JSON receipt to write")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kind", default="diverted")
    parser.add_argument("--cells", type=int, default=550)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--figure", type=Path, help="draw from --receipt into this directory"
    )
    arguments = parser.parse_args()
    if arguments.dry_run:
        for arm in ARMS:
            environment = arm_environment(arm, arguments.base / f"arm-{arm}")
            print("ARM=%s ENV=%s" % (arm, json.dumps(environment, sort_keys=True)))
        print("ROW=%s" % row_selection(arguments.kind, arguments.cells))
        return 0
    if arguments.figure is not None:
        receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
        svg_path, png_path = figure_from_receipt(receipt, arguments.figure)
        print("SVG=%s PNG=%s" % (svg_path, png_path))
        return 0
    if arguments.receipt is None:
        parser.error("--receipt is required unless --dry-run or --figure")
    logs = arguments.receipt.parent / "xla-autotune-arm-logs"
    receipt = drive(
        arguments.base, arguments.repeats, arguments.kind, arguments.cells, logs
    )
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt["logs_directory"] = str(logs)
    arguments.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for arm in ARMS:
        entry = receipt["arms"][arm]
        print(
            "ARM=%s cold_compile=%.3f cold_total=%.3f steady=%s files=%s"
            % (
                arm,
                entry["cold_compile_seconds"],
                entry["cold_total_seconds"],
                [round(value, 3) for value in entry["steady_seconds"]],
                entry["files_after_cold"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
