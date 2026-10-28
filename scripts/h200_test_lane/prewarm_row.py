"""Compile one canonical solve program and record what compiling it cost.

The pre-warm job issues each program in its own process: an interpreter retains
every executable it compiles, so a sequence run in one process exhausts the LLVM
section allocator. This wrapper installs the cache ledger, runs one program in
this process so its compile is charged here, and appends a JSON line naming the
cache keys, the compile seconds and the hit or miss outcome to the run receipt.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time

import cache_guard


def run_command(argv: list[str]) -> int:
    """Run one program in this process and return its exit status."""
    if argv[0] == "pytest":
        import pytest

        return int(pytest.main(argv[1:]))
    sys.argv = list(argv)
    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as exit_signal:
        code = exit_signal.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    return 0


def program_status(
    command_status: int, unpersisted_misses: int, allow_command_failures: bool
) -> int:
    """Return the status this row reports for the pre-warm.

    The deliverable of a pre-warm is a populated cache, so a row whose
    compilations were all persisted has warmed what it was there to warm even
    when the command that drove them failed. A lane target with failing cases
    of its own therefore does not make every pre-warm run exit nonzero; the
    command's status is still recorded in the receipt so the failure stays
    visible. A miss written nowhere is never tolerated: that is a pre-warm
    that reports a cache it does not have.
    """
    if unpersisted_misses:
        return 1
    if allow_command_failures:
        return 0
    return command_status


def build_receipt(label: str, argv: list[str], status: int, seconds: float) -> dict:
    """Return one program's compile accounting as a receipt row."""
    ledger = cache_guard.ledger()
    row = {
        "label": label,
        "argv": argv,
        "status": status,
        "command_status": status,
        "wall_seconds": round(seconds, 3),
        "hits": ledger.hit_count(),
        "misses": ledger.miss_count(),
        "unpersisted_misses": ledger.unpersisted_misses,
        "compile_seconds": round(ledger.compile_seconds(), 3),
        "programs": ledger.rows(),
    }
    row.update(cache_guard.served_directory_state())
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", required=True, help="name this program in the receipt"
    )
    parser.add_argument("--receipt", required=True, help="JSON lines file to append to")
    parser.add_argument(
        "--allow-pytest-failures",
        action="store_true",
        help=(
            "report success when every compilation was persisted, even if the "
            "command driving them exited nonzero; the command's status is still "
            "recorded in the receipt"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    argv = list(args.command)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        parser.error("no command given")

    cache_guard.install()
    started = time.monotonic()
    status = run_command(argv)
    record = build_receipt(args.label, argv, status, time.monotonic() - started)
    record["status"] = program_status(
        status, record["unpersisted_misses"], args.allow_pytest_failures
    )
    with open(args.receipt, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print("PREWARM_ROW=%s" % json.dumps(record, sort_keys=True), flush=True)
    if record["unpersisted_misses"]:
        print(
            "PREWARM_ROW_UNPERSISTED=%s label=%s misses=%d served_directory=%s"
            % (
                cache_guard.CACHE_NOT_PERSISTED,
                args.label,
                record["unpersisted_misses"],
                record["served_directory"] or "none",
            ),
            flush=True,
        )
        return 1
    return record["status"]


if __name__ == "__main__":
    raise SystemExit(main())
