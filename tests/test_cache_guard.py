"""The cache guard's entry accounting and its line-anchored marker.

A JAX persistent compilation cache holds one file per compiled program, named
``<program>-<key>-cache``, beside a subtree of XLA autotune records that no
lookup can ever hit. The guard's served-entry count is what a reader trusts to
say how large the cache it served was, so it counts executable entries and not
every file beneath the directory.

The marker is printed from a pytest session hook, where the progress line on
stdout is usually unterminated, so the guard writes it on a line of its own so
that a reader grepping for it at line start finds it on a run that passed.
"""

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

LANE_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts" / "h200_test_lane"
if str(LANE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(LANE_DIRECTORY))

import cache_guard  # noqa: E402


def write_entry(directory: Path, name: str) -> None:
    """Write an empty file at the given relative name under the serviced tree."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def populate(directory: Path) -> None:
    """Write a served directory holding entries, autotune records and noise."""
    for program in ("jit_solve", "jit_scan", "jit_while"):
        write_entry(directory, "%s-%s-cache" % (program, "0" * 8 + program))
    write_entry(directory, "xla_gpu_per_fusion_autotune_cache_dir/tmp/record.textproto")
    write_entry(directory, "notes.txt")
    write_entry(directory, "nested/jit_solve-outside-top-level-cache")


def test_program_key_entry_count_counts_only_executable_entries(tmp_path):
    """Only top-level ``-cache`` files are entries a lookup can be served by."""
    populate(tmp_path)
    assert cache_guard.program_key_entry_count(tmp_path) == 3


def test_program_key_entry_count_ignores_a_missing_directory(tmp_path):
    """A directory that was never created holds no entries."""
    assert cache_guard.program_key_entry_count(tmp_path / "absent") == 0


def test_served_directory_reports_program_key_entries(tmp_path, monkeypatch):
    """The reported served-entry count is the keyed set, not every file."""
    import jax

    populate(tmp_path)
    jax.config.update("jax_compilation_cache_dir", str(tmp_path))
    state = cache_guard.served_directory_state()
    assert state["served_directory"] == str(tmp_path)
    assert state["served_entries"] == 3


def test_marker_begins_a_line_after_unterminated_output(monkeypatch):
    """A line-anchored grep finds the marker after a partial progress line."""
    monkeypatch.delenv(cache_guard.PREWARM_RUN_VARIABLE, raising=False)
    monkeypatch.setenv(cache_guard.MISS_BUDGET_VARIABLE, "0")
    monkeypatch.setattr(cache_guard, "_LEDGER", cache_guard.CacheLedger())

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("tests/test_example.py .....", end="")
        cache_guard.emit_receipt()

    emitted = buffer.getvalue()
    marker_lines = [
        line
        for line in emitted.splitlines()
        if re.match(r"^CACHE_GUARD_MARKER=", line)
    ]
    assert marker_lines == ["CACHE_GUARD_MARKER=wall-clock-reliable misses=0 budget=0"]
    assert (
        "tests/test_example.py .....\nCACHE_GUARD_MARKER=" in emitted
    ), emitted