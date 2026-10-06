"""No committed artifact may name a reckon worktree path.

The repository-wide scan: ``git ls-files`` enumerates every tracked file, and
``worktree_path_offenders`` -- the one scanning decision, with the exemption set
and the worktree-root directory name both owned by ``nova.database.filepath`` --
returns each non-exempt path whose contents name a worktree.  The predicate does
all pattern matching; this module adds none of its own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from nova.database.filepath import worktree_path_offenders

ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    return [name for name in result.stdout.decode("utf-8").split("\0") if name]


def _working_tree_reader(path) -> str:
    """Return ``path``'s text, or '' for a binary file.

    The scan mirrors ``git grep -I``: a file carrying a NUL byte in its opening
    block is binary and is skipped, so the guard and the census share one reach.
    """
    data = (ROOT / path).read_bytes()
    if b"\0" in data[:8192]:
        return ""
    return data.decode("utf-8", errors="replace")


def _top_level(paths) -> set[str]:
    return {Path(path).parts[0] for path in paths if "/" in str(path)}


def test_no_tracked_file_names_a_worktree_path():
    offenders = worktree_path_offenders(_tracked_files(), _working_tree_reader)
    assert offenders == [], (
        "tracked files naming a worktree path (run `filepath relativize` on "
        "each): " + ", ".join(map(str, offenders))
    )


def test_the_scan_reaches_every_top_level_directory():
    """Positive control: the scan reads a file in every top-level directory."""
    tracked = _tracked_files()
    queried: list[str] = []

    def reader(path):
        queried.append(str(path))
        return _working_tree_reader(path)

    worktree_path_offenders(tracked, reader)

    assert queried, "the scan queried no file"
    # The predicate skips the exemption subtree, but each top-level directory
    # still carries a non-exempt file, so every one must be reached.
    assert _top_level(tracked) <= _top_level(queried)
    # The reader sees content, so an empty offender list is not a dead reader.
    assert reader("pyproject.toml") != ""
