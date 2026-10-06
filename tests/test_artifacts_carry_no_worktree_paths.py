"""No committed artifact may name a reckon worktree path.

The repository-wide scan searches the working tree for candidates with
``worktree_path_candidates`` -- one enumeration shared with the commit check --
and then ``worktree_path_offenders``, the one scanning decision, with the
exemption set and the worktree-root directory name both owned by
``nova.database.filepath``, returns each non-exempt path whose contents name a
worktree.  The predicate does all pattern matching; this module adds none of its
own.
"""

from __future__ import annotations

from pathlib import Path

from nova.database.filepath import (
    worktree_path_candidates,
    worktree_path_offenders,
    worktree_path_text,
)

ROOT = Path(__file__).resolve().parents[1]


def test_no_tracked_file_names_a_worktree_path():
    candidates = worktree_path_candidates(ROOT)
    # A search run from the wrong directory, without git, or after pattern drift
    # returns nothing, and an empty offender list would then read as a clean
    # tree.  The exempt owner module always names the root, so its presence is
    # the positive control that makes an empty candidate list a failure.
    assert "nova/database/filepath.py" in candidates, (
        "the candidate search returned nothing it is known to hold "
        "(nova/database/filepath.py), so this scan proves nothing: "
        f"candidates={candidates!r}"
    )
    offenders = worktree_path_offenders(
        candidates, lambda path: worktree_path_text(ROOT, path)
    )
    assert offenders == [], (
        "tracked files naming a worktree path (run `filepath relativize` on "
        "each): " + ", ".join(map(str, offenders))
    )
