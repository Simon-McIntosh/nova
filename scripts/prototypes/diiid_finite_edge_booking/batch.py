"""Run the four finite-edge rows on the booked finite-edge source route.

The rows, the profiles and the receipt shape are the established
profile-extent cohort's; this driver runs its edge arm alone, because that arm
carries the finite-edge tables the booked route exists for.

Usage::

    python batch.py --tree <worktree> --revision <sha> --output <dir> \\
        --trace-helper <trace helper> --slots 4
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location(
    "extent_batch", ROOT / "scripts/prototypes/diiid_xpoint_profile_extent/batch.py"
)
extent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extent)
extent.HERE = HERE
extent.tasks = lambda: [("edge", frame, branch) for frame, branch in extent.ROWS]
_completed = extent.completed
extent.completed = lambda folder: (
    _completed(folder) and (folder / "finite-edge.json").is_file()
)

if __name__ == "__main__":
    extent.main()