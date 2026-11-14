"""Run the four edge rows with the established profile-extent batch driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "extent_batch", ROOT / "scripts/prototypes/diiid_xpoint_profile_extent/batch.py"
)
extent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extent)
extent.HERE = HERE
extent.tasks = lambda: [("edge", frame, branch) for frame, branch in extent.ROWS]
completed = extent.completed
extent.completed = lambda folder: (
    completed(folder) and (folder / "spatial.json").is_file()
)

if __name__ == "__main__":
    extent.main()
