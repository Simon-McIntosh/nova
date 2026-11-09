"""Refuse a committed panel receipt that points outside the repository.

The receipt is read by people and tools that never saw the render, so a figure
pointer has to resolve from a fresh clone: repository-relative, inside the
published figures tree, and naming a file that is still there. A run scratch
directory, a worktree or a temporary tree satisfies none of those, and dies
with the render that wrote it.

Whether a pointer is a name the repository itself would record is decided by
``nova.database.filepath.repository_relative``, the mapping this project's
receipts are written through: a pointer that mapping would rewrite is not the
name of a published figure, and one it leaves alone is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.database.filepath import repository_relative

PUBLISHED_ROOT = Path("docs/figures")
"""The only place a receipt may point at: figures published in the repository."""


def check(receipt: Path) -> list[str]:
    relative = repository_relative(receipt)
    if Path(relative).is_absolute():
        raise SystemExit(f"{receipt} is not inside a repository")
    root = Path(receipt).resolve().parents[len(Path(relative).parts) - 1]
    records = json.loads(receipt.read_text())["panels"]
    faults = []
    for record in records:
        pointer = record.get("svg")
        name = f"frame {record.get('frame')} trip {record.get('trip')}"
        if not isinstance(pointer, str) or not pointer:
            faults.append(f"{name} carries no svg pointer")
            continue
        candidate = root / pointer
        if Path(pointer).is_absolute() or repository_relative(candidate) != pointer:
            faults.append(f"{name} is not a repository path: {pointer}")
        elif PUBLISHED_ROOT not in Path(pointer).parents:
            faults.append(f"{name} names a path outside {PUBLISHED_ROOT}: {pointer}")
        elif not candidate.exists():
            faults.append(f"{name} names a file that does not resolve: {pointer}")
    return faults


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    faults = check(arguments.receipt)
    for fault in faults:
        print(fault)
    if faults:
        raise SystemExit(f"PANEL RECEIPT UNRESOLVABLE: {arguments.receipt.name}")
    print("PANEL RECEIPTS RESOLVE")


if __name__ == "__main__":
    main()
