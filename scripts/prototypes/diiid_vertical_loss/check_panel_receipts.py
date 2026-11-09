"""Refuse a committed panel receipt that points outside the repository.

The receipt is read by people and tools that never saw the render, so a figure
pointer has to resolve from a fresh clone: repository-relative, and naming a
file that is still there.  A run scratch directory, a worktree or a temporary
tree satisfies neither, and dies with the render that wrote it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def repository_root(start: Path) -> Path:
    for parent in start.resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise SystemExit(f"{start} is not inside a repository")


def check(receipt: Path) -> list[str]:
    root = repository_root(receipt)
    records = json.loads(receipt.read_text())["panels"]
    faults = []
    for record in records:
        pointer = record.get("svg")
        name = f"frame {record.get('frame')} trip {record.get('trip')}"
        if not isinstance(pointer, str) or not pointer:
            faults.append(f"{name} carries no svg pointer")
            continue
        path = Path(pointer)
        if path.is_absolute():
            faults.append(f"{name} names an absolute path: {pointer}")
        elif not (root / path).exists():
            faults.append(f"{name} names a file that does not resolve: {pointer}")
        elif root not in (root / path).resolve().parents:
            faults.append(f"{name} resolves outside the repository: {pointer}")
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
