"""Refuse a module that discovers the repository root for itself.

``nova.database.filepath`` is this project's one mapping from an absolute path
to the name a committed artefact may record. A module that walks its own
parents looking for a ``.git`` marker re-derives that rule, and the copies
drift: the relativiser and the receipt check then disagree about which paths are
inside the repository, and a receipt that resolves for one reader fails for
another. Reuse the mapping; do not restate it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = '"' + ".git" + '"'
"""A repository marker in source is a hand-rolled root discovery."""

SHARED = "nova.database.filepath"
"""The module every repository-relative name is produced through."""


def offenders(root: Path) -> list[str]:
    faults = []
    for path in sorted(root.glob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text()
        if MARKER in text:
            faults.append(f"{path.name} discovers the repository root itself")
        elif "repository_relative(" in text and SHARED not in text:
            faults.append(f"{path.name} uses a path mapping it does not import")
    return faults


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    faults = offenders(arguments.root)
    for fault in faults:
        print(fault)
    if faults:
        raise SystemExit(f"PATH MAPPING RESTATED: {len(faults)} module(s)")
    print("PATH MAPPING REUSED")


if __name__ == "__main__":
    main()
