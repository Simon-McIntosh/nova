"""Fold per-frame probe receipts into one file without overwriting a frame.

The probe runs one process per frame, because host memory accumulates across
frames inside one process and a single long job exhausted its allocation doing
three of them.  Each process writes its own receipt; this merges them.

A frame already present is never replaced.  The one exception is a frame that
is present but incomplete -- a branch that refused in an earlier attempt and
was retried later -- where the missing branch is added and an existing branch
is still refused.  Every decision is reported, and a refused overlap exits
nonzero so a caller cannot mistake it for a merge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def merge(target_path: Path, parts: list[Path]) -> tuple[dict[str, Any], list[str]]:
    target = _load(target_path) if target_path.exists() else {"frames": {}}
    target.setdefault("frames", {})
    notes: list[str] = []
    refused: list[str] = []
    peak: dict[str, int] = {
        frame: int(value)
        for frame, value in (target.get("peak_rss_kb_by_frame") or {}).items()
    }
    for frame in target["frames"]:
        peak.setdefault(frame, -1)
    for part_path in parts:
        part = _load(part_path)
        for frame, record in part.get("frames", {}).items():
            existing = target["frames"].get(frame)
            if existing is None:
                target["frames"][frame] = record
                notes.append(f"added frame {frame} from {part_path.name}")
                peak[frame] = int(part.get("peak_rss_kb", -1))
                continue
            added = [
                name
                for name in record.get("branches", {})
                if name not in existing.get("branches", {})
            ]
            clash = [
                name
                for name in record.get("branches", {})
                if name in existing.get("branches", {})
            ]
            for name in added:
                existing.setdefault("branches", {})[name] = record["branches"][name]
                notes.append(f"added frame {frame} branch {name} from {part_path.name}")
            if clash:
                refused.append(
                    f"frame {frame} branch(es) {', '.join(sorted(clash))} already "
                    f"present; {part_path.name} was not used for them"
                )
    target["peak_rss_kb_by_frame"] = peak
    return target, notes + refused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    arguments = parser.parse_args()
    merged, notes = merge(arguments.probe, arguments.parts)
    arguments.probe.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"probe": str(arguments.probe), "notes": notes}, indent=2))
    if any("already present" in note for note in notes):
        raise SystemExit("REFUSED: a part would have overwritten a frame branch")


if __name__ == "__main__":
    main()
