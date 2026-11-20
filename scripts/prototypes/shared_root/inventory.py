"""List every reference to the retired shared root in tracked source.

Each row names the file, the line, whether the reference points at the DIII-D
corpus under ``raw/`` (left in place) or at a default under the retired root
(moved to the new root), and the referenced path itself. Run before and after
the rename and diff the two listings: the second must report only corpus rows,
each identical to the first.

    python scripts/prototypes/shared_root/inventory.py > before.txt
"""

from __future__ import annotations

import re
import subprocess
import sys

OLD_ROOT = "/work/projects/imas_gpu/sophelio/"
CORPUS_PREFIX = OLD_ROOT + "raw/"
REFERENCE = re.compile(re.escape(OLD_ROOT.rstrip("/")) + r"""[^\s'"`)\]>,]*""")


def main() -> int:
    listing = subprocess.run(
        ["git", "grep", "-n", "imas_gpu/sophelio", "--", ".", ":!docs"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows: list[str] = []
    for record in listing.splitlines():
        location, _, text = record.partition(":")
        for match in REFERENCE.finditer(text):
            path = match.group(0)
            kind = "corpus" if path.startswith(CORPUS_PREFIX) else "default"
            rows.append(f"{location}:{kind}:{path}")
    rows.sort()
    print("\n".join(rows))
    counts = {
        kind: sum(1 for row in rows if row.split(":", 2)[1] == kind)
        for kind in ("corpus", "default")
    }
    print(f"# corpus={counts['corpus']} default={counts['default']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
