"""List every reference to a retired shared root in tracked source.

The retired root is an argument rather than a constant so this file carries no
reference of its own to the directory it searches for: the scan then reads the
same tree it reports on. Each row names the file, the line, whether the
reference points at the DIII-D corpus under ``raw/`` (left in place) or at a
default (moved to the new root), and the referenced path itself. Run before and
after the rename and diff the two listings: the second must report only corpus
rows, each identical to the first.

    python scripts/prototypes/shared_root/inventory.py <retired-root> > before.txt
"""

from __future__ import annotations

import re
import subprocess
import sys

NEW_ROOT = "/work/projects/imas_gpu/nova"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: inventory.py <retired-root>", file=sys.stderr)
        return 2
    old_root = argv[1].rstrip("/") + "/"
    corpus_prefix = old_root + "raw/"
    reference = re.compile(re.escape(old_root.rstrip("/")) + r"""[^\s'"`)\]},>]*""")
    search = old_root[len("/work/projects/") :].rstrip("/")
    listing = subprocess.run(
        ["git", "grep", "-n", search, "--", ".", ":!docs"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows: list[str] = []
    for record in listing.splitlines():
        location, _, text = record.partition(":")
        line, _, text = text.partition(":")
        location = f"{location}:{line}"
        for match in reference.finditer(text):
            path = match.group(0)
            kind = "corpus" if path.startswith(corpus_prefix) else "default"
            rows.append(f"{location}:{kind}:{path}")
    rows.sort()
    print("\n".join(rows))
    counts = {
        kind: sum(1 for row in rows if row.split(":", 3)[2] == kind)
        for kind in ("corpus", "default")
    }
    print(
        f"# corpus={counts['corpus']} default={counts['default']} new_root={NEW_ROOT}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
