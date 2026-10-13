"""Measure the diverted certificate map at one named revision.

The certificate row driver stamps each receipt with the revision of the
checkout it runs from and refuses to start without one.  A worker may not
create a worktree of the shared checkout, so a revision is measured from a
source subtree extracted into a tree of its own, and the driver's revision
lookup is answered with that tree's revision.

The diverted 550-cell legacy arm carries a positive control that asserts the
measured value against a stored one, and the row is written before the
assertion runs.  This harness therefore reads the written row whether the
assertion holds or refuses, so a revision whose value moved is measured rather
than rejected.

usage: attribute_diverted_map.py TREE REVISION OUT_DIR CACHE_DIR CONTROL_JSON
"""

# ruff: noqa: E402
import sys
import types
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
REVISION = sys.argv[2]
OUT = Path(sys.argv[3]).resolve()
CACHE = Path(sys.argv[4]).resolve()
CONTROL = Path(sys.argv[5]).resolve()

sys.path.insert(0, str(TREE))
import os

os.chdir(TREE)

import scripts.prototypes.support.measure as driver

driver.subprocess = types.SimpleNamespace(check_output=lambda *a, **k: REVISION + "\n")

print(
    f"ATTRIBUTION_TREE={TREE} REVISION={REVISION} "
    f"DRIVER={Path(driver.__file__).resolve()}",
    flush=True,
)
print(f"ATTRIBUTION_CWD={Path.cwd()}", flush=True)

args = types.SimpleNamespace(
    kind="diverted",
    cells=550,
    out=OUT,
    cache=CACHE,
    control=CONTROL,
    arms=["legacy"],
    oracle_fraction_bound=2e-5,
)
try:
    driver.measure(args)
except Exception as error:  # the positive control refuses a moved value
    print(
        f"ATTRIBUTION_REFUSAL {type(error).__name__}: {error}",
        flush=True,
    )
row = OUT / "diverted-550-legacy.json"
if row.exists():
    import json

    payload = json.loads(row.read_text())
    print(
        "ATTRIBUTION_ROW "
        + json.dumps(
            {
                "revision": payload.get("revision"),
                "case": payload.get("case"),
                "arm": payload.get("arm"),
                "requested_cells": payload.get("requested_cells"),
                "map_relative_sup": payload.get("map_relative_sup"),
                "map_relative_rms": payload.get("map_relative_rms"),
                "completed": payload.get("completed"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
else:
    print(f"ATTRIBUTION_MISSING {row}", flush=True)
