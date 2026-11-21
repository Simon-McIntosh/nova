"""Run the present production route's certificate row from the diverted seed.

The route spawns a process batch for its polygon flux moments, so it must be
called from an importable module with a main guard: a spawned child re-imports
the parent's ``__main__``, and a heredoc parent has none, which kills the pool.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prototypes.bank_route_rows.measure import PRESENT_CELLS, present_route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="certificate output root")
    args = parser.parse_args()
    row = present_route(
        "diverted",
        550,
        PRESENT_CELLS[550],
        root=args.root or Path("smfp-present-route"),
    )
    print(
        "PRESENT_ROUTE "
        + json.dumps(
            {
                key: value
                for key, value in row.items()
                if key not in ("coordinates", "flux", "wall_units")
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
