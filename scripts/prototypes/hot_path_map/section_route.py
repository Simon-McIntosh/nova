"""Check the documented hot-path route against a measured line map.

``nova/equilibrium/AGENTS.md`` states, per route, the modules that route
executes. This reads the three marker blocks it carries --

    shared      modules both routes execute
    forward     modules only the forward route executes
    inverse     modules only the inverse route executes

-- so a route's measured set is ``shared | <route>``, and compares each set
with the modules the line map measured as executing at least one line. A
module filed under a route it does not execute, and a measured module left
unfiled, are both differences; the check names every one and exits nonzero.

``--add BLOCK=MODULE`` files one extra module into a marker block of a
temporary in-memory copy, which is how the check can be shown to fail: a
module no route executes is reported by name.

``--root`` additionally asserts that every module the section names exists in
the tree.
"""

import argparse
import json
import re
from pathlib import Path

MARKERS = ("shared", "forward", "inverse")
MODULE = re.compile(r"nova/[A-Za-z0-9_./]+\.py")


def block(text, name):
    start = f"<!-- hot-route:{name} -->"
    stop = f"<!-- /hot-route:{name} -->"
    if start not in text or stop not in text:
        raise SystemExit(f"section carries no {name!r} marker block")
    return set(MODULE.findall(text.split(start, 1)[1].split(stop, 1)[0]))


def add_to_block(text, name, module):
    start = f"<!-- hot-route:{name} -->"
    stop = f"<!-- /hot-route:{name} -->"
    head, rest = text.split(start, 1)
    body, tail = rest.split(stop, 1)
    return f"{head}{start}{body} {module} {stop}{tail}"


def measured(map_path):
    payload = json.loads(Path(map_path).read_text())
    routes = {"forward": set(), "inverse": set()}
    for row in payload["modules"]:
        for route in routes:
            cell = row.get(route)
            if cell and cell.get("executed"):
                routes[route].add(row["module"])
    return payload, routes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--section", type=Path, required=True)
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="BLOCK=MODULE",
        help="file BLOCK=MODULE into a marker block of a temporary copy",
    )
    args = parser.parse_args()

    text = args.section.read_text()
    for entry in args.add:
        name, module = entry.split("=", 1)
        text = add_to_block(text, name, module)
    declared = {name: block(text, name) for name in MARKERS}

    payload, routes = measured(args.map_path)
    named = declared["shared"] | declared["forward"] | declared["inverse"]

    differences = []
    for route in ("forward", "inverse"):
        claimed = declared["shared"] | declared[route]
        for module in sorted(claimed - routes[route]):
            differences.append(f"{route}: named, not executed on {route}: {module}")
        for module in sorted(routes[route] - claimed):
            differences.append(f"{route}: executed, not named: {module}")
    both = declared["forward"] & declared["inverse"]
    for module in sorted(both):
        differences.append(f"filed forward-only and inverse-only: {module}")
    shared_and_one = declared["shared"] & (declared["forward"] | declared["inverse"])
    for module in sorted(shared_and_one):
        differences.append(f"filed both shared and single-route: {module}")
    if args.root:
        for module in sorted(named):
            if not (args.root / module).exists():
                differences.append(f"named, absent from the tree: {module}")

    print("ROUTES " + json.dumps(dict(
        revision=payload.get("revision"),
        forward=len(routes["forward"]),
        inverse=len(routes["inverse"]),
        shared=len(declared["shared"]),
        forward_only=len(declared["forward"]),
        inverse_only=len(declared["inverse"]),
        named=len(named),
        differences=len(differences),
    )))
    for line in differences:
        print("DIFF", line)
    print(f"ROUTE-CHECK differences={len(differences)}")
    raise SystemExit(1 if differences else 0)


if __name__ == "__main__":
    main()