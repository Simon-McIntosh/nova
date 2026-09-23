"""Name the solver call chain of each large live-map site in optimized HLO.

The module's stack-frame table maps each instruction's ``stack_frame_id`` to
its Python frames. Map-internal instructions carry the traceback of the map's
first trace, so a site is named by the instructions its caller emitted in the
same scope, whose frames in the fixed-point and reduced-Newton modules identify
the solver code that applied the map there.
"""

import collections
import json
import re
import sys

text = open(sys.argv[1]).read()
receipt = json.load(open(sys.argv[2]))
head, _, _body = text.partition("\nENTRY")


def table(name, following):
    block = head.split("\n" + name + "\n", 1)[1].split("\n" + following, 1)[0]
    return block.splitlines()


files = {
    int(i): n.strip('"').split("/")[-1]
    for i, n in (x.split(" ", 1) for x in table("FileNames", "FunctionNames"))
}
functions = {
    int(i): n.strip('"')
    for i, n in (x.split(" ", 1) for x in table("FunctionNames", "FileLocations"))
}
locations = {}
for row in table("FileLocations", "StackFrames"):
    index, rest = row.split(" ", 1)
    fields = dict(re.findall(r"(\w+)=(-?\d+)", rest))
    locations[int(index)] = (
        files[int(fields["file_name_id"])],
        functions[int(fields["function_name_id"])],
        int(fields["line"]),
    )
frames = {}
for row in head.split("\nStackFrames\n", 1)[1].splitlines():
    match = re.match(r"(\d+) \{file_location_id=(\d+) parent_frame_id=(\d+)\}", row)
    if not match:
        break
    index, location, parent = map(int, match.groups())
    frames[index] = (location, parent)


def chain(frame):
    seen = []
    while frame in frames and frame not in [s[0] for s in seen]:
        location, parent = frames[frame]
        seen.append((frame, locations[location]))
        if parent == frame:
            break
        frame = parent
    return [loc for _, loc in seen]


MARKERS = ("compact_clipped_current_moments", "read_qualification")
big = {row["site"]: row for row in receipt["rows"] if row["instructions"] > 1000}
pattern = re.compile(r'op_name="([^"]*)" stack_frame_id=(\d+)')
callers = {site: collections.Counter() for site in big}
for match in pattern.finditer(text):
    op_name, frame = match.group(1), int(match.group(2))
    if any(marker in op_name for marker in MARKERS):
        continue
    scope = op_name.rsplit("/", 1)[0]
    if scope in callers:
        solver = tuple(
            f"{fn}:{line}"
            for file, fn, line in chain(frame)
            if file in ("fixed_point.py", "reduced_newton.py")
        )
        if solver:
            callers[scope][solver[:3]] += 1
rows = []
for site, row in sorted(big.items(), key=lambda item: -item[1]["instructions"]):
    top = callers[site].most_common(2)
    rows.append(
        dict(
            site=site,
            instructions=row["instructions"],
            shared_body=row["shared_body"],
            kinds=row["kinds"],
            callers=[[list(c), n] for c, n in top],
        )
    )
    print(row["instructions"], row["shared_body"], top[:1])
json.dump(rows, open(sys.argv[3], "w"), indent=2)
