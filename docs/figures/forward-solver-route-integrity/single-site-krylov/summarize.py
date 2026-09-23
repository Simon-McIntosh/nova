"""Build qualified-step.json from the single-site job's censuses and logs.

The per-caller attribution of ``_qualified_krylov_step`` is taken two ways:
the live-map sites whose two leading solver call chains pass through the
qualified step (or the stream serving it), and the non-map instructions whose
own stack chain passes through it.
"""

import json
from pathlib import Path
import re

D = Path("docs/figures/forward-solver-route-integrity/single-site-krylov")
OS = Path("docs/figures/forward-solver-route-integrity/operator-sharing")
BASE = {300: 302553, 1000: 304913}
NAMES = ("_qualified_krylov_step", "_single_site_krylov")
MARKERS = ("compact_clipped_current_moments", "read_qualification")


def site_share(callers_path):
    rows = json.loads(Path(callers_path).read_text())
    sites = [
        row
        for row in rows
        if any(any(n in f for f in chain for n in NAMES) for chain, _ in row["callers"])
    ]
    return dict(
        sites=len(sites),
        instructions=sum(row["instructions"] for row in sites),
        rows=[dict(instructions=r["instructions"], kinds=r["kinds"], callers=r["callers"]) for r in sites],
    )


def frame_share(hlo_path):
    text = Path(hlo_path).read_text()
    head = text.partition("\nENTRY")[0]

    def table(name, following):
        return head.split("\n" + name + "\n", 1)[1].split("\n" + following, 1)[0].splitlines()

    functions = {
        int(i): n.strip('"')
        for i, n in (x.split(" ", 1) for x in table("FunctionNames", "FileLocations"))
    }
    locations = {}
    for row in table("FileLocations", "StackFrames"):
        index, rest = row.split(" ", 1)
        fields = dict(re.findall(r"(\w+)=(-?\d+)", rest))
        locations[int(index)] = functions[int(fields["function_name_id"])]
    frames = {}
    for row in head.split("\nStackFrames\n", 1)[1].splitlines():
        match = re.match(r"(\d+) \{file_location_id=(\d+) parent_frame_id=(\d+)\}", row)
        if not match:
            break
        index, location, parent = map(int, match.groups())
        frames[index] = (location, parent)
    memo = {}

    def through(frame):
        if frame in memo:
            return memo[frame]
        seen, hit, current = set(), False, frame
        while current in frames and current not in seen:
            seen.add(current)
            location, parent = frames[current]
            if any(n in locations.get(location, "") for n in NAMES):
                hit = True
                break
            if parent == current:
                break
            current = parent
        memo[frame] = hit
        return hit

    count = 0
    for match in re.finditer(r'op_name="([^"]*)" stack_frame_id=(\d+)', text):
        if any(m in match.group(1) for m in MARKERS):
            continue
        count += through(int(match.group(2)))
    return count


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


receipt = dict(base_revision="52bfefc0a0f02412d8910aaccb632bfbd9fd43b5", ceilings=BASE, runs={})
base_callers = site_share(OS / "callers-candidate-300.json")
receipt["base_52bfefc0_300_qualified_step_sites"] = {
    k: base_callers[k] for k in ("sites", "instructions")
}
for mode in ("stream", "per-site"):
    for cells in (300, 1000):
        program = read(D / f"program-{mode}-{cells}.json")
        if program is None:
            receipt["runs"][f"{mode}-{cells}"] = None
            continue
        run = dict(
            revision=program["revision"],
            job_id=program["job_id"],
            compile_seconds=program["compile_seconds"],
            optimized_instructions=program["optimized_instructions"],
            base_optimized_instructions=BASE[cells],
            delta_against_base=program["optimized_instructions"] - BASE[cells],
            below_base=program["optimized_instructions"] < BASE[cells],
        )
        callers = D / f"callers-{mode}-{cells}.json"
        if callers.exists():
            share = site_share(callers)
            run["qualified_step_map_sites"] = share["sites"]
            run["qualified_step_map_instructions"] = share["instructions"]
            run["qualified_step_map_site_rows"] = share["rows"]
        run["qualified_step_non_map_instructions"] = frame_share(program["hlo_path"])
        receipt["runs"][f"{mode}-{cells}"] = run
for cells in (300, 1000):
    stream, per_site = receipt["runs"].get(f"stream-{cells}"), receipt["runs"].get(f"per-site-{cells}")
    if stream and per_site:
        receipt[f"per_caller_delta_{cells}"] = dict(
            total=stream["optimized_instructions"] - per_site["optimized_instructions"],
            map_sites=stream.get("qualified_step_map_sites", 0) - per_site.get("qualified_step_map_sites", 0),
            map_instructions=stream.get("qualified_step_map_instructions", 0)
            - per_site.get("qualified_step_map_instructions", 0),
            non_map_instructions=stream["qualified_step_non_map_instructions"]
            - per_site["qualified_step_non_map_instructions"],
        )
vmap = D / "vmap-cost.log"
if vmap.exists():
    text = vmap.read_text()
    match = re.search(r"\{\n.*?\n\}", text, re.S)
    receipt["vmap_cost"] = json.loads(match.group(0)) if match else None
for arm in ("dense", "elementwise"):
    text = (D / f"toy-identity-{arm}.log").read_text()
    receipt[f"toy_{arm}"] = dict(
        rows=[json.loads(line) for line in text.splitlines() if line.startswith("{")],
        summary=[line for line in text.splitlines() if line.startswith("TOY_") or line.startswith("EXIT=")],
    )
(D / "qualified-step.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k.startswith(("per_caller", "base_52"))}, indent=1))
for name, run in receipt["runs"].items():
    if run:
        print(name, run["optimized_instructions"], run["delta_against_base"], run.get("qualified_step_map_instructions"), run["qualified_step_non_map_instructions"])
