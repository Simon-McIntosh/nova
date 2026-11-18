"""Score the closed-shell ladder: shielding factor per arm and per probe.

S = psi_air / psi_iron: the poloidal flux at a probe without iron over the flux at
the same probe with it, which is the reduction the shell causes.  For a linear
spherical shell the exterior dipole flux is reduced by exactly S, so the ratio is
the measured shielding factor, and it must be the same at every exterior probe.

The iron outline is a polygon and imas-efit derives one breakpoint per ten
polygon vertices (capped at 500), so S is a function of the outline resolution.
Two arms with different vertex counts but the same knot count are back-to-back
numbers a reader cannot see through, so every arm is reported as a row of the
tag's resolution series.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.iron_linear_magnetisation.controls import exact_shielding  # noqa: E402

TOLERANCE = 0.02
CONVERGENCE_TOLERANCE = 0.005

# The exact factor comes from the same owner make_pulse.py writes the pulse
# from, so a scorer cannot drift from the case it scores.
MU_R = 1000.0
EXACT = {ratio: round(exact_shielding(ratio, MU_R), 6) for ratio in (0.9, 0.5)}

# Exterior probes score the shielding; the interior one is the control where the
# shell must not shield.
EXTERIOR = [(2.0, 0.0), (3.0, 1.0), (1.2, 0.0)]
INTERIOR = [(0.5, 0.0)]

TAGS = (("r090", 0.9), ("r050", 0.5))

# The solver's relativeError setting per arm.  The arm names carry it: e2 and fine
# were run at 1e-2, e4 and every vN arm at 1e-4 (job5/job6/job9 in the run logs).
SOLVER_TOLERANCE = {"e2": 1e-2, "fine": 1e-2}
DEFAULT_SOLVER_TOLERANCE = 1e-4

# The pairs whose members differ in that setting and nothing else, so their S
# difference measures how far the solver tolerance alone moves the answer.
SOLVER_PAIRS = (("e2", "e4"), ("fine", "v40"))

IRON_CONVERGENCE = re.compile(r"relErr=\s*(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)")
IRON_BMOD = re.compile(r"Bmax=\s*(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)")


def arc_vertices(arm: str) -> int:
    """Polygon vertices per half-arc, which sets the outline resolution.

    e2 and e4 are the solver-tolerance arms on the 20-vertex outline; fine is
    the same outline at 40; the vN arms carry N explicitly.
    """
    suffix = arm.rsplit("-", 1)[-1]
    if suffix in ("e2", "e4"):
        return 20
    if suffix == "fine":
        return 40
    if suffix.startswith("v") and suffix[1:].isdigit():
        return int(suffix[1:])
    return 0


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def arm_suffix(arm: str) -> str:
    return arm.rsplit("-", 1)[-1]


def solver_tolerance(arm: str) -> float:
    return SOLVER_TOLERANCE.get(arm_suffix(arm), DEFAULT_SOLVER_TOLERANCE)


def solver_lines(directory: Path, arm: str) -> dict:
    """The arm's own solver record: its last iron relErr and its last Bmax.

    Read from the arm's log rather than retyped, so a re-solved arm cannot leave
    the receipt quoting the previous run's numbers.
    """
    path = directory / arm
    if not path.is_file():
        return {}
    rel_err = bmax = None
    calls = 0
    for line in path.read_text(errors="replace").splitlines():
        if "IRON_CONVERGENCE" in line:
            found = IRON_CONVERGENCE.search(line)
            if found:
                rel_err = float(found.group(1))
        elif "IRON_BMOD" in line:
            found = IRON_BMOD.search(line)
            if found:
                bmax = float(found.group(1))
                calls += 1
    return {
        "tolerance": solver_tolerance(arm),
        "rel_err": rel_err,
        "bmax": bmax,
        "bmod_calls": calls,
    }


def psi_map(score: dict) -> dict[tuple[float, float], float] | None:
    if "psi" not in score:
        return None
    return {(p["r"], p["z"]): p["psi"] for p in score["psi"]}


def shielding(air: dict, iron: dict, points: list[tuple[float, float]]) -> dict:
    values = []
    for point in points:
        a, b = air.get(point), iron.get(point)
        if a is None or b is None or b == 0:
            continue
        values.append({"probe": list(point), "psi_air": a, "psi_iron": b, "S": a / b})
    if not values:
        # Every exterior probe fell outside the grid: the arm carries no
        # measurement, and the caller records it rather than dividing by zero.
        return {"values": [], "mean_S": None}
    return {"values": values, "mean_S": sum(v["S"] for v in values) / len(values)}


def main() -> None:
    scores = Path(sys.argv[1])
    arms = {
        path.stem[len("score-") :]: load(path)
        for path in sorted(scores.glob("score-*.json"))
    }

    table = []
    refused = []
    for tag, ratio in TAGS:
        air = psi_map(arms.get(f"{tag}-air", {}))
        if air is None:
            table.append({"tag": tag, "ratio": ratio, "error": "no air arm"})
            continue
        series = []
        for name, arm in sorted(arms.items()):
            if not name.startswith(f"{tag}-") or name == f"{tag}-air":
                continue
            iron = psi_map(arm)
            if iron is None:
                # A slice the reconstruction refused has no profiles_2d and so can
                # carry no measurement; it is recorded, never scored.
                refused.append({"arm": name, "reason": arm.get("error")})
                continue
            outer = shielding(air, iron, EXTERIOR)
            if outer["mean_S"] is None:
                refused.append(
                    {
                        "arm": name,
                        "reason": "no exterior probe lies inside the flux grid",
                    }
                )
                continue
            series.append(
                {
                    "arm": name,
                    "arc_vertices": arc_vertices(name),
                    "S_exterior": [round(v["S"], 3) for v in outer["values"]],
                    "S_mean": round(outer["mean_S"], 3),
                    "spread_over_mean": round(
                        (
                            max(v["S"] for v in outer["values"])
                            - min(v["S"] for v in outer["values"])
                        )
                        / outer["mean_S"],
                        4,
                    ),
                    "interior_S": [
                        round(v["S"], 4)
                        for v in shielding(air, iron, INTERIOR)["values"]
                    ],
                    "rel_error": round(outer["mean_S"] / EXACT[ratio] - 1.0, 5),
                    "exact_S": EXACT[ratio],
                    "solver": solver_lines(scores, name),
                    "convergence": {
                        "iterations": arm.get("iterations"),
                        "gs_deviation": arm.get("gs_deviation"),
                    },
                }
            )
        scored = [s for s in series if "S_mean" in s]
        best = max(scored, key=lambda s: s["arc_vertices"], default=None)
        table.append(
            {
                "tag": tag,
                "ratio": ratio,
                "exact_S": EXACT[ratio],
                "series": series,
                "finest": best,
                "verdict": (
                    "reproduces"
                    if best and abs(best["rel_error"]) <= TOLERANCE
                    else "misses"
                ),
            }
        )

    # Discretisation convergence: the step between the two finest series entries.
    convergence = []
    for entry in table:
        series = [s for s in entry.get("series", []) if "S_mean" in s]
        if len(series) < 2:
            continue
        series.sort(key=lambda s: (s["arc_vertices"], s["arm"]))
        last, prev = series[-1], series[-2]
        convergence.append(
            {
                "tag": entry["tag"],
                "steps": [[s["arm"], s["S_mean"]] for s in series],
                "last_step_delta": round(last["S_mean"] / prev["S_mean"] - 1.0, 4),
                "passed": abs(last["S_mean"] / prev["S_mean"] - 1.0)
                < CONVERGENCE_TOLERANCE,
            }
        )

    # Levers: each pair differs in one setting and nothing else, so its S step is
    # what that setting alone moves.  A lever whose step reaches the 0.5 % ceiling
    # cannot be shown converged, and a ratio whose levers all reach it is not
    # converged by any evidence here.
    levers = []
    for entry in table:
        series = {arm["arm"]: arm for arm in entry.get("series", []) if "S_mean" in arm}
        for first, second in SOLVER_PAIRS:
            a, b = f"{entry['tag']}-{first}", f"{entry['tag']}-{second}"
            if a not in series or b not in series:
                continue
            delta = series[b]["S_mean"] / series[a]["S_mean"] - 1.0
            levers.append(
                {
                    "tag": entry["tag"],
                    "lever": "solver tolerance",
                    "pair": [a, b],
                    "settings": [
                        series[a]["solver"]["tolerance"],
                        series[b]["solver"]["tolerance"],
                    ],
                    "S": [series[a]["S_mean"], series[b]["S_mean"]],
                    "delta": round(delta, 5),
                    "passed": abs(delta) < CONVERGENCE_TOLERANCE,
                }
            )
    for row in convergence:
        levers.append(
            {
                "tag": row["tag"],
                "lever": "boundary refinement",
                "pair": [row["steps"][-2][0], row["steps"][-1][0]],
                "settings": [],
                "S": [row["steps"][-2][1], row["steps"][-1][1]],
                "delta": row["last_step_delta"],
                "passed": row["passed"],
            }
        )

    print(
        json.dumps(
            {
                "tolerance": TOLERANCE,
                "convergence_tolerance": CONVERGENCE_TOLERANCE,
                "exterior_probes": [list(p) for p in EXTERIOR],
                "interior_probes": [list(p) for p in INTERIOR],
                "table": table,
                "refused": refused,
                "convergence": convergence,
                "levers": levers,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
