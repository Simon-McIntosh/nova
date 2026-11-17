"""Score the closed-shell ladder: shielding factor per arm and per probe.

S is the ratio of the poloidal flux at a probe computed with iron to the flux at
the same probe computed without iron.  For a linear spherical shell the exterior
dipole flux is reduced by exactly S, so the ratio is the measured shielding
factor; it is scored as reproducing the exact value if within 2 %.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOLERANCE = 0.02
CONVERGENCE_TOLERANCE = 0.005

# exact_shielding from make_pulse.py; the plan rounds the 0.5 case to 195.1
EXACT = {0.9: 61.101838, 0.5: 195.055750}
# Probes shared with the section 1 prototype: (2,0) and (3,1) are outside the
# shell, where the shielding reduces the coil flux; (1.2,0) and (0.5,0) are
# inside it, where the shell does not shield and the ratio is a control.
PROBE_ORDER = [(2.0, 0.0), (3.0, 1.0), (1.2, 0.0), (0.5, 0.0)]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def psi_map(score: dict) -> dict[tuple[float, float], float]:
    if "psi" not in score:
        raise SystemExit(f"no psi in {score}")
    return {(p["r"], p["z"]): p["psi"] for p in score["psi"]}


def main() -> None:
    scores = Path(sys.argv[1])
    arms = {}
    for path in sorted(scores.glob("score-*.json")):
        name = path.stem[len("score-") :]
        arms[name] = load(path)

    rows = []
    for tag, ratio in (("r090", 0.9), ("r050", 0.5)):
        air = arms.get(f"{tag}-air")
        if air is None:
            continue
        air_psi = psi_map(air)
        for arm in ("e2", "e4", "fine"):
            iron = arms.get(f"{tag}-{arm}")
            if iron is None:
                continue
            iron_psi = psi_map(iron)
            for point in PROBE_ORDER:
                a, b = air_psi.get(point), iron_psi.get(point)
                if a is None or b is None or b == 0:
                    continue
                measured = a / b
                exact = EXACT[ratio]
                rows.append(
                    {
                        "arm": f"{tag}-{arm}",
                        "ratio": ratio,
                        "probe": list(point),
                        "psi_air": a,
                        "psi_iron": b,
                        "S": measured,
                        "exact_S": exact,
                        "rel_error": measured / exact - 1.0,
                        "verdict": (
                            "reproduces"
                            if abs(measured / exact - 1.0) <= TOLERANCE
                            else "misses"
                        ),
                    }
                )
    convergence = []
    for tag in ("r090", "r050"):
        base = [r for r in rows if r["arm"] == f"{tag}-e2"]
        for arm in ("e4", "fine"):
            for a in base:
                b = next(
                    (
                        r
                        for r in rows
                        if r["arm"] == f"{tag}-{arm}" and r["probe"] == a["probe"]
                    ),
                    None,
                )
                if b is None:
                    continue
                convergence.append(
                    {
                        "tag": tag,
                        "ratio": a["ratio"],
                        "probe": a["probe"],
                        "arm": arm,
                        "S_base": a["S"],
                        "S_arm": b["S"],
                        "delta": b["S"] / a["S"] - 1.0,
                        "passed": abs(b["S"] / a["S"] - 1.0) < CONVERGENCE_TOLERANCE,
                    }
                )

    payload = {
        "tolerance": TOLERANCE,
        "convergence_tolerance": CONVERGENCE_TOLERANCE,
        "exact": {str(k): v for k, v in EXACT.items()},
        "arms": {
            name: {
                key: value
                for key, value in arm.items()
                if key
                in (
                    "iterations",
                    "gs_deviation",
                    "psi_axis",
                    "psi_boundary",
                    "ip",
                    "error",
                )
            }
            for name, arm in arms.items()
        },
        "rows": rows,
        "convergence": convergence,
    }
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
