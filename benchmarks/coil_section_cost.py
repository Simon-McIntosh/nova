"""What the exact polygon section costs a coil operator, and what it buys.

A poloidal-field coil is coupled by one of three axisymmetric elements, chosen per
subframe element through the ``segment`` column:

* :class:`nova.biot.circle.Circle` -- a point filament at the section's
  root-mean-square radius, overwritten by the exact closed-form section integral
  inside four section radii and by the target-section average inside two, where the
  target frame declares a section at all;
* :class:`nova.biot.polysection.PolySection` -- the exact section integral on every
  pair, no filament and no seam, evaluated at the target POINT;
* :class:`nova.biot.cylinder.Cylinder` -- the same exact section integral for an
  axis-aligned RECTANGLE, through the four-corner antiderivative rule.

The three differ in two independent ways and this measures both. Where a pair sits
outside the filament seam, ``Circle`` carries a truncation error the other two do
not. Where a target declares a section of its own, ``Circle`` returns the double
integral over that section and the other two return the single integral at its
centre. Which of those two effects a caller meets is decided by the operator, not by
the element: grid and force retain physical point targets, while inductance expands
strictly positive material nodes and contracts them to each original dcoil cell
before applying turns and physical-parent reduction.

The reference
-------------
Reduced inductance is measured against the uniform-current DOUBLE integral over each
coil's whole undivided section, built directly from
:func:`nova.biot.sectionaverage.averaged_greens` and scaled by the turn counts. That
is the quantity the operator is approximating, it shares no code path with the
elements, and it is independent of how either lane discretises. Field and force are
reported as route differences with the standoff at which they occur, because no
comparable independent closed form exists for them.

Costs are wall seconds for one operator assembly, every route in one process behind
a warm-up build. That understates what a first assembly pays -- the allocator and
the compiled kernels are warm by the time any figure is taken -- and it is the same
understatement for every route, taken in the same order at every case, so the
RATIOS are what to read. Run on a compute node: on a shared login node the same
assembly has been observed to vary several fold, which is larger than some of the
differences here.

    python benchmarks/coil_section_cost.py [output.json] [figure.png]
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

from nova.biot.force import Force
from nova.biot.polysection import PolySectionPolicy
from nova.biot.sectionaverage import averaged_greens, section_nodes
from nova.biot.target import ForceTargetPolicy
from nova.frame.coilset import CoilSet

MACHINE = [
    # x, z, dx, dz, nturn, name -- winding-pack outlines at ITER scale: a coil
    # population spanning a slender central solenoid stack and squat outer rings, so
    # the pair separations run from touching to a machine width. The PF1 and CS3U
    # entries are the pair the biot element tests are written on.
    (3.9431, 7.5641, 0.9590, 0.9841, 248.64, "PF1"),
    (8.2851, 6.5398, 0.6501, 0.9091, 115.2, "PF2"),
    (11.9919, 3.2085, 0.6900, 0.9390, 185.9, "PF3"),
    (11.9630, -2.2385, 0.6392, 0.9091, 169.9, "PF4"),
    (8.3908, -6.7283, 0.8125, 0.9538, 216.8, "PF5"),
    (4.2864, -7.5641, 1.5590, 1.1075, 459.4, "PF6"),
    (1.722, 5.313, 0.719, 2.075, 554.0, "CS3U"),
    (1.722, 3.188, 0.719, 2.075, 554.0, "CS2U"),
    (1.722, 1.063, 0.719, 2.075, 554.0, "CS1U"),
    (1.722, -1.063, 0.719, 2.075, 554.0, "CS1L"),
    (1.722, -3.188, 0.719, 2.075, 554.0, "CS2L"),
    (1.722, -5.313, 0.719, 2.075, 554.0, "CS3L"),
]

CASES = {"pair": ("PF1", "CS3U"), "machine": tuple(coil[-1] for coil in MACHINE)}

ROUTES = {
    "circle": ("circle", PolySectionPolicy()),
    "polysection": ("polysection", PolySectionPolicy()),
    "polysection-quadrature": (
        "polysection",
        PolySectionPolicy(exact_kernel="quadrature"),
    ),
    "polysection-banded": (
        "polysection",
        PolySectionPolicy(arrangement="banded"),
    ),
    "cylinder": ("cylinder", PolySectionPolicy()),
}

DCOIL = (-1, -5, -20)
GRID_TARGETS = 1600
FORCE_TARGETS = 20
INDUCTANCE_TARGETS = 100
CURRENT = 40e3  # amperes per turn, the scale an ITER PF conductor carries

INDUCTANCE_ENABLED = 1
"""Non-null solve selector; target resolution comes from positive material nodes."""

HEXAGON_SELF_FLUX = 3.767926648e-05
"""Independent four-dimensional double-integral limit for the audit hexagon."""


def sections(names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Return each named coil's whole undivided section as ``(4, 2)`` r-z corners."""
    section = {}
    for x, z, dx, dz, _, name in MACHINE:
        if name not in names:
            continue
        section[name] = np.array(
            [
                [x - dx / 2, z - dz / 2],
                [x + dx / 2, z - dz / 2],
                [x + dx / 2, z + dz / 2],
                [x - dx / 2, z + dz / 2],
            ]
        )
    return section


def turns(names: tuple[str, ...]) -> np.ndarray:
    """Return the turn count of each named coil, in the order given."""
    count = {coil[-1]: coil[-2] for coil in MACHINE}
    return np.array([count[name] for name in names])


def reference_inductance(names: tuple[str, ...]) -> np.ndarray:
    """Return the uniform-current double-integral inductance matrix [H].

    Each entry is the source coil's current spread at constant density over its whole
    undivided section and the resulting flux averaged over the target coil's whole
    undivided section, scaled by both turn counts. No discretisation of either coil
    enters, so this is the limit both elements are converging towards and neither can
    beat.
    """
    section = sections(names)
    nturn = turns(names)
    matrix = np.empty((len(names), len(names)))
    for column, source in enumerate(names):
        psi = averaged_greens([section[name] for name in names], section[source])[0]
        matrix[:, column] = psi * nturn * nturn[column]
    return matrix


def coilset(route: str, dcoil: float, names: tuple[str, ...]) -> CoilSet:
    """Return a coilset whose every element takes the named route.

    ``ifttt`` is off so the frame's conditional rules cannot reroute the segment;
    the rules themselves are what the ``cylinder`` route measures in isolation.
    """
    segment, policy = ROUTES[route]
    frames = CoilSet(
        dcoil=dcoil,
        nforce=FORCE_TARGETS,
        ninductance=INDUCTANCE_TARGETS,
        coil_polysection_policy=policy,
    )
    for x, z, dx, dz, nturn, name in MACHINE:
        if name not in names:
            continue
        frames.coil.insert(
            x, z, dx, dz, nturn=nturn, name=name, segment=segment, ifttt=False
        )
    return frames


def timed(call) -> tuple[float, object]:
    """Return ``(wall seconds, result)`` for one call."""
    start = time.perf_counter()
    result = call()
    return time.perf_counter() - start, result


def measure_cost(route: str, case: str, dcoil: float) -> dict:
    """Return wall seconds for each operator assembly on one route and case."""
    names = CASES[case]
    frames = coilset(route, dcoil, names)
    element = len(frames.subframe)
    grid, _ = timed(lambda: frames.grid.solve(GRID_TARGETS))
    force, _ = timed(lambda: frames.force.solve(FORCE_TARGETS))
    inductance, _ = timed(lambda: frames.inductance.solve(INDUCTANCE_TARGETS))
    return {
        "route": route,
        "case": case,
        "dcoil": dcoil,
        "element": element,
        "grid_pairs": GRID_TARGETS * element,
        "grid": grid,
        "force": force,
        "inductance": inductance,
    }


def measure_inductance(route: str, case: str, dcoil: float) -> dict:
    """Return the reduced inductance and its deviation from the double integral."""
    names = CASES[case]
    frames = coilset(route, dcoil, names)
    frames.inductance.solve(INDUCTANCE_ENABLED)
    matrix = np.asarray(frames.inductance.Psi)
    want = reference_inductance(names)
    deviation = np.abs(matrix - want) / np.abs(want)
    diagonal = np.eye(len(names), dtype=bool)
    # a reduced operator is only reciprocal to the accuracy of both its lanes, so the
    # asymmetry is reported as its own figure rather than folded into the deviation
    asymmetry = np.abs(matrix - matrix.T) / np.abs(want)
    return {
        "route": route,
        "case": case,
        "dcoil": dcoil,
        "self_max": float(deviation[diagonal].max()),
        "mutual_max": float(deviation[~diagonal].max()),
        "mutual_mean": float(deviation[~diagonal].mean()),
        "asymmetry_max": float(asymmetry.max()),
        "matrix": matrix.tolist(),
        "reference": want.tolist(),
    }


def measure_force(case: str, dcoil: float, routes: tuple[str, ...]) -> dict:
    """Return each route's coil forces and their spread about the exact lane."""
    names = CASES[case]
    force = {}
    for route in routes:
        frames = coilset(route, dcoil, names)
        frames.sloc["coil", "Ic"] = CURRENT
        frames.force.solve(FORCE_TARGETS)
        force[route] = np.column_stack(
            [np.asarray(frames.force.fr), np.asarray(frames.force.fz)]
        )
    exact = force["polysection"]
    scale = np.max(np.abs(exact))
    return {
        "case": case,
        "dcoil": dcoil,
        "coil": list(names),
        "force": {route: value.tolist() for route, value in force.items()},
        "deviation": {
            route: float(np.max(np.abs(value - exact)) / scale)
            for route, value in force.items()
        },
    }


def measure_standoff(dcoil: float) -> dict:
    """Return the field difference between the routes against standoff.

    A radial sweep outwards from one coil's centre, in units of that coil's own
    section bounding radius, which is what both elements band on. This is where a
    filament far field parts company with an exact section, so it localises the
    difference the operator tables above report as one number.
    """
    names = CASES["pair"]
    x, z, dx, dz = MACHINE[0][:4]
    radius = float(np.hypot(dx, dz) / 2)
    standoff = np.linspace(0.05, 8.0, 96)
    points = np.column_stack([x + standoff * radius, np.full(len(standoff), z)])
    field = {}
    for route in ("circle", "polysection"):
        frames = coilset(route, dcoil, names)
        frames.sloc["coil", "Ic"] = CURRENT
        frames.point.solve(points)
        field[route] = np.stack(
            [np.asarray(frames.point.br).ravel(), np.asarray(frames.point.bz).ravel()]
        )
    difference = np.abs(field["circle"] - field["polysection"])
    peak = np.max(np.abs(field["polysection"]))
    return {
        "dcoil": dcoil,
        "radius": radius,
        "standoff": standoff.tolist(),
        "difference": (difference.max(axis=0) / peak).tolist(),
        "worst_standoff": float(standoff[np.argmax(difference.max(axis=0))]),
        "worst": float(difference.max() / peak),
    }


def figure(record: dict, path: pathlib.Path) -> None:
    """Plot the cost ratio and the standoff sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = plt.subplots(1, 2, figsize=(11, 4.2))[1]
    operator = ["grid", "force", "inductance"]
    label = []
    ratio = []
    base = {
        (entry["case"], entry["dcoil"]): entry
        for entry in record["cost"]
        if entry["route"] == "circle"
    }
    for entry in record["cost"]:
        if entry["route"] != "polysection":
            continue
        key = (entry["case"], entry["dcoil"])
        label.append(f"{entry['case']}\ndcoil={entry['dcoil']}")
        ratio.append([entry[name] / base[key][name] for name in operator])
    ratio = np.array(ratio)
    position = np.arange(len(label))
    for index, name in enumerate(operator):
        axes[0].bar(position + 0.27 * index, ratio[:, index], 0.25, label=name)
    axes[0].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_xticks(position + 0.27)
    axes[0].set_xticklabels(label, fontsize=8)
    axes[0].set_ylabel("assembly seconds, exact / banded")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    for entry in record["standoff"]:
        axes[1].semilogy(
            entry["standoff"], entry["difference"], label=f"dcoil={entry['dcoil']}"
        )
    # each sub-element carries its own band, so a mesh puts the seam at four of the
    # SUB-section's radii -- a fraction of the whole coil's, which the axis is in
    axes[1].set_xlabel("standoff [whole-coil section radii]")
    axes[1].set_ylabel("|B difference| / peak |B|")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def audit_record() -> dict:
    """Return the independent audit measurements missing from the cost record."""
    diagonal = [measure_inductance("polysection", "pair", dcoil) for dcoil in DCOIL]
    angle = np.arange(6) * np.pi / 3.0
    hexagon = np.c_[6.2 + 0.075 * np.cos(angle), 0.5 + 0.075 * np.sin(angle)]
    convergence = []
    for order in range(1, 9):
        value = averaged_greens([hexagon], hexagon, order)[0][0]
        convergence.append(
            {
                "order": order,
                "nodes": len(section_nodes(hexagon, order)[1]),
                "value": float(value),
                "relative_error": float(abs(value / HEXAGON_SELF_FLUX - 1.0)),
            }
        )
    return {
        "force_equal_accuracy": equal_accuracy_force_cost(),
        "diagonal": [
            {
                "dcoil": row["dcoil"],
                "self_max": row["self_max"],
                "mutual_max": row["mutual_max"],
                "asymmetry_max": row["asymmetry_max"],
            }
            for row in diagonal
        ],
        "hexagon": convergence,
    }


def equal_accuracy_force_cost() -> dict:
    """Compare the fan and subdivision through the operator at equal accuracy.

    The comparison uses the self-dominated radial-force case. The order-three fan
    and 4096-cell requested subdivision are the first recorded settings on their
    respective ladders that straddle the same worst live-component error. The
    arbiter clips a refining lattice to the target material and applies an
    eighth-order positive rule inside each clipped piece, so neither operator target
    rule enters its value.
    """
    from benchmarks.force_target_rule import arbiter, coilset as force_coilset

    present = (0,)
    truth, arbiter_drift = arbiter(0, present)
    live = np.abs(truth) > 1e-6 * np.max(np.abs(truth))

    def solve(policy: ForceTargetPolicy, number: int) -> dict:
        frames = force_coilset(
            nforce=number if policy.rule == "subdivision" else 1,
            present=present,
        )
        force = Force(
            *frames.frames,
            name="force-cost",
            target_policy=policy,
        )
        elapsed, _ = timed(
            lambda: force.solve(number if policy.rule == "subdivision" else 1)
        )
        value = np.c_[force.fr, force.fz, force.fc][0]
        relative = np.where(live, np.abs(value - truth) / np.abs(truth), 0.0)
        return {
            "rule": policy.rule,
            "setting": number,
            "nodes": len(force),
            "seconds": elapsed,
            "value": value.tolist(),
            "relative_error": relative.tolist(),
            "worst_live_relative_error": float(relative.max()),
        }

    fan = solve(ForceTargetPolicy(rule="positive_material", order=3), 3)
    subdivision = solve(ForceTargetPolicy(rule="subdivision"), 4096)
    return {
        "arbiter": truth.tolist(),
        "arbiter_reported_drift": float(arbiter_drift),
        "live_components": live.tolist(),
        "fan": fan,
        "subdivision": subdivision,
        "node_ratio_subdivision_to_fan": subdivision["nodes"] / fan["nodes"],
        "time_ratio_subdivision_to_fan": subdivision["seconds"] / fan["seconds"],
    }


def audit_figure(record: dict, path: pathlib.Path) -> None:
    """Plot hexagonal target-rule error against node count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = [row["nodes"] for row in record["hexagon"]]
    error = [row["relative_error"] for row in record["hexagon"]]
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.loglog(nodes, error, "o-")
    axis.set_xlabel("positive target nodes")
    axis.set_ylabel("relative self-flux error")
    axis.set_title("Hexagonal target rule against the double-integral arbiter")
    axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main(argv: list[str]) -> None:
    """Run every measurement and write one JSON record."""
    if len(argv) > 1 and argv[1] == "audit":
        record = audit_record()
        print(json.dumps(record, indent=1))
        if len(argv) > 2:
            pathlib.Path(argv[2]).write_text(json.dumps(record, indent=1))
        if len(argv) > 3:
            audit_figure(record, pathlib.Path(argv[3]))
        return
    record: dict[str, list] = {
        "cost": [],
        "inductance": [],
        "force": [],
        "standoff": [],
    }
    coilset("circle", -5, CASES["pair"]).grid.solve(64)  # warm the allocator
    shipped = ("circle", "polysection", "cylinder")
    variant = ("polysection-quadrature", "polysection-banded")
    for case in CASES:
        for dcoil in DCOIL:
            for route in shipped:
                record["cost"].append(measure_cost(route, case, dcoil))
        for route in variant:
            record["cost"].append(measure_cost(route, case, -5))
    # the inductance lane resolves its target frame onto every turn, so the whole
    # coil-count sweep is affordable only at one mesh refinement
    for dcoil in DCOIL:
        for route in shipped + variant:
            record["inductance"].append(measure_inductance(route, "pair", dcoil))
    for route in shipped + variant:
        record["inductance"].append(measure_inductance(route, "machine", -5))
    for dcoil in DCOIL:
        record["force"].append(
            measure_force(
                "machine",
                dcoil,
                ("circle", "polysection", "cylinder", "polysection-banded"),
            )
        )
        record["standoff"].append(measure_standoff(dcoil))
    report(record)
    if len(argv) > 1:
        pathlib.Path(argv[1]).write_text(json.dumps(record, indent=1))
    if len(argv) > 2:
        figure(record, pathlib.Path(argv[2]))


def report(record: dict) -> None:
    """Print the cost and accuracy tables."""
    print("\ncost [s] -- operator assembly")
    print(
        f"{'case':8s} {'dcoil':>6s} {'route':24s} {'n':>5s} {'grid':>9s} "
        f"{'force':>9s} {'induct':>9s}"
    )
    for entry in record["cost"]:
        print(
            f"{entry['case']:8s} {entry['dcoil']:6g} {entry['route']:24s} "
            f"{entry['element']:5d} {entry['grid']:9.3f} {entry['force']:9.3f} "
            f"{entry['inductance']:9.3f}"
        )
    print("\ninductance -- relative deviation from the double integral")
    print(
        f"{'case':8s} {'dcoil':>6s} {'route':24s} {'self':>10s} {'mutual':>10s} "
        f"{'mut mean':>10s} {'asymmetry':>10s} {'vs exact':>10s}"
    )
    exact = {
        (entry["case"], entry["dcoil"]): np.asarray(entry["matrix"])
        for entry in record["inductance"]
        if entry["route"] == "polysection"
    }
    for entry in record["inductance"]:
        matrix = np.asarray(entry["matrix"])
        want = exact[entry["case"], entry["dcoil"]]
        spread = np.max(np.abs(matrix - want) / np.abs(want))
        print(
            f"{entry['case']:8s} {entry['dcoil']:6g} {entry['route']:24s} "
            f"{entry['self_max']:10.2e} {entry['mutual_max']:10.2e} "
            f"{entry['mutual_mean']:10.2e} {entry['asymmetry_max']:10.2e} "
            f"{spread:10.2e}"
        )
    print("\nforce -- deviation from the exact section lane, fraction of peak force")
    for entry in record["force"]:
        deviation = " ".join(
            f"{route}={value:.2e}" for route, value in entry["deviation"].items()
        )
        print(f"{entry['case']:8s} dcoil={entry['dcoil']:6g}  {deviation}")
    print("\nfield -- worst route difference on a radial sweep, fraction of peak |B|")
    for entry in record["standoff"]:
        print(
            f"dcoil={entry['dcoil']:6g}  worst={entry['worst']:.2e} at "
            f"{entry['worst_standoff']:.2f} section radii"
        )


if __name__ == "__main__":
    main(sys.argv)
