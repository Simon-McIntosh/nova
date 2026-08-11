"""What the force operator's target rule costs, and what it buys.

The force a conductor carries is an area integral over the material its current
occupies,

    F_r = 2 pi I (N/A) INT x B_z dA,    F_z = -2 pi I (N/A) INT x B_R dA,

with B the field of every conductor including the section itself.  Choosing a
target rule is choosing a quadrature rule for that integral, and this driver
measures the two the operator can use.

**The tiling** cuts the section into cells and samples the integrand at each
cell's centroid with its area for a weight.  That is a midpoint rule: second
order in the cell, so its error falls as the reciprocal of the cell count and
buying a decade costs ten times the nodes.

**The fan** places Gauss nodes on a positive decomposition of the same material.
Its integrand is analytic inside the section and only kinked across the boundary
-- the self field's curvature jumps where the current stops -- so it converges
algebraically rather than spectrally, but at an order high enough that a few
tens of nodes reach what thousands of cells do.

Three measurements, because a rate alone decides nothing:

**Convergence against an arbiter neither rule shares.**  The section is cut by a
lattice, each piece clipped to the material and integrated by a high-order fan;
refining the lattice drives the boundary pieces, where the integrand's curvature
jumps, to zero measure.  The sequence converges to the double integral itself,
and its last two rungs bound how far it has left to go.

**Action against reaction.**  Two conductors exchange equal and opposite vertical
force, so a pair's vertical forces sum to zero however either is integrated.  The
residual sum grades a target rule with no reference at all -- the force analogue
of the reciprocity asymmetry the linked-flux rule was graded by.

**Cost through the operator.**  Both rules are driven by ``Force.solve``, so the
timing carries target construction, the kernel over every source-target pair, and
the contraction back to one row per conductor.  Node count is the honest cost
proxy: every node costs one kernel evaluation against every source.

The self term is the one to watch.  A conductor's force on itself is where the
integrand is least smooth, and it dominates the radial force of a real coil, so a
rule that only handles the mutual terms well would look fine on the vertical
force and fail on the radial one.  The tables split it out.
"""

from __future__ import annotations

import time

import numpy as np
import shapely.geometry

from nova.biot.force import Force
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.sectionaverage import section_nodes
from nova.biot.target import ForceTargetPolicy
from nova.frame.coilset import CoilSet

# an ITER-scale ring pair: the outer conductor's radial force is carried by its
# own section, the vertical force of each by the other, so one case exercises
# both limits of the target rule
GEOMETRY = [
    ("PFa", 8.0, 6.5, 0.65, 0.45, 248.0, 45e3),
    ("PFb", 3.9, 7.6, 0.80, 0.60, 553.0, -30e3),
]
SEGMENT_LADDER = (1, 4, 16, 64, 256, 1024)
ORDER_LADDER = (1, 2, 3, 4, 5, 6, 8)
ARBITER_LADDER = (6, 12, 18, 24)
ARBITER_ORDER = 8
COMPONENT = ("Fr", "Fz", "Fc")


def coilset(nforce=1, dcoil=-3, present=None):
    """Return the conductors both rules are measured on.

    ``present`` selects which of them exist, so dropping every conductor but one
    leaves the term a section contributes to its own force.
    """
    frames = CoilSet(nforce=nforce, dcoil=dcoil)
    for index, (name, x, z, dx, dz, nturn, current) in enumerate(GEOMETRY):
        if present is not None and index not in present:
            continue
        frames.coil.insert(x, z, dx, dz, nturn=nturn, Ic=current, name=name)
    return frames


def rectangle(x, z, width, height):
    """Return one conductor's outline."""
    return shapely.geometry.box(
        x - width / 2, z - height / 2, x + width / 2, z + height / 2
    )


def lattice_pieces(polygon, count):
    """Return the material clipped to a ``count`` by ``count`` lattice."""
    low_x, low_z, high_x, high_z = polygon.bounds
    edge_x = np.linspace(low_x, high_x, count + 1)
    edge_z = np.linspace(low_z, high_z, count + 1)
    pieces = []
    for i in range(count):
        for j in range(count):
            box = shapely.geometry.box(
                edge_x[i], edge_z[j], edge_x[i + 1], edge_z[j + 1]
            )
            clipped = box.intersection(polygon)
            for member in getattr(clipped, "geoms", [clipped]):
                if (
                    isinstance(member, shapely.geometry.Polygon)
                    and member.area > 1e-14 * polygon.area
                ):
                    pieces.append(member)
    return pieces


def arbiter(index, present):
    """Return the converged force triple on one conductor and its drift.

    Nodes go to the closed-form polygon kernel in one call per rung, which is
    what makes a double integral affordable: the kernel holds its corner parts
    live across a call and amortises them over every node.
    """
    outline = [rectangle(*row[1:5]) for row in GEOMETRY]
    target = outline[index]
    target_nturn, target_current = GEOMETRY[index][5:]
    low_x, low_z, high_x, high_z = target.bounds
    height = high_z - low_z
    centre = np.asarray(target.centroid.coords[0])

    history = []
    for count in ARBITER_LADDER:
        points, weights = [], []
        for piece in lattice_pieces(target, count):
            piece_points, piece_weights = section_nodes(piece, order=ARBITER_ORDER)
            points.append(piece_points)
            weights.append(piece_weights)
        points = np.concatenate(points)
        weights = np.concatenate(weights)
        radial = np.zeros(len(points))
        vertical = np.zeros(len(points))
        for source_index in present:
            row = GEOMETRY[source_index]
            _, source_radial, source_vertical = polygon_analytic_greens(
                points[:, 0],
                points[:, 1],
                np.asarray(outline[source_index].exterior.coords)[:-1, :2],
            )
            current = row[5] * row[6]
            radial += source_vertical * current
            vertical -= source_radial * current
        radial *= 2.0 * np.pi * points[:, 0]
        vertical *= 2.0 * np.pi * points[:, 0]
        moment = (points[:, 1] - centre[1]) / height * vertical
        density = np.vstack([radial, vertical, moment])
        scale = target_current * target_nturn
        history.append(scale * (density @ weights) / weights.sum())
    drift = np.max(np.abs(history[-1] - history[-2]) / np.abs(history[-1]))
    return history[-1], drift


def solve_tiling(nforce, present):
    """Return the force triple, node count and solve time of the tiling rule."""
    frames = coilset(nforce, present=present)
    start = time.perf_counter()
    frames.force.solve()
    elapsed = time.perf_counter() - start
    force = frames.force
    return np.c_[force.fr, force.fz, force.fc], len(force), elapsed


def solve_fan(order, present):
    """Return the force triple, node count and solve time of the fan rule."""
    frames = coilset(present=present)
    force = Force(
        *frames.frames,
        name="force",
        target_policy=ForceTargetPolicy(rule="positive_material", order=order),
    )
    start = time.perf_counter()
    force.solve(1)
    elapsed = time.perf_counter() - start
    return np.c_[force.fr, force.fz, force.fc], len(force), elapsed


def table(title, present, balance_column):
    """Print one rule comparison over the conductors that are present."""
    truth = np.array([arbiter(index, present)[0] for index in present])
    drift = max(arbiter(index, present)[1] for index in present)
    live = np.abs(truth) > 1e-6 * np.max(np.abs(truth))
    print()
    print("%s  (arbiter lattice %d, drift %.2e)" % (title, ARBITER_LADDER[-1], drift))
    for row, index in zip(truth, present):
        print("  %-4s Fr=%14.7e  Fz=%14.7e  Fc=%14.7e" % (GEOMETRY[index][0], *row))

    def worst(value):
        """Return the worse conductor's relative error, per live component."""
        error = np.where(live, np.abs(value - truth) / np.abs(truth), 0.0)
        return np.max(error, axis=0)

    header = "  rule       nodes   " + "  ".join("%-11s" % c for c in COMPONENT)
    print(header + ("  balance      solve s" if balance_column else "   solve s"))
    for rule, ladder, solve in (
        ("tile", SEGMENT_LADDER, solve_tiling),
        ("fan ", ORDER_LADDER, solve_fan),
    ):
        for step in ladder:
            value, nodes, elapsed = solve(step, present)
            line = "  %s %-6d %5d   %s" % (
                rule,
                step,
                nodes,
                "  ".join("%-11.4e" % e for e in worst(value)),
            )
            if balance_column:
                imbalance = abs(value[:, 1].sum()) / np.max(np.abs(value[:, 1]))
                line += "  %.4e" % imbalance
            print(line + "   %7.3f" % elapsed)


def main():
    """Print the convergence, balance and cost tables."""
    table("both conductors present", (0, 1), balance_column=True)
    for index in range(len(GEOMETRY)):
        table(
            "%s alone: the term a section carries on itself" % GEOMETRY[index][0],
            (index,),
            balance_column=False,
        )


if __name__ == "__main__":
    main()
