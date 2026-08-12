"""Cost and independent accuracy of a finite-cross-section arc grid solve.

Two modes, because they answer different questions and must not share a process:

``time`` (the default)
    wall clock of ``grid.solve`` on a segmented winding inserted with
    ``filament=False``, which is what routes the solve through the Bow kernel --
    the default filament arc never forms an incomplete third kind at all.  One
    variant per interpreter: a second solve in the same process is some 40%
    faster from allocator and JIT warmup rather than from caching, so a
    comparison taken that way invents a difference that is not there.  Run on a
    compute node; login-node timing has a five-fold spread.

``figure``
    the amplitude fold the third kind is reduced by.  What the superseded route
    refused, what it silently answered wrong INSIDE the domain it accepted, and
    the reduced value against an elementary reference across the fold boundary.

``audit``
    the complete Bow element against a direct three-dimensional volume integral
    that shares neither its elliptic reduction nor its fixed-node zeta rule.
"""

import json
import pathlib
import sys
import time

import numpy as np
from scipy.constants import mu_0

RADIUS = 3.945
HEIGHT = 2.0
SECTION = {"rect": (0, 0, 0.06, 0.03)}
SEGMENTS = 10
TARGETS = 2000
QUARTER_TURN = np.pi / 2


def direct_volume_arbiter(
    targets,
    order=64,
    *,
    radius=RADIUS,
    height=HEIGHT,
    width=SECTION["rect"][2],
    depth=SECTION["rect"][3],
    start=-0.2,
    end=0.2,
):
    """Integrate the rectangular arc directly over radius, level and angle.

    This is the defining three-dimensional Biot-Savart volume integral.  It shares
    neither Bow's elliptic reduction nor its fixed-node zeta rule, and therefore
    grades the complete element rather than comparing two reductions of its rows.
    The returned columns are ``Ax, Ay, Az, Bx, By, Bz`` in SI units per ampere.
    """
    node, weight = np.polynomial.legendre.leggauss(order)
    source_radius = radius + 0.5 * width * node
    level = height + 0.5 * depth * node
    half_sweep = 0.5 * (end - start)
    angle = 0.5 * (start + end) + half_sweep * node
    radius_weight = 0.5 * width * weight
    level_weight = 0.5 * depth * weight
    angle_weight = half_sweep * weight
    source_r, source_z, source_phi = np.meshgrid(
        source_radius, level, angle, indexing="ij"
    )
    volume_weight = (
        radius_weight[:, None, None]
        * level_weight[None, :, None]
        * angle_weight[None, None, :]
        / (width * depth)
    )
    source = np.stack(
        [
            source_r * np.cos(source_phi),
            source_r * np.sin(source_phi),
            source_z,
        ],
        axis=-1,
    )
    tangent = np.stack(
        [
            -source_r * np.sin(source_phi),
            source_r * np.cos(source_phi),
            np.zeros_like(source_r),
        ],
        axis=-1,
    )
    result = []
    for target in np.asarray(targets, dtype=float):
        separation = target - source
        distance = np.linalg.norm(separation, axis=-1)
        vector = (
            mu_0
            / (4.0 * np.pi)
            * np.sum(
                volume_weight[..., None] * tangent / distance[..., None],
                axis=(0, 1, 2),
            )
        )
        field = (
            mu_0
            / (4.0 * np.pi)
            * np.sum(
                volume_weight[..., None]
                * np.cross(tangent, separation)
                / distance[..., None] ** 3,
                axis=(0, 1, 2),
            )
        )
        result.append(np.concatenate([vector, field]))
    return np.asarray(result)


def bow_arbiter_record() -> dict:
    """Return Bow's worst sampled error against the direct volume integral."""
    from nova.biot.biotframe import Source, Target
    from nova.biot.bow import Bow
    from nova.frame.coilset import CoilSet

    angle = np.array([-0.2, 0.0, 0.2])
    radius = 2.0
    coilset = CoilSet()
    coilset.winding.insert(
        np.column_stack(
            [radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)]
        ),
        {"rect": (0.0, 0.0, 0.08, 0.04)},
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
    )
    frame = coilset.subframe
    source = Source(
        {column: np.asarray(frame[column]) for column in frame.columns},
        index=list(frame.index),
    )
    targets = np.array(
        [
            [2.5, 0.1, 0.3],
            [2.045 * np.cos(0.205), 2.045 * np.sin(0.205), 0.025],
        ]
    )
    target = Target({"x": targets[:, 0], "y": targets[:, 1], "z": targets[:, 2]})
    bow = Bow(source, target, turns=[False, False], reduce=[False, False])
    measured = np.concatenate([mu_0 * bow.Avector[:, 0], bow.Bvector[:, 0]], axis=-1)
    reference = direct_volume_arbiter(
        targets, order=64, radius=radius, height=0.0, width=0.08, depth=0.04
    )
    prior = direct_volume_arbiter(
        targets, order=48, radius=radius, height=0.0, width=0.08, depth=0.04
    )
    row_scale = np.max(np.abs(reference), axis=-1)
    relative = np.max(np.abs(measured - reference), axis=-1) / row_scale
    refinement = np.max(np.abs(reference - prior), axis=-1) / row_scale
    return {
        "targets": targets.tolist(),
        "production": measured.tolist(),
        "direct_volume_order_64": reference.tolist(),
        "production_relative_error": relative.tolist(),
        "worst_production_relative_error": float(relative.max()),
        "arbiter_relative_change_48_to_64": refinement.tolist(),
        "worst_arbiter_relative_change_48_to_64": float(refinement.max()),
    }


def subtracted_angle_third_kind(n, phi, m):
    """Return the third kind through the superseded fold, for the comparison.

    The residual amplitude is FORMED as ``phi - k pi`` with a floored,
    half-turn-offset count, and Carlson's first argument is recovered from the
    sine as ``1 - sin^2``.  Both are what the shipped routine no longer does;
    kept here so the figure can measure the arrangement it replaced rather than
    describe it.  Returns ``nan`` where the residual leaves the closed quarter
    the arrangement asserted, which is where it used to abort.
    """
    from scipy.special import elliprf, elliprj

    turns = (phi + QUARTER_TURN) // np.pi
    residual = phi - turns * np.pi
    sine = np.sin(residual)
    x = 1.0 - sine**2
    y = 1.0 - m * sine**2
    unit = np.ones_like(x)
    complete = (
        elliprf(np.zeros_like(m), 1.0 - m, unit)
        + elliprj(np.zeros_like(m), 1.0 - m, unit, 1.0 - n) * n / 3.0
    )
    third = elliprj(x, y, unit, 1.0 - n * sine**2)
    value = (
        2.0 * turns * complete + sine * elliprf(x, y, unit) + sine**3 * third * n / 3.0
    )
    return np.where(abs(residual) <= QUARTER_TURN, value, np.nan)


def zero_modulus_third_kind(n, phi):
    """Return the elementary third kind at a vanishing modulus, for any amplitude.

    ``Pi(n, phi, 0) = arctan(sqrt(1-n) tan phi)/sqrt(1-n)`` over one quarter, and
    the same half-turn quasi-periodicity carries it over the rest:
    ``Pi(n, t pi + d, 0) = 2 t Pi(n, 0) + Pi(n, d, 0)`` with the complete value
    ``pi/(2 sqrt(1-n))``.  Taken through the residual's TANGENT so it keeps its
    relative accuracy where the amplitude approaches a quarter turn.
    """
    root = np.sqrt(1.0 - n)
    turns = np.round(phi / np.pi)
    residual = phi - turns * np.pi
    return turns * np.pi / root + np.arctan(root * np.tan(residual)) / root


def winding(coilset, segments=SEGMENTS):
    """Insert a closed ring of finite-cross-section arc segments."""
    azimuth = np.linspace(-np.pi, np.pi, 1 + 3 * segments)
    points = np.stack(
        [
            RADIUS * np.cos(azimuth),
            RADIUS * np.sin(azimuth),
            np.full_like(azimuth, HEIGHT),
        ],
        axis=-1,
    )
    for index in range(segments):
        coilset.winding.insert(
            points[3 * index : 1 + 3 * (index + 1)],
            SECTION,
            nturn=1,
            minimum_arc_nodes=4,
            Ic=1,
            filament=False,
            ifttt=False,
        )
    return coilset


def time_grid_solve(segments=SEGMENTS, targets=TARGETS):
    """Report the wall clock of one fresh-process bow grid solve."""
    from nova.frame.coilset import CoilSet

    coilset = winding(CoilSet(field_attrs=["Ay", "Br", "Bz"]), segments)
    sections = np.unique(coilset.subframe.segment.values)
    build = time.perf_counter()
    coilset.grid.solve(targets, 0.5)
    build = time.perf_counter() - build
    shape = coilset.grid.shape
    field = np.asarray(coilset.grid.br)
    print(f"segment labels      {list(sections)}")
    print(f"sources             {len(coilset.subframe)}")
    print(f"grid                {shape[0]} x {shape[1]} = {field.size} targets")
    print(f"grid.solve          {build:.3f} s")
    print(f"finite              {bool(np.all(np.isfinite(field)))}")
    print(f"|Br| range          {abs(field).min():.3e} to {abs(field).max():.3e}")
    return build


def figure(path="bow_grid_solve.png"):
    """Render the amplitude fold: what was refused, what was wrong, what agrees."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.special

    from nova.biot.constants import Constants

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    fig.subplots_adjust(hspace=0.34, wspace=0.26)

    # (a) the ulp neighbourhood of every half-turn boundary, accepted or refused
    steps = np.arange(-6, 7)
    turns = np.arange(0, 15)
    offsets = np.array(
        [
            [
                np.nextafter(
                    turn * np.pi - QUARTER_TURN, np.inf if step > 0 else -np.inf
                )
                if step
                else turn * np.pi - QUARTER_TURN
                for step in steps
            ]
            for turn in turns
        ]
    )
    for row, turn in enumerate(turns):
        for column, step in enumerate(steps):
            phi = offsets[row, column]
            for _ in range(max(abs(step) - 1, 0)):
                phi = np.nextafter(phi, np.inf if step > 0 else -np.inf)
            offsets[row, column] = phi
    count = (offsets + QUARTER_TURN) // np.pi
    refused = abs(offsets - count * np.pi) > QUARTER_TURN
    axes[0, 0].pcolormesh(
        steps - 0.5, turns - 0.5, refused.astype(float), cmap="Reds", vmin=0, vmax=1.4
    )
    axes[0, 0].set_xlabel("representable steps from the half-turn boundary")
    axes[0, 0].set_ylabel("half turns of amplitude")
    axes[0, 0].set_title(
        f"(a) subtracted-angle fold: {refused.sum()} of {refused.size}\n"
        "amplitudes REFUSED at the quarter-turn boundary",
        fontsize=10,
    )

    # the folded pair covers the same amplitudes with a non-negative cosine
    parity = 1.0 - 2.0 * abs(np.remainder(np.round(offsets / np.pi), 2.0))
    cosine = parity * np.cos(offsets)
    resolution = 4 * Constants.eps * np.maximum(1.0, abs(offsets))
    axes[0, 1].pcolormesh(
        steps - 0.5,
        turns - 0.5,
        (cosine < -resolution).astype(float),
        cmap="Reds",
        vmin=0,
        vmax=1.4,
    )
    axes[0, 1].set_xlabel("representable steps from the half-turn boundary")
    axes[0, 1].set_ylabel("half turns of amplitude")
    axes[0, 1].set_title(
        "(b) folded pair: every amplitude carried,\n"
        "the residual cosine non-negative throughout",
        fontsize=10,
    )

    # (c) the silent defect: relative error inside the accepted domain
    gap = 10.0 ** -np.linspace(1.0, 14.0, 60)
    amplitude = QUARTER_TURN - gap
    for colour, m in zip("C0 C1 C3".split(), [0.4, 0.9, 1 - 1e-6]):
        modulus = np.full_like(amplitude, m)
        reference = scipy.special.ellipkinc(amplitude, m)
        for style, route in [
            ("--", subtracted_angle_third_kind),
            ("-", lambda n, phi, m: Constants.ellippinc(n, phi, m)),
        ]:
            value = route(np.zeros_like(amplitude), amplitude, modulus)
            axes[1, 0].loglog(
                gap,
                np.maximum(abs(value - reference) / abs(reference), 1e-17),
                style,
                color=colour,
                label=f"m = {m:g}" if style == "-" else None,
            )
    axes[1, 0].axhline(np.finfo(float).eps, color="0.6", lw=0.8, zorder=0)
    axes[1, 0].set_xlabel("quarter turn minus amplitude")
    axes[1, 0].set_ylabel("relative error against $F(\\varphi\\,|\\,m)$")
    axes[1, 0].set_title(
        "(c) inside the accepted domain: dashed, the subtracted\n"
        "angle; solid, the folded pair (vanishing characteristic)",
        fontsize=10,
    )
    axes[1, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].invert_xaxis()

    # (d) across a fold boundary, against two independent references
    boundary = QUARTER_TURN + np.pi
    phi = boundary + np.linspace(-0.4, 0.4, 161)
    floor = 1e-17
    worst = 0.0
    cases = [
        ("C0", 0.0, 1 - 1e-6, lambda n, p, m: scipy.special.ellipkinc(p, m[0])),
        ("C2", 0.3, 0.0, lambda n, p, m: zero_modulus_third_kind(n[0], p)),
    ]
    for colour, n, m, reference in cases:
        characteristic, modulus = np.full_like(phi, n), np.full_like(phi, m)
        exact = reference(characteristic, phi, modulus)
        folded = Constants.ellippinc(characteristic, phi, modulus)
        retired = subtracted_angle_third_kind(characteristic, phi, modulus)
        worst = max(worst, float(np.max(abs(folded / exact - 1.0))))
        axes[1, 1].semilogy(
            phi - boundary,
            np.maximum(abs(retired / exact - 1.0), floor),
            "--",
            color=colour,
            lw=1.1,
            label=f"n = {n:g}, m = {m:g}: subtracted angle",
        )
        axes[1, 1].semilogy(
            phi - boundary,
            np.maximum(abs(folded / exact - 1.0), floor),
            "-",
            color=colour,
            lw=1.4,
            label=f"n = {n:g}, m = {m:g}: folded pair",
        )
    axes[1, 1].axhline(np.finfo(float).eps, color="0.6", lw=0.8, zorder=0)
    axes[1, 1].axvline(0.0, color="k", lw=0.8, ls=":")
    axes[1, 1].set_ylim(floor, 1e-4)
    axes[1, 1].set_xlabel("amplitude minus the fold boundary $3\\pi/2$ (radians)")
    axes[1, 1].set_ylabel("relative deviation from the reference")
    axes[1, 1].set_title(
        "(d) across the fold boundary: the folded pair holds\n"
        "round-off through it against either reference",
        fontsize=10,
    )
    axes[1, 1].legend(fontsize=7, loc="upper right")

    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")
    print(f"panel (a) refused {refused.sum()} of {refused.size} amplitudes")
    print(f"panel (d) worst relative deviation {worst:.3e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "time"
    if mode == "figure":
        figure(*sys.argv[2:])
    elif mode == "audit":
        record = bow_arbiter_record()
        print(json.dumps(record, indent=1))
        if len(sys.argv) > 2:
            pathlib.Path(sys.argv[2]).write_text(json.dumps(record, indent=1))
    else:
        time_grid_solve()
