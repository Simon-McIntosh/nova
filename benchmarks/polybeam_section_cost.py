"""Cost and accuracy of the straight polygon-section closed form.

What the prism costs against ``Beam``, which is the same straight segment for a
RECTANGULAR section through a corner tensor rather than a contour sum.  The two
answers to expect are structural:

* ``Beam`` evaluates eight corners of a box and six ratios once, whatever the
  section, so its cost is a constant;
* the prism evaluates two axial limits at two ends of every EDGE, so its cost is
  linear in the corner count -- which is what makes the rectangle the fair
  comparison and the hexagon the price of the capability.

Neither carries a quadrature, an elliptic integral or a moment recursion, so both
should land far below the arc pair: a straight path has no curvature and therefore
no toroidal integration.

The variants come in two families and they are not interchangeable.  ``prism-``
calls the reduction directly, which is the sharpest comparison between two
evaluations.  ``beam-`` and ``polybeam-`` go through the FRAME -- a coilset, a
source assembly, the local-to-global transform and the matrix storage -- which is
the comparison a caller actually pays and the only one in which the two element
classes are measured on the same terms.

One variant per process: repeats inside a single interpreter warm the allocator
and understate the first-build cost operator assembly actually pays.  Run as
``python benchmarks/polybeam_section_cost.py <variant>``; ``list`` prints the
variants, and one JSON record goes to stdout.

Run the cost variants on a compute node.  On a shared login node the same call has
been observed to vary five-fold, which is larger than every difference this is
meant to resolve.  ``collect`` folds a file of those records into the artifact and
its figure; ``study`` writes the accuracy and convergence artifacts, which are
deterministic and need no node.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time

import numpy as np

R0, Z0 = 6.2, 0.0
CELL_RADIUS = 0.06  # hexagon circumradius of a ~560-cell ITER plasma mesh
RECT = (0.1, 0.08)  # a representative rectangular winding-pack section
LIMITS = (-0.4, 0.6)
N_TARGET = 512
REPEATS = 3
CHORD = (0.1, 0.9)  # a straight segment's own end azimuths on the path radius

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/polybeam-prism-section"


def hexagon(x0=R0, y0=Z0, radius=CELL_RADIUS):
    """Return regular hexagon vertices, counter-clockwise."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([x0 + radius * np.cos(angle), y0 + radius * np.sin(angle)])


def rectangle(x0=R0, y0=Z0, width=RECT[0], height=RECT[1]):
    """Return rectangle vertices, counter-clockwise."""
    return np.array(
        [
            [x0 - width / 2, y0 - height / 2],
            [x0 + width / 2, y0 - height / 2],
            [x0 + width / 2, y0 + height / 2],
            [x0 - width / 2, y0 + height / 2],
        ]
    )


SECTIONS = {"rectangle": rectangle, "hexagon": hexagon}

FRAME_SECTIONS = {
    "rectangle": {"rect": (0, 0, *RECT)},
    "hexagon": {"hex": (0, 0, np.sqrt(3.0) * CELL_RADIUS, 2.0 * CELL_RADIUS)},
}


def targets(count=N_TARGET, reach=8.0 * CELL_RADIUS):
    """Return a target cloud ringing the section, in the prism's own frame.

    A spiral rather than a grid, so no target is degenerate with a corner and none
    repeats another's contour distance, and the axial coordinate crosses both end
    planes so neither limit sits always on one side.
    """
    turn = np.linspace(0.0, 6.0 * np.pi, count)
    grow = 0.3 + 0.7 * turn / turn[-1]
    return (
        R0 + reach * grow * np.cos(turn),
        Z0 + reach * grow * np.sin(turn),
        np.linspace(LIMITS[0] - 0.3, LIMITS[1] + 0.3, count),
    )


def _prism(name: str) -> tuple:
    """Return the reduction's per-pair cost on one section."""
    from nova.biot.polybeam import polygon_beam_greens

    vertices = SECTIONS[name]()
    x, y, z = targets()

    def once():
        polygon_beam_greens(x, y, z, vertices, *LIMITS)

    return _timed(once, len(x)), len(x)


def straight_coilset(section: str, segment: str | None):
    """Return a coilset carrying one thickened straight segment, and its cloud.

    ``minimum_arc_nodes`` of zero is what keeps the path straight -- an arc fit
    would otherwise take a chord's own end points as an arc -- and both elements
    are named EXPLICITLY, because the frame's routing sends a thickened LINE to
    ``Beam`` whatever its section and would measure the same element twice.
    """
    from nova.frame.coilset import CoilSet

    angle = np.asarray(CHORD)
    path = np.stack(
        [R0 * np.cos(angle), R0 * np.sin(angle), Z0 * np.ones_like(angle)], axis=-1
    )
    coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay"])
    coilset.winding.insert(
        path,
        FRAME_SECTIONS[section],
        nturn=1,
        Ic=1,
        minimum_arc_nodes=0,
        filament=False,
        ifttt=False,
    )
    if segment is not None:
        coilset.subframe.loc[:, "segment"] = segment
    turn = np.linspace(0.0, 6.0 * np.pi, N_TARGET)
    grow = 0.3 + 0.7 * turn / turn[-1]
    radius = R0 + 8.0 * CELL_RADIUS * grow * np.cos(turn)
    azimuth = np.linspace(CHORD[0] - 0.3, CHORD[1] + 0.3, N_TARGET)
    height = Z0 + 8.0 * CELL_RADIUS * grow * np.sin(turn)
    cloud = np.stack(
        [radius * np.cos(azimuth), radius * np.sin(azimuth), height], axis=-1
    )
    return coilset, cloud


def _frame(section: str, segment: str | None) -> tuple:
    """Return an element class's per-pair cost through the frame, same cloud."""
    coilset, cloud = straight_coilset(section, segment)

    def once():
        coilset.point.solve(cloud)

    return _timed(once, len(cloud)), len(cloud)


def _element(section: str, element: str) -> tuple:
    """Return one element class's per-pair cost with the operator left out.

    Between the two families: the class is built and its six global rows formed
    from a source frame and a target cloud that already exist, so the local frame
    and the rotation are paid and the coilset, the segment dispatch, the turn
    scaling and the matrix storage are not.  A fresh instance per call, because
    both classes cache their own geometry and timing a warm one would measure an
    attribute lookup.
    """
    from nova.biot.beam import Beam
    from nova.biot.biotframe import Source, Target
    from nova.biot.polybeam import PolyBeam

    coilset, cloud = straight_coilset(section, element)
    frame = coilset.subframe
    column = {name: np.asarray(frame[name]) for name in frame.columns}
    index = list(np.asarray(frame.index))
    factory = {"beam": Beam, "polybeam": PolyBeam}[element]

    def once():
        instance = factory(
            Source(column, index=index),
            Target({"x": cloud[:, 0], "y": cloud[:, 1], "z": cloud[:, 2]}),
        )
        instance.Avector
        instance.Bvector

    return _timed(once, len(cloud)), len(cloud)


def _beam_reduction(section: str) -> tuple:
    """Return ``Beam``'s own row expressions per pair, off the frame.

    The like-for-like against ``prism-``: the three local rows and their corner
    contraction, evaluated by ``Beam``'s own code on a source frame and a target
    cloud that already exist.  ``Beam`` caches its geometry and its rows do not, so
    what is timed here is the transcendentals and the contraction with the corner
    offsets warm, where the prism's reduction forms its own geometry inside every
    call -- the asymmetry runs against the prism, so a comparison it wins is real.

    The section is immaterial to the cost: ``Beam`` evaluates the box its width and
    height bound whatever the corners are, which is the whole difference between a
    constant cost and one linear in the corner count.
    """
    from nova.biot.beam import Beam
    from nova.biot.biotframe import Source, Target

    coilset, cloud = straight_coilset(section, "beam")
    frame = coilset.subframe
    instance = Beam(
        Source(
            {name: np.asarray(frame[name]) for name in frame.columns},
            index=list(np.asarray(frame.index)),
        ),
        Target({"x": cloud[:, 0], "y": cloud[:, 1], "z": cloud[:, 2]}),
    )
    instance.theta  # warm the cached geometry, as a repeated solve would find it

    def once():
        for row in ("_Az_hat", "_Bx_hat", "_By_hat"):
            instance._intergrate(getattr(instance, row))

    return _timed(once, len(cloud)), len(cloud)


def _timed(call, pairs: int) -> float:
    """Return the median of ``REPEATS`` timings, in microseconds per pair.

    The median rather than the minimum: the first call in a process carries the
    import-time and allocator warmup that operator assembly pays too, and the
    minimum would hide exactly that.
    """
    elapsed = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - start)
    return 1e6 * statistics.median(elapsed) / pairs


VARIANTS = {
    **{f"prism-{name}": (lambda name=name: _prism(name)) for name in SECTIONS},
    **{
        f"{element}-{name}": (lambda name=name, element=element: _frame(name, element))
        for element in ("beam", "polybeam")
        for name in FRAME_SECTIONS
    },
    **{
        f"reduction-beam-{name}": (lambda name=name: _beam_reduction(name))
        for name in FRAME_SECTIONS
    },
    **{
        f"element-{element}-{name}": (
            lambda name=name, element=element: _element(name, element)
        )
        for element in ("beam", "polybeam")
        for name in FRAME_SECTIONS
    },
}


# ---------------------------------------------------------------------------
# The accuracy artifacts, which are deterministic.


def accuracy_envelope(name: str = "hexagon", count: int = 96):
    """Return the closed form's deviation from the converged section quadrature.

    Measured against distance to the section CONTOUR, which is what the accuracy
    tracks -- a target a centroid-distance away can sit on an edge of a thin
    section, and a target inside the conductor has no centroid distance worth
    banding by.  The reference is the acceptance gate's own rule, imported rather
    than reimplemented so the figure and the test cannot disagree.
    """
    sys.path.insert(0, str(ROOT / "tests"))
    import shapely.geometry
    from test_biotpolybeam import section_average

    vertices = SECTIONS[name]()
    scale = float(np.max(np.ptp(vertices, axis=0)))
    centre = vertices.mean(axis=0)
    boundary = shapely.geometry.LinearRing(vertices)
    turn = np.linspace(0.0, 8.0 * np.pi, count)
    reach = scale * np.logspace(-3.5, 1.2, count)
    x = centre[0] + reach * np.cos(turn)
    y = centre[1] + reach * np.sin(turn)
    z = np.linspace(LIMITS[0] - 0.2, LIMITS[1] + 0.2, count)
    from nova.biot.polybeam import polygon_beam_greens

    got = np.stack(polygon_beam_greens(x, y, z, vertices, *LIMITS))
    want = section_average(x, y, z, vertices, *LIMITS)
    distance = np.array(
        [boundary.distance(shapely.geometry.Point(a, b)) for a, b in zip(x, y)]
    )
    envelope = np.abs(got - want) / np.max(np.abs(want), axis=-1)[:, np.newaxis]
    inside = np.array(
        [
            shapely.geometry.Polygon(vertices).contains(shapely.geometry.Point(a, b))
            for a, b in zip(x, y)
        ]
    )
    return distance / scale, envelope, inside, scale


def beam_parity(count: int = 256):
    """Return the rectangle parity residual against ``Beam``, row by row.

    Both closed forms of the same integral, so this is the one comparison with no
    truncation in it at all -- the residual is the two evaluation orders' round-off
    and nothing else.
    """
    from nova.biot.polybeam import polygon_beam_greens

    vertices = rectangle()
    low, high = np.min(vertices, axis=0), np.max(vertices, axis=0)
    rng = np.random.default_rng(11)
    x = R0 + rng.uniform(-0.6, 0.6, count)
    y = Z0 + rng.uniform(-0.6, 0.6, count)
    z = rng.uniform(LIMITS[0] - 0.4, LIMITS[1] + 0.4, count)
    ones = np.ones(count)
    ui = (np.stack([low[0] * ones, high[0] * ones]) - x)[:, None, None]
    vj = (np.stack([low[1] * ones, high[1] * ones]) - y)[None, :, None]
    wk = (np.stack([LIMITS[0] * ones, LIMITS[1] * ones]) - z)[None, None]
    radius = np.sqrt(ui**2 + vj**2 + wk**2)
    theta = {
        1: wk / np.sqrt(ui**2 + vj**2),
        2: ui / np.sqrt(vj**2 + wk**2),
        3: vj / np.sqrt(wk**2 + ui**2),
        4: vj * wk / (ui * radius),
        5: wk * ui / (vj * radius),
        6: ui * vj / (wk * radius),
    }
    stack = [
        ui * vj * np.arcsinh(theta[1])
        + vj * wk * np.arcsinh(theta[2])
        + wk * ui * np.arcsinh(theta[3])
        - 0.5
        * (
            ui**2 * np.arctan(theta[4])
            + vj**2 * np.arctan(theta[5])
            + wk**2 * np.arctan(theta[6])
        ),
        -ui * np.arcsinh(theta[1])
        - wk * np.arcsinh(theta[2])
        + vj * np.arctan(theta[5]),
        vj * np.arcsinh(theta[1])
        + wk * np.arcsinh(theta[3])
        - ui * np.arctan(theta[4]),
    ]
    index = np.arange(1, 3)
    sign = (-1.0) ** (index[:, None, None] + index[None, :, None] + index[None, None])
    area = float(np.prod(high - low))
    beam = np.stack(
        [np.einsum("ijk,ijk...", sign, data) / (4 * np.pi * area) for data in stack]
    )
    prism = np.stack(polygon_beam_greens(x, y, z, vertices, *LIMITS))
    return np.abs(prism - beam) / np.max(np.abs(beam), axis=-1)[:, np.newaxis]


def swept_limit(radii=(3.0, 1e01, 1e02, 1e03, 1e04, 1e05, 1e06), length=0.5):
    """Return ``PolyBow``'s departure from the prism as its major radius grows.

    At fixed section and fixed arc LENGTH, so the sweep shrinks as the radius
    grows and the arc straightens onto the prism.  Both the closing and the floor
    are the result: the arc's rows are differences of quantities of order the
    squared major radius, so past some radius the comparison stops measuring the
    limit and starts measuring the arc kernel's own conditioning.
    """
    from scipy.constants import mu_0

    from nova.biot.polybeam import polygon_beam_greens
    from nova.biot.polygonarc import polygon_arc_greens

    base = hexagon(0.0, 0.0)
    offset = np.array([[0.10, 0.05], [-0.08, 0.02], [0.0, 0.12], [0.012, 0.006]])
    rows = []
    for radius in radii:
        vertices = base + np.array([radius, 0.0])
        target_r, target_z = radius + offset[:, 0], offset[:, 1]
        arc = (
            np.stack(
                polygon_arc_greens(
                    target_r,
                    target_z,
                    np.zeros_like(target_r),
                    vertices,
                    -0.5 * length / radius,
                    0.5 * length / radius,
                )
            )
            / mu_0
        )
        prism = np.stack(
            polygon_beam_greens(
                target_r,
                -target_z,
                np.zeros_like(target_r),
                vertices * np.array([1.0, -1.0]),
                -0.5 * length,
                0.5 * length,
            )
        )
        want = np.stack([prism[0], prism[1], -prism[2]])
        got = np.stack([arc[1], arc[2], arc[4]])
        scale = np.max(np.abs(want), axis=-1)[:, np.newaxis]
        rows.append(np.max(np.abs(got - want) / scale, axis=-1))
    return np.asarray(radii, dtype=float), np.asarray(rows)


# ---------------------------------------------------------------------------
# Artifacts and figures.


def _axes(*args, **kwargs):
    """Return a figure and axes with the plotting backend fixed to a file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.subplots(*args, **kwargs)


def study() -> None:
    """Write the accuracy and convergence artifacts, and their two figures."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = ("A_z", "B_x", "B_y")

    distance, envelope, inside, scale = accuracy_envelope()
    near = distance <= 0.5
    record = {
        "section": "hexagon",
        "section_extent": scale,
        "reference": "signed-fan section quadrature of the filament kernel, 48 nodes",
        "rows": list(rows),
        "worst_near_contour": float(np.max(envelope[:, near])),
        "worst_far_from_contour": float(np.max(envelope[:, ~near])),
        "worst_inside_the_conductor": float(np.max(envelope[:, inside])),
        "band_near": 1e-12,
        "band_far": 1e-11,
        "targets": int(len(distance)),
        "targets_inside_the_conductor": int(np.count_nonzero(inside)),
    }
    (FIGURES / "accuracy.json").write_text(json.dumps(record, indent=2) + "\n")

    plt, (figure, axes) = _axes(figsize=(7.4, 4.6))
    floor = np.finfo(float).eps
    for row, label in enumerate(rows):
        for mask, marker, note in ((~inside, "o", ""), (inside, "x", " (inside)")):
            axes.loglog(
                distance[mask],
                np.maximum(envelope[row][mask], floor),
                marker,
                color=f"C{row}",
                ms=4,
                mfc="none" if marker == "o" else None,
                label=f"{label}{note}",
                alpha=0.85,
            )
    axes.axhline(1e-12, color="C3", ls="--", lw=1)
    axes.axhline(1e-11, color="C3", ls=":", lw=1)
    axes.axvline(0.5, color="0.5", lw=1)
    axes.text(
        0.55,
        2.4e-12,
        "acceptance bands, 1e-12 near / 1e-11 far",
        color="C3",
        fontsize=8,
    )
    axes.set_xlabel("distance to section contour / section extent")
    axes.set_ylabel("|closed form - converged quadrature| / row scale")
    axes.set_title(
        "prism accuracy envelope, hexagonal section\n"
        "round-off at every distance, on the contour and inside the conductor"
    )
    axes.legend(loc="upper left", fontsize=7, ncols=3)
    axes.set_ylim(1e-16, 4e-11)
    axes.grid(True, which="both", alpha=0.25)
    figure.tight_layout(pad=0.4)
    figure.savefig(FIGURES / "accuracy-envelope.svg")
    plt.close(figure)

    parity = beam_parity()
    radius, closing = swept_limit()
    convergence = {
        "beam_parity": {
            "reference": "nova.biot.beam.Beam, same rectangle and same targets",
            "rows": list(rows),
            "worst_by_row": [float(value) for value in np.max(parity, axis=-1)],
        },
        "swept_limit": {
            "reference": "polygon_arc_greens, fixed arc length 0.5 m",
            "rows": ["A_phi", "B_r", "B_z"],
            "radius": radius.tolist(),
            "worst_by_row": closing.tolist(),
        },
    }
    (FIGURES / "convergence.json").write_text(json.dumps(convergence, indent=2) + "\n")

    plt, (figure, axes) = _axes(1, 2, figsize=(9.6, 4.0))
    for row, label in enumerate(rows):
        axes[0].semilogy(np.maximum(parity[row], floor), ".", label=label, alpha=0.7)
    axes[0].axhline(1e-13, color="C3", ls="--", lw=1, label="band 1e-13")
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("|prism - Beam| / row scale")
    axes[0].set_title("exact parity on the rectangle they share")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.25)
    for row, label in enumerate(("A_phi", "B_r", "B_z")):
        axes[1].loglog(radius, closing[:, row], "o-", label=label, alpha=0.8)
    axes[1].loglog(radius, 8e-02 / radius, "k--", lw=1, label="first order in 1/R")
    axes[1].set_xlabel("arc major radius [m]")
    axes[1].set_ylabel("|PolyBow - prism| / row scale")
    axes[1].set_title("the swept element straightens onto the prism")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.25)
    figure.tight_layout(pad=0.4)
    figure.savefig(FIGURES / "convergence.svg")
    plt.close(figure)
    print(json.dumps({"wrote": sorted(p.name for p in FIGURES.iterdir())}))


def collect(source: str) -> None:
    """Fold a file of per-variant records into the cost artifact and its figure.

    One line per PROCESS, so a variant appears once for each fresh interpreter it
    was run in; the artifact carries the median of those and the spread across
    them, because a single process's figure on a shared machine is not a
    measurement and the spread is what says whether the differences below are
    resolved at all.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    record: dict[str, list[float]] = {}
    pairs: dict[str, int] = {}
    for line in pathlib.Path(source).read_text().splitlines():
        if not line.strip().startswith("{"):
            continue
        entry = json.loads(line)
        record.setdefault(entry["variant"], []).append(entry["microseconds_per_pair"])
        pairs[entry["variant"]] = entry["pairs"]
    cost = {name: statistics.median(value) for name, value in record.items()}
    payload = {
        "microseconds_per_pair": {
            name: round(value, 3) for name, value in sorted(cost.items())
        },
        "spread": {
            name: round(max(value) - min(value), 3)
            for name, value in sorted(record.items())
        },
        "processes": {name: len(value) for name, value in sorted(record.items())},
        "pairs": dict(sorted(pairs.items())),
        "repeats_within_a_process": REPEATS,
        "note": (
            "fresh process per measurement, median of three first-run timings within "
            "each, median across processes; measured on a sun_debug compute node"
        ),
    }
    for prefix, key in (("", "framed_ratio"), ("element-", "element_ratio")):
        for section in FRAME_SECTIONS:
            pair = (f"{prefix}beam-{section}", f"{prefix}polybeam-{section}")
            if all(name in cost for name in pair):
                payload.setdefault(key, {})[section] = round(
                    cost[pair[1]] / cost[pair[0]], 3
                )
    if "prism-rectangle" in cost and "prism-hexagon" in cost:
        payload["corner_scaling"] = round(
            cost["prism-hexagon"] / cost["prism-rectangle"], 3
        )
    if "reduction-beam-rectangle" in cost:
        payload["reduction_ratio"] = {
            name.split("-", 1)[1]: round(
                cost[name] / cost["reduction-beam-rectangle"], 3
            )
            for name in ("prism-rectangle", "prism-hexagon")
            if name in cost
        }
    (FIGURES / "cost.json").write_text(json.dumps(payload, indent=2) + "\n")

    order = [
        name
        for name in (
            "beam-rectangle",
            "polybeam-rectangle",
            "beam-hexagon",
            "polybeam-hexagon",
            "element-beam-rectangle",
            "element-polybeam-rectangle",
            "element-beam-hexagon",
            "element-polybeam-hexagon",
            "reduction-beam-rectangle",
            "prism-rectangle",
            "prism-hexagon",
        )
        if name in cost
    ]
    plt, (figure, axes) = _axes(figsize=(7.6, 4.2))
    value = [cost[name] for name in order]
    spread = [max(record[name]) - min(record[name]) for name in order]
    family = {
        "reduction-beam": "0.4",
        "prism": "0.6",
        "beam": "C0",
        "polybeam": "C1",
        "element-beam": "C0",
        "element-polybeam": "C1",
    }
    shade = [family[name.rsplit("-", 1)[0]] for name in order]
    bar = axes.bar(range(len(order)), value, yerr=spread, color=shade, capsize=3)
    axes.bar_label(bar, fmt="%.1f", fontsize=8, padding=6)
    axes.set_xticks(range(len(order)))
    axes.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    axes.set_ylabel("microseconds per target-source pair")
    axes.set_yscale("log")
    axes.set_title(
        "straight-segment element cost, sun_debug node\n"
        "grey is the bare reduction, element- the class alone, the rest the frame too"
    )
    axes.grid(True, axis="y", alpha=0.25)
    figure.tight_layout(pad=0.4)
    figure.savefig(FIGURES / "cost.svg")
    plt.close(figure)
    print(json.dumps(payload))


def main(argument: list[str]) -> None:
    """Run one variant, or one of the two artifact modes."""
    name = argument[0] if argument else "list"
    if name == "list":
        print("\n".join(sorted(VARIANTS) + ["collect <records>", "study"]))
        return
    if name == "study":
        study()
        return
    if name == "collect":
        collect(argument[1])
        return
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; try 'list'")
    cost, pairs = VARIANTS[name]()
    print(
        json.dumps(
            {
                "variant": name,
                "microseconds_per_pair": round(cost, 4),
                "pairs": pairs,
                "repeats": REPEATS,
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
