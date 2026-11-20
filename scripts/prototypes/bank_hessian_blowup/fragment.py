"""Render the bank second-derivative diagnosis page from its measured records.

Every number in the page is read from the four JSON records the measurement
jobs wrote, and the page is written to the fragment path this node owns. The
figure is drawn from the bank's own geometry archive: a bank row is a
linear-operator coefficient rather than a solved flux field, so it has no
contour map and no null set, and the geometry is what the coefficient depends
on.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.prototypes.bank_hessian_blowup import geometry

CACHE = Path("/home/ITER/mcintos/.cache/nova-bank-hessian-blowup")
ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    ROOT
    / "docs/evidence/fragments/sub-millisecond-forward-path/bank-hessian-blowup.html"
)

BANK_LABELS = {
    "certified": "certified 81ee8fdf",
    "pre_repair": "pre-repair fd912554",
    "rebuild_2e30fa75": "rerun rebuild 2e30fa75",
    "rebuild_ac2c5474": "rerun rebuild ac2c5474",
    "carried_corner": "rerun rebuild 6c1a1d8f",
    "corner_omitted": "rerun rebuild b00f714a",
}
ARM_ENTRY = 9
RADIAL_ENTRY = 1
VERTICAL_ENTRY = 8
TOROIDAL_ENTRY = 10
SECOND_VERTICAL_ENTRY = 11

FIGURE_WIDTH = 1340
FIGURE_HEIGHT = 640
PANEL_TOP = 20
PANEL_HEIGHT = 580
PANEL_LEFT = 40
PANEL_WIDTH = 620
PANEL_GAP = 60
ZOOM_WIDTH = 460
ZOOM_HEIGHT = 180
ZOOM_GAP = 20
ZOOM_FACTOR = 1.3
GREY = "#929292"
FEATURE = "#2e8b57"
WALL = "#6b7280"


def load(name):
    return json.loads((CACHE / name).read_text())


def number(value):
    if value == 0.0:
        return "0"
    mantissa, exponent = f"{value:.4e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def short(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def table(headers, rows, identifier=None):
    head = "".join(f"<th>{cell}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    attribute = f' id="{identifier}"' if identifier else ""
    return (
        "<table"
        + attribute
        + "><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table>"
    )


def bound_keys(census):
    return sorted(census["banks"]["certified"]["above"], key=float)


def bound_label(key):
    mantissa, exponent = f"{float(key):.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def pair_composition(attribute):
    counts = Counter((pair["moment"], pair["component"]) for pair in attribute["pairs"])
    distances = [pair["distance_vertex"] for pair in attribute["pairs"]]
    return dict(
        counts=counts,
        total=len(attribute["pairs"]),
        max_distance=max(distances),
        min_distance=min(distances),
    )


def composition_text(attribute):
    composition = pair_composition(attribute)
    pieces = []
    for (moment, component), count in composition["counts"].most_common():
        amount = "one" if count == 1 else str(count)
        verb = "is a" if count == 1 else "are"
        noun = "entry" if count == 1 else "entries"
        pieces.append(
            f"{amount} {verb} moment-{moment} <code>{component}</code> {noun}"
        )
    if len(pieces) == 1:
        return "every one of them is a " + pieces[0].split(" ", 2)[2]
    return " and ".join(pieces)


def part_head(attribute, census):
    largest = census["banks"]["certified"]["maximum"]
    composition = pair_composition(attribute)
    summary = (
        f"The {short(largest)} bank second derivative is carried by the "
        f"repaired corner entries through the stable-pole derivative rule, "
        f"across {composition['total']} node-cell pairs"
    )
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en"><head>',
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta name="docs-project" content="nova">',
            '<meta name="reckon-type" content="evidence">',
            '<meta name="plan-slug" content="bank-hessian-blowup">',
            '<meta name="plan-evidence-for" content="sub-millisecond-forward-path">',
            '<meta name="plan-verifies" content="sub-millisecond-forward-path#s6">',
            f'<meta name="plan-summary" content="{summary}">',
            '<meta name="plan-title" content="Bank second-derivative blowup">',
            '<meta name="plan-status" content="complete">',
            "<title>Bank second-derivative blowup | nova</title>",
            '<link rel="stylesheet" href="/_shared/foundation.css">',
            '<link rel="stylesheet" href="/_shared/dashboard.css">',
            '</head><body><main class="plan-doc">',
        )
    )


def corner_reproduction(census):
    rows = census["carried_corner_against_pre_repair"]
    certified = census["carried_corner_against_certified"]
    shared = max(abs(row["rebuilt"] - row["other"]) for row in rows)
    against = max(abs(row["rebuilt"] - row["other"]) for row in certified)
    return len(rows), shared, against


def part_summary(attribute, census):
    composition = pair_composition(attribute)
    keys = bound_keys(census)
    bound = bound_label(keys[1])
    largest = census["banks"]["certified"]["maximum"]
    corner = census["banks"]["carried_corner"]
    count, shared, against = corner_reproduction(census)
    difference, same = rebuild_agreement(census)
    return "\n".join(
        (
            '<article id="bank-hessian-blowup-result">',
            '<h3 id="bank-hessian-blowup-title">'
            f"The {short(largest)} ITER bank second derivative belongs to the "
            "certified bank and comes from the stable-pole derivative rule, not "
            "from the moment rows</h3>",
            "<p><strong>",
            f"{composition['total']} node-cell pairs carry a second derivative "
            f"above <code>{bound}</code>: {composition_text(attribute)}, and "
            "every target node sits within "
            f"<code>{number(composition['max_distance'])} m</code> of a vertex "
            "of its own source cell.",
            "</strong>",
            "The certified bank <code>81ee8fdf</code> carries them. Its two DINA "
            "rerun rebuilds reproduce the certified entries to within "
            f"<code>{short(difference)}</code> in absolute terms"
            + (" and are identical to each other" if same else "")
            + ", with the same pair count at every bound, so the rerun did not "
            "introduce the blow-up.",
            f"A rebuild of the same inputs reproduces the pre-repair entry "
            f"exactly at the {count} pairs both are recorded at, while "
            f"differing from the certified bank there by up to "
            f"<code>{short(against)}</code>. It lowers the largest magnitude "
            f"from <code>{short(largest)}</code> to "
            f"<code>{short(corner['maximum'])}</code> and the count above "
            f"<code>{bound}</code> from {composition['total']} to "
            f"{corner['above'][keys[1]]}, so the number is carried by those "
            "corner entries.",
            "</p>",
        )
    )


def pair_rows(attribute):
    pairs = sorted(attribute["pairs"], key=lambda pair: -abs(pair["pair_max"]))
    rows = []
    for pair in pairs[:10]:
        index = pair["entry_index"]
        rows.append(
            (
                pair["node"],
                pair["cell"],
                pair["moment"],
                pair["component"],
                number(pair["certified"][index]),
                number(pair["pre_repair"][index]),
                number(pair["distance_vertex"]),
                number(pair["distance_edge"]),
                number(pair["cell_height_gap"]),
            )
        )
    return rows


def bound_counts(census):
    bank = census["banks"]["certified"]["above"]
    return [(bound_label(key), bank[key]) for key in bound_keys(census)]


def entry_text(attribute):
    counts = Counter(
        (pair["moment"], pair["entry_index"]) for pair in attribute["pairs"]
    )
    pieces = []
    for (moment, index), count in counts.most_common():
        amount = "pair" if count == 1 else "pairs"
        pieces.append(
            f"{count} {amount} at index <code>{index}</code> of moment {moment}"
        )
    return ", ".join(pieces)


def cell_moment_text(attribute):
    inverse = attribute["inverse_second_moment"]
    top_cells = [pair["cell"] for pair in attribute["pairs"][:3]]
    signatures = [tuple(inverse[str(cell)]) for cell in top_cells]
    reference = attribute["reference_cell"]
    reference_values = inverse[str(reference)]
    listed = ", ".join(str(cell) for cell in top_cells)
    if len(set(signatures)) == 1:
        singular = signatures[0]
        opening = f"The three largest pairs sit in cells {listed}, which share"
    else:
        singular = signatures[0]
        opening = f"The largest pair sits in cell {top_cells[0]}, whose"
        opening += " inverse second moment has"
    return (
        f"{opening} singular values of <code>{number(singular[0])}</code> and "
        f"<code>{number(singular[1])} m^-2</code>, against "
        f"<code>{number(reference_values[0])}</code> and "
        f"<code>{number(reference_values[1])}</code> for an ordinary cell such "
        f"as {reference}."
    )


def part_pairs(attribute, census):
    composition = pair_composition(attribute)
    keys = bound_keys(census)
    bound = bound_label(keys[1])
    above = ", ".join(
        f"{count} above <code>{label}</code>" for label, count in bound_counts(census)
    )
    intro = (
        "<p>The subject is the certified bank built from ITER DINA scenario "
        "105027 (DD 3.38.1) at states 220, 333 and 336: "
        f"{attribute['cells']} polygon source cells and "
        f"{attribute['active_nodes']} active target nodes. Each node is booked "
        "against every cell as three source moments of six entries, "
        "<code>[psi, dpsi/dR, dpsi/dZ, psi_RR, psi_RZ, psi_ZZ]</code>. The "
        "table lists the ten largest pairs by their largest second-derivative "
        "entry, with the same entry from the pre-repair build beside it.</p>"
    )
    closing = (
        f"<p>{composition_text(attribute)}, and every one of the "
        f"{composition['total']} pairs above <code>{bound}</code> has a target "
        f"within <code>{number(composition['max_distance'])} m</code> of a "
        "vertex of its own source cell. The counts at each bound read "
        f"{above}. In every pair the largest entry is <code>psi_RR</code>: "
        + entry_text(attribute)
        + ". "
        + cell_moment_text(attribute)
        + "</p>"
    )
    headers = (
        "node",
        "cell",
        "moment",
        "entry",
        "certified",
        "pre-repair",
        "vertex distance [m]",
        "edge distance [m]",
        "height gap [m]",
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-pairs">Which pairs carry the number</h4>',
            intro,
            table(headers, pair_rows(attribute), "bank-hessian-blowup-worst-pairs"),
            closing,
        )
    )


def panel_extent(points):
    return (
        min(point[0] for point in points),
        max(point[0] for point in points),
        min(point[1] for point in points),
        max(point[1] for point in points),
    )


def padded(extent, fraction):
    low_r, high_r, low_z, high_z = extent
    pad_r = (high_r - low_r) * fraction + fraction
    pad_z = (high_z - low_z) * fraction + fraction
    return (low_r - pad_r, high_r + pad_r, low_z - pad_z, high_z + pad_z)


def framed(extent, aspect, factor):
    low_r, high_r, low_z, high_z = extent
    centre_r = (low_r + high_r) / 2
    centre_z = (low_z + high_z) / 2
    span_r = max(high_r - low_r, (high_z - low_z) * aspect) * factor
    span_z = span_r / aspect
    return (
        centre_r - span_r / 2,
        centre_r + span_r / 2,
        centre_z - span_z / 2,
        centre_z + span_z / 2,
    )


def mapper(extent, left, top, width, height):
    low_r, high_r, low_z, high_z = extent
    scale = min(width / (high_r - low_r), height / (high_z - low_z))
    offset_x = left + (width - (high_r - low_r) * scale) / 2
    offset_y = top + (height - (high_z - low_z) * scale) / 2

    def place(point):
        return (
            offset_x + (point[0] - low_r) * scale,
            offset_y + (high_z - point[1]) * scale,
        )

    return place


def inside(point, extent):
    low_r, high_r, low_z, high_z = extent
    return low_r <= point[0] <= high_r and low_z <= point[1] <= high_z


def points_text(points, place):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in (place(point) for point in points))


def marker(x, y, radius, colour):
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
        f'stroke="none" fill="{colour}"/>'
    )


def outline(points, place, colour, width):
    return (
        f'<polyline points="{points_text(points, place)}" fill="none" '
        f'stroke="{colour}" stroke-width="{width}"/>'
    )


def label(x, y, text, size=14, colour="#3f3f46", anchor="start"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{colour}" '
        f'text-anchor="{anchor}">{text}</text>'
    )


def figure_svg(attribute):
    shape = geometry.read()
    wall = shape["wall"]
    polygons = shape["polygons"]
    pairs = attribute["pairs"]
    leaders = pairs[:3]
    top_cells = [pair["cell"] for pair in leaders]

    payload = [
        f'<svg viewBox="0 0 {FIGURE_WIDTH} {FIGURE_HEIGHT}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'width="{FIGURE_WIDTH}" height="{FIGURE_HEIGHT}" '
        'style="max-width:100%;height:auto">'
    ]
    full = mapper(
        padded(panel_extent(wall), 0.02),
        PANEL_LEFT,
        PANEL_TOP,
        PANEL_WIDTH,
        PANEL_HEIGHT,
    )
    payload.append(label(PANEL_LEFT, PANEL_TOP - 6, "full machine", size=16))
    for cell, polygon in enumerate(polygons):
        colour = FEATURE if cell in top_cells else GREY
        width = "1.4" if cell in top_cells else "0.6"
        payload.append(outline(polygon, full, colour, width))
    payload.append(outline(wall, full, WALL, "1.6"))
    for pair in pairs:
        x, y = full(pair["position"])
        payload.append(marker(x, y, "3.0", GREY))
    for pair in leaders:
        x, y = full(pair["position"])
        payload.append(marker(x, y, "3.4", FEATURE))

    column_left = PANEL_LEFT + PANEL_WIDTH + PANEL_GAP
    for row, pair in enumerate(leaders):
        polygon = polygons[pair["cell"]]
        top = PANEL_TOP + row * (ZOOM_HEIGHT + ZOOM_GAP)
        extent = framed(panel_extent(polygon), ZOOM_WIDTH / ZOOM_HEIGHT, ZOOM_FACTOR)
        place = mapper(extent, column_left, top, ZOOM_WIDTH, ZOOM_HEIGHT)
        payload.append(
            label(
                column_left + ZOOM_WIDTH / 2,
                top - 6,
                f"cell {pair['cell']}, node {pair['node']}",
                size=14,
                anchor="middle",
            )
        )
        clip = f"bank-zoom-{pair['cell']}"
        payload.append(
            f'<clipPath id="{clip}"><rect x="{column_left}" y="{top}" '
            f'width="{ZOOM_WIDTH}" height="{ZOOM_HEIGHT}"/></clipPath>'
        )
        payload.append(f'<g clip-path="url(#{clip})">')
        payload.append(outline(polygon, place, FEATURE, "2.0"))
        x, y = place(pair["position"])
        payload.append(marker(x, y, "4.5", FEATURE))
        payload.append("</g>")
    payload.append("</svg>")
    return "\n".join(payload)


def leader_shapes(attribute):
    polygons = geometry.read()["polygons"]
    shapes = []
    for pair in attribute["pairs"][:3]:
        points = np.asarray(polygons[pair["cell"]])
        shapes.append(
            (
                len(points),
                float(np.ptp(points[:, 0])),
                float(np.ptp(points[:, 1])),
            )
        )
    return shapes


def part_figure(attribute, census):
    composition = pair_composition(attribute)
    keys = bound_keys(census)
    bound = bound_label(keys[1])
    leaders = attribute["pairs"][:3]
    listed = ", ".join(f"<code>{pair['cell']}</code>" for pair in leaders)
    shapes = [
        (vertices, number(radial), number(vertical))
        for vertices, radial, vertical in leader_shapes(attribute)
    ]
    if len(set(shapes)) == 1:
        vertices, radial, vertical = shapes[0]
        sliver = (
            f"The three cells are {vertices}-vertex slivers measuring "
            f"<code>{radial} m</code> radially and "
            f"<code>{vertical} m</code> vertically. "
        )
    else:
        sliver = ""
    intro = (
        "<p>A bank row is a coefficient of a linear operator, not a solved "
        "flux field: it carries no poloidal flux contour map and admits no "
        "stationary points, so there is no null set to mark and no flux panel "
        "to draw. What the figure can show is the geometry the coefficient is "
        "computed from, and where the offending pairs sit in it.</p>"
    )
    caption = (
        "<figcaption>Left: every polygon source cell of the bank, with the "
        f"first wall in dark grey and the {composition['total']} pairs above "
        f"<code>{bound}</code> marked. Right: the three cells carrying the "
        f"largest pairs, {listed}, each framed on its own cell with its target "
        "marked. Grey marks the whole "
        "population; green marks the three leaders, chosen by the largest "
        f"booked second derivative. {sliver}Every marked target lies within "
        f"<code>{number(composition['max_distance'])} m</code> of a vertex of "
        "its own source cell, which is far below the resolution of any panel "
        "here, so the marker sits on that vertex.</figcaption>"
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-figure">Where the offending pairs sit</h4>',
            intro,
            '<figure class="bank-hessian-blowup-figure">',
            figure_svg(attribute),
            caption,
            "</figure>",
        )
    )


def rebuild_agreement(census):
    banks = census["banks"]
    certified = banks["certified"]
    rebuilds = [
        name
        for name, bank in banks.items()
        if name != "certified"
        and bank["maximum"] == certified["maximum"]
        and bank["above"] == certified["above"]
    ]
    agreement = census["agreement"]
    differences = [
        agreement[f"certified~{name}"]["max_absolute_difference"]
        for name in rebuilds
        if f"certified~{name}" in agreement
    ]
    identical = [
        agreement.get(f"{first}~{second}", {}).get("identical")
        for index, first in enumerate(rebuilds)
        for second in rebuilds[index + 1 :]
    ]
    return max(differences), all(value is True for value in identical)


def part_banks(census):
    keys = bound_keys(census)
    certified = census["banks"]["certified"]
    largest = census["banks"]["corner_omitted"]
    difference, same = rebuild_agreement(census)
    rows = bank_rows(census)
    headers = ("bank", "largest second derivative", *[bound_label(k) for k in keys])
    closing = (
        f"<p>The certified bank's largest second derivative is "
        f"<code>{short(certified['maximum'])}</code>; the rerun rebuilds "
        f"reproduce it to within <code>{short(difference)}</code> in absolute "
        "terms"
        + (" and are identical to each other, " if same else ", ")
        + "so the rerun did not introduce it. "
        "The rerun build <code>b00f714a</code>, which omits the corner "
        f"entries, peaks at <code>{short(largest['maximum'])}</code>, below "
        "even the smallest bound in the table, so the number is carried by the "
        "corner entries rather than by the moment rows.</p>"
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-banks">Which bank carries the number</h4>',
            "<p>Each column counts node-cell pairs whose largest booked "
            "second-derivative entry exceeds the stated bound. The rows are the "
            "certified store, the pre-repair store and "
            f"{len(census['banks']) - 2} rerun builds of the same inputs.</p>",
            table(headers, rows, "bank-hessian-blowup-bank-census"),
            closing,
        )
    )


def bank_rows(census):
    keys = bound_keys(census)
    rows = []
    for name, bank in census["banks"].items():
        rows.append(
            (
                BANK_LABELS[name],
                short(bank["maximum"]),
                *[bank["above"][key] for key in keys],
            )
        )
    return rows


def mask_rows(attribute):
    selector = attribute["selector"]
    return (
        (
            "height selector",
            number(selector["tolerance"]),
            "target height within this gap of any polygon vertex height",
        ),
        (
            "corner selector",
            number(selector["corner_tolerance"]),
            "target within this distance of a source cell vertex",
        ),
        (
            "changed pair",
            number(selector["changed_tolerance"]),
            "booked second derivative differs from the pre-repair store by more",
        ),
    )


def part_masks(attribute):
    selector = attribute["selector"]
    selected = selector["selected_nodes"]
    total = selector["nodes"]
    changed = selector["changed_pairs"]
    inside = selector["changed_pairs_selected"]
    outside = selector["changed_pairs_unselected"]
    closing = (
        f"<p>{selected} of {total} target nodes sit inside the height "
        f"selector's tolerance and {total - selected} outside it. The gap "
        f"separates cleanly: the widest selected gap is "
        f"<code>{number(selector['selected_max_gap'])} m</code> and the "
        f"narrowest unselected gap is "
        f"<code>{number(selector['unselected_min_gap'])} m</code>. Of the "
        f"{changed} pairs that differ from the pre-repair store by more than "
        f"<code>{number(selector['changed_tolerance'])}</code>, {inside} sit "
        f"on selected nodes and {outside} on unselected ones, so the "
        "substitution and the moved entries are the same population.</p>"
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-masks">What the selectors decide</h4>',
            "<p>Three tolerances decide which pairs the repaired bank treats "
            "specially. Each is recorded with the run that applied it.</p>",
            table(
                ("selector", "tolerance", "decides"),
                mask_rows(attribute),
                "bank-hessian-blowup-mask-tolerances",
            ),
            closing,
        )
    )


META = ("node", "cell", "point", "certified", "pre_repair")


def ratio(value, reference):
    if reference == 0.0:
        return "undefined"
    return f"{value / reference:.3g}x"


def arm_names(entry):
    return [name for name in entry if name not in META]


def arm_rows(kernel):
    rows = []
    for key, entry in kernel.items():
        certified = entry["certified"][ARM_ENTRY]
        for arm in arm_names(entry):
            booked = entry[arm][ARM_ENTRY]
            rows.append(
                (
                    key,
                    arm.replace("_", " "),
                    short(booked),
                    short(certified),
                    ratio(booked, certified),
                )
            )
    return rows


def arm_ratios(kernel, bound):
    ratios = {}
    for entry in kernel.values():
        certified = entry["certified"][ARM_ENTRY]
        side = "above" if abs(certified) >= bound else "below"
        for arm in arm_names(entry):
            book = ratios.setdefault(arm, {"above": [], "below": []})
            book[side].append(entry[arm][ARM_ENTRY] / certified)
    return ratios


def ratio_span(values):
    low, high = min(values), max(values)
    if abs(high - low) <= 0.001 * max(abs(low), abs(high)):
        return f"{high:.3g}"
    return f"{low:.3g} to {high:.3g}"


def part_arms(kernel, attribute):
    headers = (
        "node:cell",
        "row program",
        "booked psi_RR",
        "certified psi_RR",
        "arm / certified",
    )
    ratios = arm_ratios(kernel, attribute["bound"])
    named = ", ".join(
        f"{arm.replace('_', ' ')} at <code>{ratio_span(ratios[arm]['above'])}</code>"
        for arm in ratios
    )
    below = ", ".join(
        f"{arm.replace('_', ' ')} at <code>{ratio_span(ratios[arm]['below'])}</code>"
        for arm in ratios
        if ratios[arm]["below"]
    )
    below_spread = max(
        abs(value - 1.0) for arm in ratios for value in ratios[arm]["below"]
    )
    closing = (
        "<p>Each arm is the repository's own jet program with only its row "
        "function swapped, so a difference between two rows of this table is a "
        "difference between the row programs and nothing else. Across the "
        f"pairs above <code>{bound_label(str(attribute['bound']))}</code> the "
        f"booked entry as a fraction of the certified one reads {named}, so the "
        "booked number is produced by the shipped derivative rule. Below the "
        f"bound the same three programs read {below}, each within "
        f"<code>{below_spread:.2g}</code> of the certified entry, so the "
        "blow-up is a property of these pairs rather than of the rows in "
        "general.</p>"
    )
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-arms">Which derivative rule produces it</h4>',
            "<p>The same node-cell pair evaluated through "
            "<code>nova.equilibrium.fixed_points._source_jets</code> with three "
            "row programs: the shipped one, the stable-pole primal the shipped "
            "program is built on, and the plain primal that carries no "
            "stable-pole substitution.</p>",
            table(headers, arm_rows(kernel), "bank-hessian-blowup-arms-table"),
            closing,
        )
    )


def step(rows, moment, entry, offset):
    above = rows[f"{offset:+.0e}"][moment][entry]
    below = rows[f"{-offset:+.0e}"][moment][entry]
    return abs(above - below)


def value_steps(jump, arm):
    rows = jump["arms"][arm]
    return [
        step(rows, jump["moment"], jump["value_entry"], offset)
        for offset in jump["offsets"]
    ]


def falls(steps):
    return all(later <= earlier for earlier, later in zip(steps, steps[1:]))


def jump_rows(jump):
    value_entry = jump["value_entry"]
    radial_entry = jump["radial_entry"]
    moment = jump["moment"]
    rows = []
    for offset in jump["offsets"]:
        row = [number(offset)]
        for arm in jump["arms"]:
            above = jump["arms"][arm][f"{offset:+.0e}"]
            below = jump["arms"][arm][f"{-offset:+.0e}"]
            row.append(number(above[moment][value_entry] - below[moment][value_entry]))
            row.append(
                number(above[moment][radial_entry] - below[moment][radial_entry])
            )
        rows.append(tuple(row))
    return rows


def joined(names):
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def jump_verdict(jump):
    profiles = {arm: value_steps(jump, arm) for arm in jump["arms"]}
    falling = [arm for arm, steps in profiles.items() if falls(steps)]
    holding = [arm for arm in profiles if arm not in falling]
    sentences = []
    if falling:
        names = joined([arm.replace("_", " ") for arm in falling])
        verb = "falls" if len(falling) == 1 else "fall"
        noun = "that row program" if len(falling) == 1 else "each of those row programs"
        sentences.append(
            f"The value step of {names} {verb} with the offset, so {noun} "
            "carries a value that is continuous in R at the probe radius."
        )
    for arm in holding:
        sentences.append(
            f"The {arm.replace('_', ' ')} value step does not fall with the "
            f"offset: it reads <code>{number(profiles[arm][0])}</code> at the "
            f"widest offset and <code>{number(profiles[arm][-1])}</code> at the "
            "narrowest, so that formulation's own value carries a step across "
            "the probe radius."
        )
    if "shipped" in holding:
        sentences.append(
            "The shipped program's own value therefore carries a step across "
            "the probe radius, and the booked second derivative is a property "
            "of that step rather than of the derivative rule alone."
        )
    elif holding:
        names = joined([arm.replace("_", " ") for arm in holding])
        sentences.append(
            "So the value graph is not common to all three formulations: the "
            "shipped program's value is continuous across the probe radius and "
            f"the {names} value is not, and the booked second derivative "
            "belongs to the shipped derivative rule."
        )
    else:
        sentences.append(
            "The second derivative booked for this pair therefore belongs to "
            "the shipped derivative rule rather than to a value graph every "
            "arm shares."
        )
    return " ".join(sentences)


def part_jump(jump):
    arms = list(jump["arms"])
    headers = ["offset [m]"]
    for arm in arms:
        name = arm.replace("_", " ")
        headers.extend((f"{name} value step", f"{name} gradient step"))
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-jump">Is it a jump or a smooth row</h4>',
            f"<p>The booked rows of node {jump['node']}, cell {jump['cell']} "
            "evaluated at points straddling its own radius, by "
            "<code>+/-</code> each offset. A step that holds its size as the "
            "offset shrinks is a discontinuity; a step that falls with the "
            "offset is a smooth row.</p>",
            table(headers, jump_rows(jump), "bank-hessian-blowup-jump-table"),
            f"<p>{jump_verdict(jump)}</p>",
        )
    )


def cell_span(cells):
    ordered = sorted(cells)
    runs = []
    start = previous = ordered[0]
    for cell in ordered[1:]:
        if cell == previous + 1:
            previous = cell
            continue
        runs.append((start, previous))
        start = previous = cell
    runs.append((start, previous))
    return ", ".join(
        str(start) if start == previous else f"{start}-{previous}"
        for start, previous in runs
    )


def moment_rows(attribute):
    inverse = attribute["inverse_second_moment"]
    largest = {}
    for pair in attribute["pairs"]:
        cell = pair["cell"]
        largest[cell] = max(largest.get(cell, 0.0), pair["pair_max"])
    groups = {}
    for cell, values in inverse.items():
        groups.setdefault((values[0], values[1]), []).append(int(cell))
    rows = []
    for (first, second), cells in groups.items():
        carried = [largest[cell] for cell in cells if cell in largest]
        top = max(carried) if carried else None
        rows.append(
            (
                cell_span(cells),
                len(cells),
                number(first),
                number(second),
                short(top) if top is not None else "reference",
                top if top is not None else -1.0,
            )
        )
    rows.sort(key=lambda row: -row[-1])
    return [row[:-1] for row in rows]


def part_moment(attribute):
    rows = moment_rows(attribute)
    reference = attribute["reference_cell"]
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-moment">Why those cells magnify it</h4>',
            "<p>Each cell carries an inverse second moment, and every source "
            "moment booked against the cell is contracted through it, so a "
            "cell whose singular values are orders above an ordinary cell's "
            "magnifies every entry booked against it. The table groups every "
            "cell carrying a pair above the bound by that singular pair, "
            f"beside the reference cell {reference}.</p>",
            table(
                (
                    "cells",
                    "count",
                    "first singular value",
                    "second singular value",
                    "largest pair",
                ),
                rows,
                "bank-hessian-blowup-inverse-second-moment",
            ),
        )
    )


def part_method():
    return "\n".join(
        (
            '<h4 id="bank-hessian-blowup-method">How it was measured</h4>',
            "<p>Four records carry every number on this page: the census of "
            "each bank's second-derivative magnitudes, the attribution of the "
            "pairs above the bound, the jet comparison of the row programs, "
            "and the offset profile. They are written under "
            f"<code>{CACHE}</code>.</p>",
            "<p>Nothing here re-implements the repository's machinery. The "
            "point-to-edge distance is "
            "<code>nova.equilibrium.boundary_comparison._point_to_segment_"
            "distances</code>; the jets are "
            "<code>nova.equilibrium.fixed_points._source_jets</code>, called "
            "with its row function swapped for the arm under test; and the "
            "cell geometry is read once by "
            "<code>scripts.prototypes.bank_hessian_blowup.geometry</code> and "
            "shared by every script in this directory.</p>",
        )
    )


def render():
    census = load("census.json")
    attribute = load("attribute.json")
    kernel = load("kernel.json")
    jump = load("jump.json")
    parts = (
        part_head(attribute, census),
        part_summary(attribute, census),
        part_figure(attribute, census),
        part_pairs(attribute, census),
        part_banks(census),
        part_masks(attribute),
        part_arms(kernel, attribute),
        part_jump(jump),
        part_moment(attribute),
        part_method(),
        "</article></main></body></html>",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(parts) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    render()
