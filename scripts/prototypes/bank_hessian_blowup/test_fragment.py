"""Checks that the bank page states what its measurement records say.

Each claim on the page is recomputed here from the records the measurement
jobs wrote, so a number typed into the renderer and a record rerun that moves
the result are both failures. The renderer is also checked for numbers it
could only have typed, and every script in the directory for a distance or a
jet it could only have re-implemented.

The records live under the measurement cache; where that cache is absent the
page cannot be rendered and these checks are skipped.
"""

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
CACHE = Path("/home/ITER/mcintos/.cache/nova-bank-hessian-blowup")
PAGE = (
    ROOT
    / "docs/evidence/fragments/sub-millisecond-forward-path/bank-hessian-blowup.html"
)
RECORDS = ("census.json", "attribute.json", "kernel.json", "jump.json")
SCRIPTS = (
    "attribute",
    "diagnose",
    "fragment",
    "geometry",
    "jump",
    "kernel",
)

needs_records = pytest.mark.skipif(
    not all((CACHE / name).exists() for name in RECORDS) or not PAGE.exists(),
    reason="the measurement records and the rendered page are not present",
)


def record(name):
    return json.loads((CACHE / name).read_text())


def page():
    return PAGE.read_text()


def number(value):
    mantissa, exponent = f"{value:.4e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def source(name):
    return (DIRECTORY / f"{name}.py").read_text()


@needs_records
def test_page_carries_a_figure_and_a_table_per_claim():
    text = page()
    assert "<svg" in text
    assert text.count("<table") >= 5
    assert "bank-hessian-blowup-figure" in text


@needs_records
def test_every_table_row_has_one_cell_per_header():
    for table in re.findall(r"<table.*?</table>", page(), re.S):
        headers = len(re.findall(r"<th>", table.split("</thead>")[0]))
        for row in re.findall(r"<tr>(.*?)</tr>", table.split("<tbody>")[1], re.S):
            assert len(re.findall(r"<td>", row)) == headers


@needs_records
def test_page_carries_no_unrendered_placeholder():
    text = page()
    assert "PLACEHOLDER" not in text
    assert "None" not in text.split("<main")[1]


@needs_records
def test_the_headline_is_the_largest_booked_second_derivative():
    census = record("census.json")
    attribute = record("attribute.json")
    largest = census["banks"]["certified"]["maximum"]
    assert number(largest) in page()
    assert attribute["pairs"][0]["pair_max"] == largest


@needs_records
def test_the_pair_composition_is_read_from_the_record():
    attribute = record("attribute.json")
    counts = Counter((pair["moment"], pair["component"]) for pair in attribute["pairs"])
    majority = counts.most_common(1)[0]
    assert majority == ((1, "psi_RR"), 37)
    assert counts[(2, "psi_RR")] == 1
    text = page()
    assert f"{majority[1]} are moment-1 <code>psi_RR</code> entries" in text
    assert "one is a moment-2 <code>psi_RR</code> entry" in text


@needs_records
def test_every_pair_lies_within_the_recorded_vertex_distance():
    attribute = record("attribute.json")
    distances = [pair["distance_vertex"] for pair in attribute["pairs"]]
    assert number(max(distances)) in page()
    assert max(distances) < 1e-12


@needs_records
def test_the_leading_cell_second_moment_is_read_from_the_record():
    attribute = record("attribute.json")
    cell = str(attribute["pairs"][0]["cell"])
    values = attribute["inverse_second_moment"][cell]
    text = page()
    assert number(values[0]) in text
    assert number(values[1]) in text


@needs_records
def offset_steps(jump, arm, entry):
    moment = jump["moment"]
    steps = []
    for offset in jump["offsets"]:
        above = jump["arms"][arm][f"{offset:+.0e}"][moment][entry]
        below = jump["arms"][arm][f"{-offset:+.0e}"][moment][entry]
        steps.append(abs(above - below))
    return steps


@needs_records
def test_the_offset_steps_fall_with_the_offset():
    jump = record("jump.json")
    for arm in jump["arms"]:
        steps = offset_steps(jump, arm, jump["value_entry"])
        assert steps == sorted(steps, reverse=True)
        assert steps[0] > 1000 * steps[-1]


@needs_records
def test_the_page_names_the_arm_whose_gradient_steps():
    jump = record("jump.json")
    graded = {
        arm: offset_steps(jump, arm, jump["radial_entry"]) for arm in jump["arms"]
    }
    stepped = [
        arm for arm, steps in graded.items() if steps != sorted(steps, reverse=True)
    ]
    assert stepped == ["stable_poles"]
    poles = graded["stable_poles"]
    assert poles[-1] > 100 * poles[-2]
    assert poles[-1] > poles[0]
    narrowest = jump["offsets"][-1]
    row = jump["arms"]["stable_poles"]
    values = (
        row[f"{narrowest:+.0e}"][jump["moment"]][jump["radial_entry"]],
        row[f"{-narrowest:+.0e}"][jump["moment"]][jump["radial_entry"]],
    )
    assert values[0] * values[1] < 0
    for arm in ("shipped", "plain"):
        assert graded[arm] == sorted(graded[arm], reverse=True)
        assert graded[arm][0] > graded[arm][-1]
    text = page()
    assert "stable poles gradient step does not fall" in text
    assert "stable poles derivative rule" in text
    assert number(graded["stable_poles"][-1]) in text


def test_the_renderer_types_no_scientific_literals():
    text = source("fragment")
    tree = ast.parse(text)
    typed = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and re.search(r"\d[eE][-+]?\d", ast.get_source_segment(text, node) or "")
    ]
    assert typed == []


def test_the_renderer_reads_geometry_and_never_measures_it():
    text = source("fragment")
    assert "geometry.read()" in text
    assert "np.linalg.norm" not in text


def test_the_shared_owners_are_imported_not_reimplemented():
    assert "_point_to_segment_distances" in source("geometry")
    assert "_source_jets" in source("kernel")
    assert "_source_jets" in source("jump")
    for name in SCRIPTS:
        text = source(name)
        assert "def point_segment_distance" not in text
        if name != "geometry":
            assert "def segment_distance" not in text
            assert "def vertex_distance" not in text
        if name not in ("kernel", "jump"):
            assert "def _source_jets" not in text
            assert "def _packed_analytic_moments_primal" not in text
