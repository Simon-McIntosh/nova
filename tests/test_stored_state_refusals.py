"""Assert the stored-state refusals from the arms' own receipts.

The arms are produced by ``scripts/prototypes/stored_state_refusals/
run_recorded_cases.py`` on a GPU lane, which drives the earlier comparison
record's own case reader, fixture machine, receipt mesh, stored seed, fitted
conductor currents and policy.  This module holds those receipts to the values
the record reports, each written here rather than read back from a value the
runner also wrote, and needs no GPU.

``STORED_STATE_REFUSALS_RECEIPTS`` names the directory holding the arm
receipts; unset, it is the durable cache path the runner's output is copied to.
The measured refusals are asserted under the comparison's continuation policy,
which is the arm the record shows reaching them; the default-policy arms are
asserted only to be unqualified, because under today's fixtures they stop
short.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/prototypes/stored_state_refusals/run_recorded_cases.py"
FIGURES = (
    ROOT / "docs/figures/forward-solve-near-stored-states/"
    "stored-state-refusals-on-the-recorded-mesh"
)

RECORD_PATH = (
    "docs/evidence/fragments/inverse-equilibrium-baseline/efit-comparison.html"
)
RECORDED_RESIDUAL = 0.030364200526436996
RECORDED_REASON = 12
RESIDUAL_BAND = 3.0
MESH_NODES = {"tcv": 355, "iter": 341}
COMPARISON_MODULE = "tests.test_magnetics_fit_efit_comparison"
ARMS = (
    "tcv-stored-default",
    "tcv-stored-tightened",
    "iter-stored-default",
    "iter-stored-tightened",
)


def _receipts():
    directory = os.environ.get("STORED_STATE_REFUSALS_RECEIPTS")
    if directory is None:
        directory = "~/.cache/nova-stored-state-refusals/recorded-cases"
    path = Path(directory).expanduser()
    if not path.is_dir():
        raise FileNotFoundError("no arm receipts at %s" % path)
    return path


def _receipt(label):
    path = _receipts() / ("%s.json" % label)
    if not path.is_file():
        raise FileNotFoundError("arm %s left no receipt at %s" % (label, path))
    return json.loads(path.read_text())


def _svg(label):
    return FIGURES / ("%s.svg" % label)


def test_every_arm_is_unqualified():
    """No arm is qualified or converged: the refusal is the finding."""
    for label in ARMS:
        receipt = _receipt(label)
        assert receipt["qualified"] is False, label
        assert receipt["converged"] is False, label
        assert receipt["termination_reason"] in (3, 8, 12), label


def test_tcv_stops_at_the_recorded_residual():
    """The TCV continuation arm stops where the record says it stopped."""
    receipt = _receipt("tcv-stored-tightened")
    assert receipt["case"] == "tcv"
    assert receipt["termination_reason"] == RECORDED_REASON
    residual = receipt["residual"]
    assert residual is not None and math.isfinite(residual) and residual > 0.0, (
        "TCV stopped with no finite residual (%s)" % receipt["residual_state"]
    )
    factor = residual / RECORDED_RESIDUAL
    assert 1.0 / RESIDUAL_BAND <= factor <= RESIDUAL_BAND, (
        "TCV residual %s is %.3fx the recorded %s"
        % (residual, factor, RECORDED_RESIDUAL)
    )


def test_iter_admits_no_magnetic_axis():
    """The ITER continuation arm refuses with no magnetic axis admitted."""
    receipt = _receipt("iter-stored-tightened")
    assert receipt["case"] == "iter"
    assert receipt["termination_reason"] == 3
    assert receipt["residual"] is None
    axis = receipt["axis"]
    assert axis["admitted"] is False
    assert axis["error"] == "NoQualifiedAxisError"


def test_arms_are_the_comparison_records_own_fixture():
    """The arms are the comparison's reader, mesh, route and seed, not restated.

    The runner's import is read from the committed source rather than from the
    receipt it writes, so a runner that began restating the fixtures would fail
    this even if it went on labelling its output a comparison.
    """
    source = RUNNER.read_text()
    assert "from %s import (" % COMPARISON_MODULE in source
    for name in ("_attempt", "_problem", "_read_case", "_residual_fields"):
        assert name in source, name
    coefficients = {}
    for case in ("tcv", "iter"):
        receipt = _receipt("%s-stored-default" % case)
        assert receipt["seed_policy"] == "ExplicitSolveSeed"
        assert receipt["route"] == "newton_krylov"
        assert receipt["current_present"] is True
        mesh = receipt["mesh"]
        assert mesh["kind"] == "StencilMesh"
        assert mesh["nodes"] == MESH_NODES[case], mesh
        assert mesh["machine_nodes"] == MESH_NODES[case], mesh
        assert mesh["stencil_rows"] > 0
        coefficients[case] = receipt["coefficients"]
        assert len(coefficients[case]) >= 3
    assert coefficients["tcv"] != coefficients["iter"]


def test_the_record_cited_is_the_comparison_record():
    """The evidence fragment names the record these values come from."""
    fragment = (
        ROOT / "docs/evidence/fragments/forward-solve-near-stored-states/"
        "stored-state-refusals-on-the-recorded-mesh.html"
    )
    assert RECORD_PATH in fragment.read_text()


def test_each_committed_panel_draws_both_series():
    """Every arm's committed SVG carries both series, and one SVG per figure.

    The panel a receipt names lives in the run directory, which is archived on
    release, so the check reads the published figure instead.
    """
    for label in ARMS:
        svg = _svg(label)
        assert svg.is_file(), svg
        assert not (svg.parent / ("%s-preview.png" % svg.stem)).exists(), svg
        for series, colour in (("imas-efit", "#929292"), ("nova", "#3366cc")):
            count = _paths_with_colour(svg, colour)
            assert count > 0, "%s draws no %s path" % (svg.name, series)


def _paths_with_colour(svg, colour: str) -> int:
    root = ElementTree.parse(svg).getroot()
    count = 0
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "path":
            continue
        if not element.get("d"):
            continue
        if "stroke: %s" % colour in (element.get("style") or ""):
            count += 1
    return count
