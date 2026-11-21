"""Pin the numbers the imas-efit row-exchange record quotes.

``docs/evidence/fragments/inverse-equilibrium-baseline/efit-on-nova-rows.html``
states measured quantities, and nothing else in the repository would fail if one
of them were rewritten.  This module fails in that case.  It reads the receipt
committed beside the figures
(``docs/figures/inverse-equilibrium-baseline/efit-on-nova-rows/receipt.json``,
written from the run's own records by ``write_receipt.py``), recomputes every
derived figure — the deltas against the generating state, the band verdicts, the
amplitude trade, the control ratios — and requires the record to quote each one.

It follows the repository's receipt-reconciliation form
(``tests/test_outboard_census_receipt_consistency.py``), takes the panel
comparison from the module that owns that rule rather than restating it
(``tests/test_figure_panel_pin._vector_content_digest``, because a raw file
digest of an SVG moves with its wall-clock ``dc:date`` and generated ids), and
leaves worktree-path refusal to ``tests/test_artifacts_carry_no_worktree_paths.py``
and the commit hook, which own it.  The structural claims it does make are that
the record's ids are unique, that its block markup balances, and that the basis
and edge constraint it names are labelled as configuration reads where the
solver log announces only the live counts.

Run it as a node check::

    PYTHONPATH=. JAX_PLATFORMS=cpu python -m pytest -p no:cacheprovider -q \
      tests/test_efit_on_nova_rows_receipt.py
"""

from __future__ import annotations

import json
import pathlib
import re
import xml.etree.ElementTree as ElementTree
from html.parser import HTMLParser

import numpy as np
import pytest

from tests.test_figure_panel_pin import _vector_content_digest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = (
    ROOT / "docs" / "figures" / "inverse-equilibrium-baseline" / "efit-on-nova-rows"
)
FRAGMENT = (
    ROOT
    / "docs"
    / "evidence"
    / "fragments"
    / "inverse-equilibrium-baseline"
    / "efit-on-nova-rows.html"
)
FRAGMENT_ID = "efit-on-nova-rows"
PANELS = {
    "nominal": FIGURES / "efit-generating-overlay-10.svg",
    "control": FIGURES / "efit-generating-overlay-11.svg",
}


def normalise(text: str) -> str:
    """Fold the typographic characters the record uses to their ASCII forms."""
    return (
        text.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )


@pytest.fixture(scope="module")
def receipt():
    return json.loads((FIGURES / "receipt.json").read_text())


@pytest.fixture(scope="module")
def record():
    return normalise(FRAGMENT.read_text())


def ship(value: float, digits: int) -> str:
    """Return one number in the exact spelling the record uses."""
    return f"{value:.{digits}f}"


def exponent(value: float) -> str:
    """Return one small number in the record's ``8.7e-7`` spelling."""
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e")


def derived(receipt):
    """Return every figure the record quotes, computed from raw states."""
    generating = receipt["generating"]
    efit = receipt["efit"]
    fit = receipt["fit"]
    target = receipt["control_scale"] * generating["plasma_current"]
    span = generating["plasma_span"]
    return dict(
        generating=generating,
        efit=efit,
        fit=fit,
        target=target,
        axis_offset=float(
            np.linalg.norm(
                np.asarray(efit["10"]["axis"]) - np.asarray(generating["axis"])
            )
        ),
        axis_flux_fraction=(efit["10"]["flux_axis"] - generating["flux_axis"]) / span,
        boundary_flux_fraction=(
            efit["10"]["flux_boundary"] - generating["flux_boundary"]
        )
        / span,
        q95_gap=abs(efit["10"]["q95"]) - abs(generating["q95"]),
        nominal_current_fraction=(
            efit["10"]["plasma_current"] - generating["plasma_current"]
        )
        / abs(generating["plasma_current"]),
        control_current_fraction=(efit["11"]["plasma_current"] - target) / abs(target),
        control_fit_current_fraction=(fit["11"]["state"]["plasma_current"] - target)
        / abs(target),
        span_ratio=efit["11"]["plasma_span"] / efit["10"]["plasma_span"],
    )


def test_the_record_quotes_every_state_it_tabulated(record, receipt):
    """Each state appears at the precision the record prints it."""
    generating = receipt["generating"]
    efit = receipt["efit"]

    expected = [
        # the generating state, as a block and as table cells
        f"plasma_current={generating['plasma_current']!r}",
        f"flux_axis={generating['flux_axis']!r}",
        f"flux_boundary={generating['flux_boundary']!r}",
        f"flux_span={generating['plasma_span']!r}",
        f"q95={generating['q95']!r}",
        f"edge_safety_factor={generating['edge_safety_factor']!r}",
        # table cells, four, three and two decimals
        ship(abs(generating["plasma_current"]) / 1e6, 4),
        ship(abs(efit["11"]["plasma_current"]) / 1e6, 4),
        ship(generating["flux_axis"], 3),
        ship(efit["10"]["flux_axis"], 3),
        ship(efit["11"]["flux_axis"], 3),
        ship(generating["flux_boundary"], 3),
        ship(efit["10"]["flux_boundary"], 3),
        ship(efit["11"]["flux_boundary"], 3),
        ship(generating["plasma_span"], 3),
        ship(efit["10"]["plasma_span"], 3),
        ship(efit["11"]["plasma_span"], 3),
        # axes and nulls
        "({}, {})".format(
            ship(generating["axis"][0], 5), ship(generating["axis"][1], 5)
        ),
        "({}, {})".format(
            ship(efit["10"]["axis"][0], 5), ship(efit["10"]["axis"][1], 5)
        ),
        "({}, {})".format(
            ship(efit["11"]["axis"][0], 5), ship(efit["11"]["axis"][1], 5)
        ),
        # safety factor with the stored sign it carries
        "{} ({})".format(
            ship(abs(generating["q95"]), 3),
            "-1" if generating["q95_sign"] < 0 else "+1",
        ),
        "{} ({})".format(
            ship(abs(efit["10"]["q95"]), 3),
            "-1" if efit["10"]["q95_sign"] < 0 else "+1",
        ),
        "{} ({})".format(
            ship(abs(efit["11"]["q95"]), 3),
            "-1" if efit["11"]["q95_sign"] < 0 else "+1",
        ),
        ship(efit["10"]["q_95_dd"], 3),
    ]
    missing = [value for value in expected if value not in record]
    assert not missing, f"the record no longer quotes {missing}"


def test_the_record_quotes_the_bands_it_measured(record, receipt):
    """Every stated delta is recomputed from the receipt's raw states."""
    figures = derived(receipt)

    expected = [
        exponent(abs(figures["nominal_current_fraction"])),
        ship(abs(figures["axis_offset"]), 4),
        ship(abs(figures["axis_flux_fraction"]) * 100.0, 2),
        ship(abs(figures["boundary_flux_fraction"]) * 100.0, 2),
        ship(abs(figures["q95_gap"]), 3),
    ]
    missing = [value for value in expected if value not in record]
    assert not missing, f"the record no longer quotes {missing}"

    # the q95 band is judged on the record's own reading: one calculator, one
    # surface definition, the stored values reported beside it
    single = receipt["q95_single_calculator"]
    assert "All five section-8 bands are met" in record
    assert "The q95 band is not met" not in record
    assert "Four of the five" not in record
    assert "0.42 %" not in record
    # the definitional share is 0.083, never the 0.065 residual
    for sentence in re.split(r"(?<=\.)\s", record):
        if "definitional share" in sentence:
            assert "0.083" in sentence, f"share attached to another number: {sentence}"
    # every verdict is recomputed here against the receipt's own thresholds
    bands = receipt["bands"]
    assert abs(single["gap"]["psi95"]) < bands["q95"]
    assert abs(single["gap"]["rho95"]) < bands["q95"]
    assert abs(figures["q95_gap"]) > bands["q95"]
    assert abs(figures["axis_offset"]) < bands["axis_metres"]
    assert abs(figures["nominal_current_fraction"]) < bands["current_fraction"]
    assert abs(figures["axis_flux_fraction"]) < bands["flux_fraction"]
    assert abs(figures["boundary_flux_fraction"]) < bands["flux_fraction"]


def test_the_record_quotes_the_amplitude_trade_it_attributed(record, receipt):
    """The split is re-projected from the record's own p' and FF' arrays."""
    compact = record.replace(" ", "")
    generating = receipt["generating"]
    recovered = receipt["efit"]["10"]["profile"]
    shape = np.asarray(recovered["shape"], dtype=float)
    for name in ("pressure", "diamagnetic"):
        values = np.asarray(recovered[f"{name}_values"], dtype=float)
        projection = float(values @ shape / (shape @ shape))
        assert projection == pytest.approx(recovered[name], rel=1e-12)
    pressure = abs(recovered["pressure"]) / generating["pressure_amplitude"] - 1.0
    diamagnetic = (
        1.0 - abs(recovered["diamagnetic"]) / generating["diamagnetic_amplitude"]
    )
    expected = [
        ship(pressure * 100.0, 2),
        ship(diamagnetic * 100.0, 2),
        ship(abs(recovered["pressure"]), 1),
        ship(abs(recovered["diamagnetic"]), 5),
        ship(generating["pressure_amplitude"], 1),
        ship(generating["diamagnetic_amplitude"], 5),
    ]
    missing = [value for value in expected if value not in compact]
    assert not missing, f"the record no longer quotes {missing}"


def test_the_record_quotes_the_nova_fit_it_measured(record, receipt):
    """Both arms, their singular values and residuals are the measured ones."""
    fit = receipt["fit"]
    expected = [
        f"coefficients={fit['10']['coefficients']}".replace("'", ""),
        f"singular_values={fit['10']['singular_values']}".replace("'", ""),
        f"residual_sup={fit['10']['residual_sup']!r}",
        f"terminal_residual={fit['10']['terminal_residual']!r}",
        f"axis={fit['10']['state']['axis']}".replace("'", ""),
        f"plasma_current={fit['10']['state']['plasma_current']!r}",
        ship(abs(fit["11"]["coefficients"][0]), 3),
        ship(abs(fit["11"]["coefficients"][1]), 3),
        exponent(fit["11"]["residual_sup"]),
    ]
    missing = [value for value in expected if value not in record]
    assert not missing, f"the record no longer quotes {missing}"
    assert fit["11"]["converged"] is False
    assert "converged=false" in record
    assert "1.6e-4" not in record
    assert "14.995" not in record


def test_the_record_quotes_the_control_ratios(record, receipt):
    """The control's currents and span ratio are recomputed, not transcribed."""
    figures = derived(receipt)
    expected = [
        exponent(abs(figures["control_current_fraction"])),
        ship(abs(figures["control_fit_current_fraction"]) * 100.0, 2),
        ship(figures["span_ratio"], 4),
        ship(abs(receipt["fit"]["11"]["state"]["q95"]), 3),
    ]
    missing = [value for value in expected if value not in record]
    assert not missing, f"the record no longer quotes {missing}"


def test_the_solver_lines_are_quoted_and_the_basis_is_labelled(record, receipt):
    """Live counts come from the log; family and edge read as configuration."""
    assert receipt["solver"]["dynamic_ndeg"] in record
    assert receipt["solver"]["design_rank"] in record
    assert "configuration read" in record


def test_the_basis_log_line_is_quoted_verbatim(record, receipt):
    """The family is pinned by the solver's own LLS_COND label, not the file."""
    lines = receipt["basis_log"]["lls_cond"]
    assert len(lines) == 8, f"{len(lines)} LLS_COND lines recorded"
    assert all("basis=bernstein" in line for line in lines)
    first, last = lines[0], lines[-1]
    # the design grows from 14 to 15 columns and reaches full column rank
    assert "iter=2" in first and "n=14 rank=14" in first
    assert "iter=9" in last and "n=15 rank=15" in last
    assert "LLS_COND: " + first in record
    assert "<code>rank=15</code> of <code>n=15</code>" in record
    assert "m=87" in first and "path=bvls" in first


def test_the_q95_rows_are_pinned_and_the_band_read_on_one_calculator(record, receipt):
    """The four calculator values, their gaps, and the share arithmetic."""
    single = receipt["q95_single_calculator"]
    nova, efit = single["states"]
    quarters = [
        ship(nova["q_at_psi95"], 4),
        ship(efit["q_at_psi95"], 4),
        ship(nova["q_at_rho95"], 4),
        ship(efit["q_at_rho95"], 4),
        ship(nova["axis_flux"], 4),
        ship(nova["boundary_flux"], 4),
        ship(efit["axis_flux"], 4),
        ship(efit["boundary_flux"], 4),
        ship(single["gap"]["psi95"], 3),
        ship(single["gap"]["rho95"], 3),
    ]
    missing = [value for value in quarters if value not in record]
    assert not missing, f"the record no longer quotes {missing}"

    # the gaps are the differences of the quoted rows, not a separate number
    assert single["gap"]["psi95"] == pytest.approx(
        nova["q_at_psi95"] - efit["q_at_psi95"], abs=1e-12
    )
    assert single["gap"]["rho95"] == pytest.approx(
        nova["q_at_rho95"] - efit["q_at_rho95"], abs=1e-12
    )
    bands = receipt["bands"]
    assert single["gap"]["psi95"] < bands["q95"]

    # the definition difference is the stored values' difference, and the
    # definitional share is what the one-calculator reading does not explain
    stored = single["stored"]
    cross = abs(abs(stored["nova_q95"]) - abs(stored["efit_q95"]))
    share = cross - single["gap"]["psi95"]
    assert ship(cross, 3) == "0.148" and "0.148" in record
    assert ship(share, 3) == "0.083" and "0.083" in record
    assert ship(abs(stored["efit_q_95_dd"]), 3) in record
    assert ship(abs(stored["nova_q95"]), 3) in record


def test_the_edge_constraint_is_shown_by_the_solver_not_the_config(record, receipt):
    """Live counts move with the switch; the dropped gradients vanish."""
    on = receipt["edge_control"]["edge_on"]
    off = receipt["edge_control"]["edge_off"]
    # one live coefficient at a configured degree of two, then two
    counted = ((on["dynamic_ndeg"], "kppcur= 1"), (off["dynamic_ndeg"], "kppcur= 2"))
    for line, count in counted:
        assert "max_pp= 2" in line and count in line, line
        assert "regime= 2" in line, line
    assert "rank_pp=1 n_pp=1 rank_ff=1 n_ff=1" in on["design_rank"]
    assert "rank_pp=2 n_pp=2 rank_ff=2 n_ff=2" in off["design_rank"]
    # both runs quote their own lines in the record
    for line in (
        on["dynamic_ndeg"],
        off["dynamic_ndeg"],
        on["design_rank"],
        off["design_rank"],
    ):
        assert line in record, f"the record no longer quotes {line}"

    # the fitted gradients terminate at exactly zero under the edge drop
    assert on["profiles"]["nodes"] == off["profiles"]["nodes"] == 65
    for name, digits in (("dpressure_dpsi", 4), ("f_df_dpsi", 7)):
        assert on["profiles"][name][-1] == 0.0
        assert off["profiles"][name][-1] != 0.0
        assert ship(abs(on["profiles"][name][0]), digits) in record
        assert ship(abs(off["profiles"][name][-1]), digits) in record


def test_the_imas_efit_wall_and_hardware_are_stated_and_pinned(record, receipt):
    """Clause 1 asks for the run's wall and hardware; the cost section reads it."""
    efit_run = receipt["efit_run"]
    job = efit_run["job"]
    runs = efit_run["runs"]
    for value in (job["partition"], job["node"], str(job["cores"]), job["id"]):
        assert value in record, f"the record no longer names {value}"
    assert job["nodes"] == 1
    assert job["exit"] == "0:0"
    # the job wall is a whole number of seconds and bounds both pulse walls
    assert isinstance(job["wall_seconds"], int) and job["wall_seconds"] > 0
    for name in ("nominal", "scaled"):
        assert runs[name]["seconds"] > 0
        assert runs[name]["seconds"] < job["wall_seconds"]
        assert ship(runs[name]["seconds"], 2) in record, (
            f"the record no longer quotes the {name} wall"
        )
    assert runs["nominal"]["pulse"] == 10 and runs["scaled"]["pulse"] == 11
    # the stated figure is a duration, not a bare coincidence with another number
    assert f"<code>{job['wall_seconds']} s</code>" in record
    assert "sacct" in record


def test_the_record_states_the_relations_it_names(record):
    """Each wording fix states a measured relation, not a paraphrase."""
    # the current is a difference from the generating value, not a ratio to it
    assert (
        "differs from the generating value by a fraction <code>8.7e-7</code>" in record
    )
    assert "is <code>8.7e-7</code> of the generating value" not in record
    # the bands are read after the one additive gauge constant
    assert "read after the one additive constant" in record
    assert "<code>+0.3 Wb</code>" in record
    # the profile sign follows the stated flux relation and the actual gradient
    assert "<code>nova = +psi_DDv4</code>" in record
    assert "<code>dp/d\u03c8</code>" in record and "<code>F dF/d\u03c8</code>" in record
    assert "negated total flux" in record
    assert "whose sign is opposite nova's" not in record
    # the provenance cites the rerun that produced the recorded fit
    assert "<code>1321531</code>" in record and "efit-fit2" in record
    assert "StencilMesh" in record


class _IdentifierCounter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.identifiers = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "id":
                self.identifiers.append(value)


def test_every_element_identifier_is_unique():
    """A fragment that reuses an id sends an anchor to the wrong element."""
    counter = _IdentifierCounter()
    counter.feed(FRAGMENT.read_text())
    duplicated = sorted(
        {
            identifier
            for identifier in counter.identifiers
            if counter.identifiers.count(identifier) > 1
        }
    )
    assert not duplicated, f"duplicated ids {duplicated}"


def test_both_panels_parse_and_name_both_states(receipt):
    """Both committed panels are well-formed, labelled and unchanged."""
    for label, path in PANELS.items():
        assert path.exists(), f"missing {label} panel {path.name}"
        assert path.stat().st_size > 20_000, f"{path.name} is not a rendered panel"
        root = ElementTree.parse(path).getroot()
        assert root.tag.endswith("svg"), f"{path.name} is not SVG"
        text = normalise(path.read_text())
        for marker in (
            "generating state (nova forward solve)",
            "imas-efit reconstruction",
        ):
            assert marker in text, f"{path.name} does not label {marker}"
        assert path.name in normalise(FRAGMENT.read_text()), (
            f"the record does not cite {path.name}"
        )
        assert receipt["panels"][path.name] == _vector_content_digest(
            path.read_bytes()
        ), f"{path.name} no longer draws what the receipt recorded"


def test_the_record_cites_the_check_that_guards_it():
    """The record names the receipt and the check, so a reader can rerun them."""
    text = normalise(FRAGMENT.read_text())
    assert "receipt.json" in text
    assert "tests/test_efit_on_nova_rows_receipt.py" in text


def test_every_paragraph_opens_and_closes_in_order():
    """A close with no open, or an open inside an open, is silent on the page."""
    text = FRAGMENT.read_text()
    depth = 0
    for event in re.finditer(r"</?p\b[^>]*>", text):
        if event.group().startswith("</"):
            assert depth == 1, f"paragraph closed with none open at {event.start()}"
            depth = 0
        else:
            assert depth == 0, f"nested paragraph at offset {event.start()}"
            depth = 1
    assert depth == 0, "an unclosed paragraph"


def test_every_container_closes():
    """Sections, paragraphs, tables, figures and pre blocks balance."""
    text = FRAGMENT.read_text()
    for tag in ("section", "p", "table", "figure", "pre", "tbody", "thead", "tr"):
        opened = len(re.findall(rf"<{tag}\b", text))
        closed = len(re.findall(rf"</{tag}>", text))
        assert opened == closed, f"{tag}: {closed} closed, {opened} opened"
