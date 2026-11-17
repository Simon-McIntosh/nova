"""Pin the numbers the imas-efit row-exchange record quotes.

``docs/evidence/fragments/inverse-equilibrium-baseline/efit-on-nova-rows.html``
states measured quantities, and nothing else in the repository would fail if one
of them were rewritten.  This module fails in that case.  It reads the receipt
this directory carries (written from the run's own records by
``write_receipt.py``), recomputes every derived figure — the deltas against the
generating state, the band verdicts, the amplitude trade, the control ratios —
and requires the record to quote each one.

It also holds the two structural claims the record makes about itself: that its
element ids are unique, and that the basis and edge constraint it names are
labelled as configuration reads where the solver log announces only the live
counts.

Run it as a node check::

    PYTHONPATH=. JAX_PLATFORMS=cpu python -m pytest -p no:cacheprovider -q \
      docs/figures/inverse-equilibrium-baseline/efit-on-nova-rows/test_fragment_receipt.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import xml.etree.ElementTree as ElementTree
from html.parser import HTMLParser

import pytest

HERE = pathlib.Path(__file__).resolve().parent
FRAGMENT = (
    HERE.parents[2]
    / "evidence"
    / "fragments"
    / "inverse-equilibrium-baseline"
    / "efit-on-nova-rows.html"
)
FRAGMENT_ID = "efit-on-nova-rows"
PANELS = {
    "nominal": HERE / "efit-generating-overlay-10.svg",
    "control": HERE / "efit-generating-overlay-11.svg",
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
    return json.loads((HERE / "receipt.json").read_text())


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
    """Return every figure the record quotes, computed from the receipt."""
    generating = receipt["generating"]
    efit = receipt["efit"]
    fit = receipt["fit"]
    target = receipt["control_scale"] * generating["plasma_current"]
    return dict(
        generating=generating,
        efit=efit,
        fit=fit,
        target=target,
        nominal_current_fraction=(
            efit["efit-10"]["plasma_current"] - generating["plasma_current"]
        )
        / abs(generating["plasma_current"]),
        control_current_fraction=(efit["efit-11"]["plasma_current"] - target)
        / abs(target),
        control_fit_current_fraction=(fit["11"]["state"]["plasma_current"] - target)
        / abs(target),
        span_ratio=efit["efit-11"]["flux_span"] / efit["efit-10"]["flux_span"],
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
        f"flux_span={generating['flux_span']!r}",
        f"q95={generating['q95']!r}",
        f"edge_safety_factor={generating['edge_safety_factor']!r}",
        # table cells, four, three and two decimals
        ship(abs(generating["plasma_current"]) / 1e6, 4),
        ship(abs(efit["efit-11"]["plasma_current"]) / 1e6, 4),
        ship(generating["flux_axis"], 3),
        ship(efit["efit-10"]["flux_axis"], 3),
        ship(efit["efit-11"]["flux_axis"], 3),
        ship(generating["flux_boundary"], 3),
        ship(efit["efit-10"]["flux_boundary"], 3),
        ship(efit["efit-11"]["flux_boundary"], 3),
        ship(generating["flux_span"], 3),
        ship(efit["efit-10"]["flux_span"], 3),
        ship(efit["efit-11"]["flux_span"], 3),
        # axes and nulls
        "({}, {})".format(
            ship(generating["axis"][0], 5), ship(generating["axis"][1], 5)
        ),
        "({}, {})".format(
            ship(efit["efit-10"]["axis"][0], 5), ship(efit["efit-10"]["axis"][1], 5)
        ),
        "({}, {})".format(
            ship(efit["efit-11"]["axis"][0], 5), ship(efit["efit-11"]["axis"][1], 5)
        ),
        # safety factor with the stored sign it carries
        "{} ({})".format(
            ship(abs(generating["q95"]), 3),
            "-1" if generating["q95_sign"] < 0 else "+1",
        ),
        "{} ({})".format(
            ship(abs(efit["efit-10"]["q95"]), 3),
            "-1" if efit["efit-10"]["q95_sign"] < 0 else "+1",
        ),
        "{} ({})".format(
            ship(abs(efit["efit-11"]["q95"]), 3),
            "-1" if efit["efit-11"]["q95_sign"] < 0 else "+1",
        ),
        ship(efit["efit-10"]["q_95_dd"], 3),
    ]
    missing = [value for value in expected if value not in record]
    assert not missing, f"the record no longer quotes {missing}"


def test_the_record_quotes_the_bands_it_measured(record, receipt):
    """Every band verdict and delta the record states is the measured one."""
    figures = derived(receipt)
    axis_offset = receipt["band_rows"]["imas-efit"]["axis_offset_metres"]
    axis_flux = receipt["band_rows"]["imas-efit"]["axis_flux_fraction_of_span"]
    boundary_flux = receipt["band_rows"]["imas-efit"]["boundary_flux_fraction_of_span"]
    q95_gap = abs(receipt["band_rows"]["imas-efit"]["q95_magnitude_difference"])

    expected = [
        exponent(abs(figures["nominal_current_fraction"])),
        ship(axis_offset, 4),
        ship(abs(axis_flux) * 100.0, 2),
        ship(abs(boundary_flux) * 100.0, 2),
        ship(q95_gap, 3),
    ]
    missing = [value for value in expected if value not in record]
    assert not missing, f"the record no longer quotes {missing}"

    # the record must state the q95 band as missed, and never as a passed five
    assert "The q95 band is not met" in record
    assert "Four of the five" in record
    assert "0.42 %" not in record
    assert "All five" not in record
    assert "five bands are met" not in record
    assert receipt["band_rows"]["imas-efit"]["bands"]["q95"] is False
    assert receipt["band_rows"]["nova fit"]["bands"]["q95"] is True


def test_the_record_quotes_the_amplitude_trade_it_attributed(record, receipt):
    """The split attributed to imas-efit is what its profiles show."""
    compact = record.replace(" ", "")
    amplitudes = receipt["profile_amplitudes"]
    pressure = (
        abs(amplitudes["efit_pressure"]) / amplitudes["generating_pressure"] - 1.0
    )
    diamagnetic = (
        1.0 - abs(amplitudes["efit_diamagnetic"]) / amplitudes["generating_diamagnetic"]
    )
    expected = [
        ship(pressure * 100.0, 2),
        ship(diamagnetic * 100.0, 2),
        ship(abs(amplitudes["efit_pressure"]), 1),
        ship(abs(amplitudes["efit_diamagnetic"]), 5),
        ship(amplitudes["generating_pressure"], 1),
        ship(amplitudes["generating_diamagnetic"], 5),
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
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert receipt["panels"][path.name] == digest, (
            f"{path.name} was rewritten after the receipt recorded it"
        )


def test_the_record_cites_the_check_that_guards_it():
    """The record names the receipt and the check, so a reader can rerun them."""
    text = normalise(FRAGMENT.read_text())
    assert "receipt.json" in text
    assert "test_fragment_receipt.py" in text


def test_no_paragraph_is_nested_in_another():
    """A paragraph opened inside a paragraph closes the outer one silently."""
    text = FRAGMENT.read_text()
    depth_of_open = 0
    for event in re.finditer(r"</?p\b[^>]*>", text):
        if event.group().startswith("</"):
            depth_of_open = max(depth_of_open - 1, 0)
        else:
            assert depth_of_open == 0, f"nested paragraph at offset {event.start()}"
            depth_of_open += 1
    assert depth_of_open == 0, "an unclosed paragraph"


def test_every_container_closes():
    """Sections, tables, figures and pre blocks open and close in equal number."""
    text = FRAGMENT.read_text()
    for tag in ("section", "table", "figure", "pre", "tbody", "thead", "tr"):
        opened = len(re.findall(rf"<{tag}\b", text))
        closed = len(re.findall(rf"</{tag}>", text))
        assert opened == closed, f"{tag}: {opened} opened, {closed} closed"


def test_the_record_carries_no_worktree_path():
    """A committed path into a removed worktree never resolves again."""
    text = FRAGMENT.read_text()
    assert "reckon" + "-worktrees" not in text
