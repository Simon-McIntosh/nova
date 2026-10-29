"""Guards on the certificate driver's banked part receipts.

Every case and resolution the certificate measures keeps one part receipt under
``docs/figures/gs-absolute-accuracy/solovev/production-route-parts/``. The driver
redraws a banked row's figure without entering the solver, so a receipt is the
only claim about a solve that path has. It may be served only where the tree
reproduces the carrier the solve was built on and the current it was declared
against; otherwise the row is solved, or refused by name, but never reported as
a solve this tree did not run.

And the diverted 300-cell row itself must converge inside the qualification
bound.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes

CASE = "diverted-single-null"
REQUESTED_CELLS = -300
PART_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "docs/figures/gs-absolute-accuracy/solovev/production-route-parts"
)
BANKED_PART = PART_DIRECTORY / f"{CASE}-production-route-cells-300.json"


@pytest.fixture(scope="module")
def setup() -> dict[str, object]:
    """Return the machine, fixture and declared current for this row."""

    configure_dtypes()
    return certificate._certificate_case_setup(CASE, REQUESTED_CELLS, None, {})


def _banked_receipt() -> dict[str, object]:
    return json.loads(BANKED_PART.read_text(encoding="utf-8"))


def _receipt_for(setup: dict[str, object], target_current: float) -> dict:
    """Return a receipt describing the given declared here and current."""

    return {
        "case": CASE,
        "requested_cells": REQUESTED_CELLS,
        "realised_cells": int(len(setup["machine"].node)),
        "cache": dict(setup["machine"].cache),
        "solver": {"target_current_a": target_current},
        "render_data": {},
    }


def test_receipt_from_another_carrier_is_refused(setup: dict[str, object]) -> None:
    """A receipt whose declared current is from another carrier is refused."""

    banked = _banked_receipt()
    recorded = float(banked["solver"]["target_current_a"])
    declared = float(setup["target_current"])
    assert recorded != pytest.approx(declared, rel=1.0e-12), (
        "the banked receipt now matches the tree, so it no longer exercises the "
        "refusal this guard exists for"
    )
    verdict = certificate._receipt_verdict(banked, setup)
    assert verdict["accepted"] is False
    assert "target_current" in verdict["reason"]
    assert repr(recorded) in verdict["reason"]
    assert repr(declared) in verdict["reason"]


def test_receipt_from_this_carrier_is_accepted(setup: dict[str, object]) -> None:
    """The check accepts the receipt it can reproduce."""

    receipt = _receipt_for(setup, float(setup["target_current"]))
    verdict = certificate._receipt_verdict(receipt, setup)
    assert verdict["accepted"] is True, verdict["reason"]
    assert verdict["reason"] == "carrier_and_current_match"


def test_receipt_with_a_foreign_carrier_key_is_refused(
    setup: dict[str, object],
) -> None:
    """The carrier identity alone refuses a receipt even at the right current."""

    receipt = _receipt_for(setup, float(setup["target_current"]))
    receipt["cache"] = {**receipt["cache"], "semantic_key": "0000000000000000"}
    verdict = certificate._receipt_verdict(receipt, setup)
    assert verdict["accepted"] is False
    assert "carrier_semantic_key" in verdict["reason"]


def test_receipt_with_a_different_cell_count_is_refused(
    setup: dict[str, object],
) -> None:
    """A receipt that realised another number of cells is refused."""

    receipt = _receipt_for(setup, float(setup["target_current"]))
    receipt["realised_cells"] = int(receipt["realised_cells"]) + 1
    verdict = certificate._receipt_verdict(receipt, setup)
    assert verdict["accepted"] is False
    assert "realised_cells" in verdict["reason"]


def test_measure_refuses_a_foreign_receipt_before_it_serves(
    setup: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused receipt sends the driver into the solver, not into the figure.

    The solver is replaced by a marker so the test shows which path the driver
    took rather than what it returned.
    """

    banked = _banked_receipt()
    assert certificate._receipt_verdict(banked, setup)["accepted"] is False

    part = tmp_path / BANKED_PART.name
    part.write_text(json.dumps(banked), encoding="utf-8")

    class _EnteredSolver(RuntimeError):
        pass

    def _refuse_to_solve(*_arguments, **_keywords):
        raise _EnteredSolver("the driver entered the solver")

    rendered: list[dict] = []
    monkeypatch.setattr(certificate, "_part_path", lambda *_a, **_k: part)
    monkeypatch.setattr(
        certificate, "_certificate_case_setup", lambda *_a, **_k: setup
    )
    monkeypatch.setattr(
        certificate,
        "_render_persisted_row",
        lambda row, path=None: rendered.append(row) or copy.deepcopy(row),
    )
    monkeypatch.setattr(certificate.ForwardProfile, "solve", _refuse_to_solve)

    with pytest.raises(_EnteredSolver):
        certificate._measure(CASE, REQUESTED_CELLS)
    assert rendered == [], "a refused receipt was rendered instead of solved"


@pytest.mark.slow
def test_diverted_300_row_converges(setup: dict[str, object]) -> None:
    """The diverted 300-cell row reaches the certificate's residual bound.

    The row is solved through the same seam the driver uses and the banked part
    is never written, so the test cannot refresh the receipt it is measuring
    against.
    """

    import jax

    profile = setup["profile"]
    target_current = setup["target_current"]
    assert isinstance(profile, certificate.ForwardProfile)
    seed, _requested_class, _receipt = certificate._production_seed(
        profile,
        CASE,
        target_current,
        setup["current_centroid"],
        setup["current_receipt"],
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{CASE}:{REQUESTED_CELLS}",
    )
    solved = profile.solve(request)
    jax.block_until_ready(solved.equilibrium.flux)
    residual = float(solved.equilibrium.fixed_point.residual)
    assert residual <= certificate.TERMINAL_RESIDUAL_BOUND, (
        f"terminal residual {residual!r} above "
        f"{certificate.TERMINAL_RESIDUAL_BOUND!r}"
    )