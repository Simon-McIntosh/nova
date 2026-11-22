"""Acceptance clause bounds and design budget coefficients, bound to their documents.

Each clause declares its bound and the plan section that states it, and the
design document's per-resolution membership, current and first-moment
coefficients and orders are declared for both analytic cases beside it.
"""

import html
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

DOCUMENTS = Path(__file__).resolve().parents[1]
PLAN_DOCUMENT = DOCUMENTS / "docs/plans/converged-forward-solve.html"
DESIGN_DOCUMENT = DOCUMENTS / "docs/research/converged-forward-solve-design.html"

ACCEPTANCE_SECTION = "s6"
BUDGET_SECTION = "reconstruction-orders"

EXECUTABLE_SIZE_LIMIT_BYTES = 50_000_000  # 50 MB, decimal
HOST_MEMORY_LIMIT_BYTES = 64_000_000_000  # 64 GB resident, decimal
COMPILE_WALL_LIMIT_SECONDS = 60.0
FRAME_RESIDUAL_BOUND = 1e-6
NORMAL_FORM_SADDLE_ORDER = 2.0


@dataclass(frozen=True)
class AcceptanceClause:
    """One stated acceptance clause, its bound and the plan section stating it."""

    key: str
    clause: str
    bound: float | None
    bound_text: str
    plan_section: str


ACCEPTANCE_CLAUSES: tuple[AcceptanceClause, ...] = (
    AcceptanceClause("global_residual", "global residual", 1e-12, "≤ 1e-12", "§6"),
    AcceptanceClause(
        "constraint_row", "every constraint row", 1e-12, "|row| ≤ 1e-12", "§6"
    ),
    AcceptanceClause(
        "null_position",
        "axis and every admitted X-point",
        1.0,
        "within one pitch of the analytic",
        "§6",
    ),
    AcceptanceClause(
        "map_fidelity",
        "map fidelity at the analytic state, no solve",
        1e-2,
        "≤ 1e-2 relative sup, exact mode, on the analytic certificate rows",
        "§6",
    ),
    AcceptanceClause(
        "program_count",
        "compiled programs per mesh shape",
        1.0,
        "one per call site (read, booking, map, solve), never a second compile "
        "of one call site for a second geometry of the same shape "
        "(lead, 2026-11-22)",
        "§6",
    ),
    AcceptanceClause(
        "executable_size",
        "executable size",
        EXECUTABLE_SIZE_LIMIT_BYTES,
        "≤ 50 MB",
        "§6",
    ),
    AcceptanceClause(
        "compile_wall",
        "compile wall",
        COMPILE_WALL_LIMIT_SECONDS,
        "≤ 60 s for the first call of a program served from the pinned pre-warm "
        "cache on the H200; the cold compile is recorded and carries no bound "
        "(lead, 2026-11-18, as kernel-compile-barrier states it)",
        "§6",
    ),
    AcceptanceClause(
        "host_memory",
        "host memory during that first call",
        HOST_MEMORY_LIMIT_BYTES,
        "≤ 64 GB resident on the host; the cold compile's peak is recorded; "
        "the 151 GB RSS measured on 2026-10-04 is the failure signature",
        "§6",
    ),
    AcceptanceClause(
        "tracing",
        "tracing",
        None,
        "jax.jit and jax.vmap over the production solve",
        "§6",
    ),
)

# Design budget coefficients: (case, metric, arm) -> (C, p). The weak case
# carries its stated budget; the diverted case's stated coefficients are the
# quadratic level-set before measurement, while its normal-form saddle
# acceptance uses order 2 with a coefficient measured from that arm.
DESIGN_BUDGET_COEFFICIENTS: dict[tuple[str, str, str], tuple[float, float]] = {
    ("weak", "membership_envelope", "stated"): (0.274714, 2.0),
    ("weak", "current_envelope", "stated"): (0.331217, 2.0),
    ("weak", "first_moment_envelope", "stated"): (0.205488, 2.0),
    ("diverted", "membership_envelope", "before"): (0.233718, 0.5),
    ("diverted", "current_envelope", "before"): (0.385203, 0.5),
    ("diverted", "first_moment_envelope", "before"): (0.238981, 0.5),
}

_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td>(.*?)</td>", re.S)
_TAG = re.compile(r"<[^>]+>")
_METRIC_CELL = re.compile(
    r'<td data-case="(?P<case>[^"]+)" data-metric="(?P<metric>[^"]+)"'
    r"(?P<attrs>[^>]*)>(?P<tag>[^<]*)</td>"
    r"",
    re.S,
)


def _text(fragment: str) -> str:
    """Collapse an HTML fragment to its single-spaced visible text."""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", fragment))).strip()


def _section_html(document: Path, anchor: str) -> str:
    """The HTML from a heading with ``id=anchor`` to the next heading."""
    text = document.read_text(encoding="utf-8")
    marker = f'<h2 id="{anchor}">'
    start = text.index(marker)
    following = text.find("<h2", start + len(marker))
    return text[start:] if following < 0 else text[start:following]


def _plan_clause_rows() -> tuple[tuple[str, str], ...]:
    """The clause and bound cells of the plan's acceptance table, in order."""
    section = _section_html(PLAN_DOCUMENT, ACCEPTANCE_SECTION)
    table = re.search(r"<table>(.*?)</table>", section, re.S)
    assert table is not None, f"no acceptance table in {PLAN_DOCUMENT}"
    rows = []
    for row in _ROW.findall(table.group(1)):
        cells = _CELL.findall(row)
        if len(cells) == 2:
            rows.append((_text(cells[0]), _text(cells[1])))
    return tuple(rows)


def _assert_clause_table_matches(
    clauses: tuple[AcceptanceClause, ...], rows: tuple[tuple[str, str], ...]
) -> None:
    declared = tuple((clause.clause, clause.bound_text) for clause in clauses)
    assert declared == rows, (
        "acceptance clause table no longer matches the declared bounds:\n"
        f"declared: {declared!r}\n"
        f"plan:     {rows!r}"
    )


def _design_budget_rows() -> dict[tuple[str, str, str], tuple[float, float]]:
    """The machine-readable budget coefficients published by the design document."""
    text = DESIGN_DOCUMENT.read_text(encoding="utf-8")
    table = re.search(rf'<table id="{BUDGET_SECTION}">(.*?)</table>', text, re.S)
    assert table is not None, f"no budget table in {DESIGN_DOCUMENT}"
    rows = {}
    for cell in _METRIC_CELL.finditer(table.group(1)):
        attrs = cell.group("attrs")
        arm = "before" if "data-before-budget-coefficient" in attrs else "stated"
        coefficient = re.search(r'data-(?:before-)?budget-coefficient="([^"]*)"', attrs)
        order = re.search(r'data-(?:before-)?budget-order="([^"]*)"', attrs)
        assert coefficient is not None and order is not None, (
            f"budget table cell lacks a coefficient or order: {cell.group(0)!r}"
        )
        rendered = re.search(r"C=([0-9.]+), p=([0-9.]+)", cell.group("tag"))
        assert rendered is not None
        published = (float(coefficient.group(1)), float(order.group(1)))
        printed = (float(rendered.group(1)), float(rendered.group(2)))
        assert published == printed, (
            f"{cell.group('case')} {cell.group('metric')} publishes "
            f"C={published[0]!r}, p={published[1]!r} but prints "
            f"C={printed[0]!r}, p={printed[1]!r}"
        )
        rows[(cell.group("case"), cell.group("metric"), arm)] = published
    return rows


def _assert_design_coefficients_match(
    declared: dict[tuple[str, str, str], tuple[float, float]],
    rows: dict[tuple[str, str, str], tuple[float, float]],
) -> None:
    assert declared == rows, (
        "design budget coefficients no longer match the declared values:\n"
        f"declared: {sorted(declared.items())!r}\n"
        f"document: {sorted(rows.items())!r}"
    )


def test_the_acceptance_section_heading_is_the_declared_one():
    heading = _section_html(PLAN_DOCUMENT, ACCEPTANCE_SECTION)
    assert _text(heading).startswith("§6")
    assert {clause.plan_section for clause in ACCEPTANCE_CLAUSES} == {"§6"}


def test_every_acceptance_clause_is_declared_once():
    keys = [clause.key for clause in ACCEPTANCE_CLAUSES]
    assert len(keys) == len(set(keys)) == 9
    for clause in ACCEPTANCE_CLAUSES:
        if clause.bound is not None:
            assert clause.bound > 0.0


def test_acceptance_clause_table_matches_the_plan_bound():
    _assert_clause_table_matches(ACCEPTANCE_CLAUSES, _plan_clause_rows())


def test_the_frame_row_bound_is_stated_by_the_plan():
    section = _text(_section_html(PLAN_DOCUMENT, ACCEPTANCE_SECTION))
    stated = re.search(r"at or below ([0-9.e+-]+)", section)
    assert stated is not None, "the frame row bound is no longer stated"
    assert float(stated.group(1)) == FRAME_RESIDUAL_BOUND, (
        f"the plan states {stated.group(1)!r} for the frame row, "
        f"declared {FRAME_RESIDUAL_BOUND!r}"
    )


def test_the_solve_default_carries_the_residual_bound():
    from nova.equilibrium.solve import DEFAULT_TOLERANCE

    residual = next(
        clause for clause in ACCEPTANCE_CLAUSES if clause.key == "global_residual"
    )
    assert residual.bound is not None
    assert DEFAULT_TOLERANCE == residual.bound, (
        f"solve default tolerance {DEFAULT_TOLERANCE!r} is not the stated bound "
        f"{residual.bound!r}"
    )


def test_design_budget_coefficients_match_the_design_document():
    _assert_design_coefficients_match(DESIGN_BUDGET_COEFFICIENTS, _design_budget_rows())


def test_both_analytic_cases_publish_three_budget_metrics():
    published = _design_budget_rows()
    for case in ("weak", "diverted"):
        metrics = {metric for row_case, metric, _arm in published if row_case == case}
        assert metrics == {
            "membership_envelope",
            "current_envelope",
            "first_moment_envelope",
        }


def test_the_normal_form_saddle_acceptance_uses_second_order():
    text = DESIGN_DOCUMENT.read_text(encoding="utf-8")
    assert NORMAL_FORM_SADDLE_ORDER == 2.0
    assert (
        f"acceptance budget <code>C_k(h/R₀)^{NORMAL_FORM_SADDLE_ORDER:g}</code>" in text
    )
    row = re.search(
        r"<tr><td>diverted saddle neighbourhood, acceptance</td>(.*?)</tr>",
        text,
        re.S,
    )
    assert row is not None
    assert row.group(1).count(f"p={NORMAL_FORM_SADDLE_ORDER:g}") == 3


def test_a_changed_bound_fails_with_both_values_printed():
    corrupted = (
        replace(ACCEPTANCE_CLAUSES[0], bound=1e-10, bound_text="≤ 1e-10"),
        *ACCEPTANCE_CLAUSES[1:],
    )
    with pytest.raises(AssertionError) as refusal:
        _assert_clause_table_matches(corrupted, _plan_clause_rows())
    assert "≤ 1e-10" in str(refusal.value)
    assert "≤ 1e-12" in str(refusal.value)


def test_a_corrupted_coefficient_fails_with_both_values_printed():
    corrupted = dict(DESIGN_BUDGET_COEFFICIENTS)
    corrupted[("weak", "membership_envelope", "stated")] = (0.2747, 2.0)
    with pytest.raises(AssertionError) as refusal:
        _assert_design_coefficients_match(corrupted, _design_budget_rows())
    assert "0.2747" in str(refusal.value)
    assert "0.274714" in str(refusal.value)
