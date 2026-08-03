"""Cover the machine-description evidence contract.

Every authored field carries one evidence state, and each state has to pay for
itself: a citation, an interval with the assumptions behind it, or an explicit
statement of what is missing. These tests pin those obligations, the canonical
ordering, and the round trip through JSON that lets a manifest carry a ledger.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nova.imas.machine_evidence import (
    EvidenceError,
    EvidenceLedger,
    EvidenceRecord,
    FieldEvidence,
    SourceReference,
    Uncertainty,
)

SOURCE = SourceReference(
    title="A machine description",
    url="https://example.invalid/machine.pdf",
    locator="p. 7",
    machine="mast",
    text_verified=True,
)
INTERVAL = Uncertainty(lower=1.0, upper=2.0, unit="ohm")


def _record(evidence: FieldEvidence, **changes) -> EvidenceRecord:
    return replace(
        EvidenceRecord(
            path="pf_passive/loop(vertw)/resistance",
            evidence=evidence,
            first_shot=11695,
            last_shot=30473,
            statement="a stated fact",
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "evidence",
    [FieldEvidence.MEASURED, FieldEvidence.PUBLISHED],
)
def test_measured_and_published_fields_must_cite_a_source(evidence) -> None:
    with pytest.raises(EvidenceError, match="must cite a source"):
        _record(evidence).validate()

    _record(evidence, source=SOURCE).validate()


@pytest.mark.parametrize(
    "evidence",
    [FieldEvidence.GENERATED, FieldEvidence.FITTED],
)
def test_derived_fields_must_bound_the_value_and_state_assumptions(evidence) -> None:
    with pytest.raises(EvidenceError, match="must carry an uncertainty"):
        _record(evidence, assumptions=("a stated prior",)).validate()
    with pytest.raises(EvidenceError, match="must state its assumptions"):
        _record(evidence, uncertainty=INTERVAL).validate()

    _record(evidence, uncertainty=INTERVAL, assumptions=("a stated prior",)).validate()


def test_unresolved_field_states_what_is_missing_and_bounds_nothing() -> None:
    with pytest.raises(EvidenceError, match="must state what is missing"):
        _record(FieldEvidence.UNRESOLVED).validate()
    with pytest.raises(EvidenceError, match="cannot bound a value it lacks"):
        _record(
            FieldEvidence.UNRESOLVED,
            assumptions=("no source",),
            uncertainty=INTERVAL,
        ).validate()
    with pytest.raises(EvidenceError, match="cannot offer one candidate"):
        _record(
            FieldEvidence.UNRESOLVED,
            assumptions=("no source",),
            candidates=("the only option",),
        ).validate()

    _record(
        FieldEvidence.UNRESOLVED,
        assumptions=("no source states it",),
        candidates=("one bank", "the other bank"),
    ).validate()


def test_a_resolved_field_cannot_keep_candidates_open() -> None:
    with pytest.raises(EvidenceError, match="cannot keep open candidates"):
        _record(
            FieldEvidence.PUBLISHED,
            source=SOURCE,
            candidates=("one bank", "the other bank"),
        ).validate()


def test_candidates_are_canonically_ordered() -> None:
    with pytest.raises(EvidenceError, match="must be sorted"):
        _record(
            FieldEvidence.UNRESOLVED,
            assumptions=("no source",),
            candidates=("second", "first"),
        ).validate()


def test_only_an_unresolved_field_can_block_the_forward_model() -> None:
    with pytest.raises(EvidenceError, match="cannot block the forward model"):
        _record(
            FieldEvidence.MEASURED,
            source=SOURCE,
            blocks_axisymmetric_forward_model=True,
        ).validate()

    blocked = _record(
        FieldEvidence.UNRESOLVED,
        assumptions=("no source",),
        blocks_axisymmetric_forward_model=True,
    )
    blocked.validate()
    assert EvidenceLedger.create([blocked]).forward_model_blockers() == (blocked.path,)


def test_ledger_rejects_two_states_for_one_field_over_the_same_shots() -> None:
    measured = _record(FieldEvidence.MEASURED, source=SOURCE)
    unresolved = _record(FieldEvidence.UNRESOLVED, assumptions=("no source",))

    with pytest.raises(EvidenceError, match="two evidence states"):
        EvidenceLedger.create([measured, unresolved])


def test_ledger_accepts_one_field_changing_state_between_shot_ranges() -> None:
    early = _record(
        FieldEvidence.GENERATED,
        last_shot=20000,
        uncertainty=INTERVAL,
        assumptions=("a nominal prior",),
    )
    late = _record(
        FieldEvidence.FITTED,
        first_shot=20001,
        uncertainty=INTERVAL,
        assumptions=("identified on held-out shots",),
    )

    ledger = EvidenceLedger.create([late, early])

    assert ledger.records == (early, late)
    assert ledger.for_shot(15000).records == (early,)
    assert ledger.for_shot(25000).records == (late,)
    assert ledger.for_shot(11000).records == ()


def test_ledger_ordering_and_digest_are_independent_of_input_order() -> None:
    first = _record(FieldEvidence.MEASURED, source=SOURCE)
    second = _record(FieldEvidence.MEASURED, path="wall/limiter", source=SOURCE)

    forward = EvidenceLedger.create([first, second])
    reversed_input = EvidenceLedger.create([second, first])

    assert forward.records == reversed_input.records
    assert forward.digest == reversed_input.digest
    assert EvidenceLedger(records=(second, first)).records != forward.records
    with pytest.raises(EvidenceError, match="canonically ordered"):
        EvidenceLedger(records=(second, first)).validate()


def test_ledger_round_trips_through_canonical_json() -> None:
    ledger = EvidenceLedger.create(
        [
            _record(FieldEvidence.MEASURED, source=SOURCE),
            _record(
                FieldEvidence.GENERATED,
                path="pf_passive/loop(vertw)/resistivity",
                uncertainty=INTERVAL,
                assumptions=("a bulk material prior",),
            ),
            _record(
                FieldEvidence.UNRESOLVED,
                path="tf/r0",
                assumptions=("not sourced",),
            ),
        ]
    )

    restored = EvidenceLedger.from_list(ledger.as_list())

    assert restored == ledger
    assert restored.canonical_bytes() == ledger.canonical_bytes()
    assert b"mtime" not in ledger.canonical_bytes()
    assert ledger.state_counts() == {
        "measured": 1,
        "published": 0,
        "generated": 1,
        "fitted": 0,
        "unresolved": 1,
    }


def test_decoder_rejects_unknown_states_and_malformed_rows() -> None:
    rows = EvidenceLedger.create([_record(FieldEvidence.MEASURED, source=SOURCE)])
    payload = rows.as_list()
    payload[0]["evidence"] = "assumed"

    with pytest.raises(EvidenceError, match="unknown evidence state"):
        EvidenceLedger.from_list(payload)
    with pytest.raises(EvidenceError, match="must be an object"):
        EvidenceLedger.from_list(["not a record"])
    with pytest.raises(EvidenceError, match="fields differ"):
        EvidenceLedger.from_list([{"path": "tf/r0"}])


def test_citation_must_be_followable() -> None:
    with pytest.raises(EvidenceError, match="must be https"):
        replace(SOURCE, url="http://example.invalid/machine.pdf").validate()
    with pytest.raises(EvidenceError, match="must be lowercase"):
        replace(SOURCE, machine="MAST").validate()
    with pytest.raises(EvidenceError, match="non-empty trimmed"):
        replace(SOURCE, locator=" p. 7").validate()
    assert SourceReference.from_dict(SOURCE.as_dict()) == SOURCE


def test_interval_must_be_ordered_finite_and_carry_a_unit() -> None:
    with pytest.raises(EvidenceError, match="precedes lower bound"):
        Uncertainty(lower=2.0, upper=1.0, unit="ohm").validate()
    with pytest.raises(EvidenceError, match="must be finite"):
        Uncertainty(lower=float("nan"), upper=1.0, unit="ohm").validate()
    with pytest.raises(EvidenceError, match="uncertainty unit"):
        Uncertainty(lower=1.0, upper=2.0, unit="").validate()

    assert INTERVAL.contains(1.5)
    assert not INTERVAL.contains(2.5)
    assert Uncertainty.from_dict(INTERVAL.as_dict()) == INTERVAL


def test_shot_support_must_be_a_real_interval() -> None:
    with pytest.raises(EvidenceError, match="must not precede first shot"):
        _record(FieldEvidence.MEASURED, source=SOURCE, last_shot=11000).validate()
    with pytest.raises(EvidenceError, match="must be non-negative"):
        _record(
            FieldEvidence.MEASURED,
            source=SOURCE,
            first_shot=-1,
            last_shot=0,
        ).validate()
