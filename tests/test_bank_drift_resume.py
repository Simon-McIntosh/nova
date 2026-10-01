"""Contract of the paired operand-solve probe's resume path.

A campaign that ends early leaves the identities it finished checkpointed beside
its emission file, so re-running it must continue from there rather than start
again.  An allocation that re-solves every landed identity spends its wall clock
on work already done, and a re-emission that drops the checkpointed rows makes
an incomplete campaign look like a fresh one.

The contract is pinned here with a stubbed per-row worker, so it holds without
an allocation, a cache or a solve: the worker records the identities it is
asked for, which is the observable that says whether a landed identity was
re-run, and the emission file is read back to say whether every identity is
carried once.
"""

from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import pytest

from benchmarks import bank_drift_paired_probe as probe

CHECKPOINTED = ["21978/35", "21983/35"]
UNLANDED = "21990/35"


def _bank_row(shot: int, slice_index: int) -> dict[str, object]:
    """Return one decomposition-bank selection row."""

    return {"shot": shot, "slice_index": slice_index}


class _StubSolve:
    """The terminal state one arm of one identity resolves to."""

    converged = True
    terminal_residual = 1e-14
    termination_reason = "converged"

    def __init__(self) -> None:
        self.state = np.zeros(2)


class _StubReachability:
    """Return a terminal state per arm without running a solve."""

    def _mast_states(self, observed, state, target_current, carrier_identity=None):
        return {"pure": _StubSolve(), "mixed": _StubSolve()}

    def _grid_geometry(self, profile, state):
        return {
            "flux": np.zeros((2, 2)),
            "axis": np.zeros(2),
            "class_margin": 0.5,
        }


class _StubProducer:
    """A per-row worker that records the identities it is asked to solve.

    The recording sits on the first call the emission makes inside its row loop,
    so an identity the resume path skips never reaches it.  That makes the
    recorded list the observable for "was this landed identity re-run", rather
    than an inference from the shape of the output file.
    """

    def __init__(self, bank_rows: list[dict[str, object]]) -> None:
        self.bank_rows = bank_rows
        self.asked: list[str] = []
        self.DECOMPOSITION_BANK = "stub-bank"
        self.SHOT_STORE = "stub-store"
        self.response_carrier = types.SimpleNamespace(
            DEFAULT_CARRIER="stub-carrier",
            DEFAULT_RECEIPT="stub-receipt",
        )

    def _reachability_module(self):
        return _StubReachability()

    def configure_dtypes(self) -> None:
        return None

    def configure_persistent_compilation_cache(self, root):
        return types.SimpleNamespace(
            directory=str(root), receipt=lambda: {"directory": str(root)}
        )

    def _persisted_response_cache(self, carrier, receipt):
        return "stub-response-cache", {"carrier": carrier, "receipt": receipt}

    def _carrier_semantic_identity(self, evidence):
        return "stub-carrier-identity"

    def select_slices_by_shot(self, bank):
        return [(dict(row), "qualified") for row in self.bank_rows]

    def _mast_case_from_selection(self, store, selected_row, qualification):
        identity = f"{int(selected_row['shot'])}/{int(selected_row['slice_index'])}"
        self.asked.append(identity)
        return "stub-case", "stub-context"

    def _passive_inclusive_case(self, case, context, response_cache):
        profile = types.SimpleNamespace(operator=object(), source=None)
        return (
            {"reference": {"plasma_current_a": 1.0}, "state": np.zeros(2)},
            profile,
            "stub-policy",
        )

    def _ObservedProfile(self, profile):  # noqa: N802 - the producer's own name
        return types.SimpleNamespace(profile=profile)


def _checkpoint(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"tree_label": "main", "rows": rows}


def test_resume_keeps_checkpointed_rows_and_skips_their_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A landed identity is not re-solved and every identity is carried once.

    The checkpoint holds two identities the campaign already finished, and the
    bank offers those two beside one that never landed.  The stub worker records
    only the calls the row loop makes, so the two checkpointed identities
    appearing in that record is exactly the defect: work already done is done
    again.  The emission file is then read back to say that the landings are
    kept rather than regenerated -- the checkpointed row objects are compared
    whole, not merely counted -- and that no identity is carried twice.
    """

    out_path = tmp_path / "emit.json"
    landed = [
        {
            "identity": identity,
            "stages": {"profile_support": {"digest": marker}},
            "arms": {},
            "exception": None,
        }
        for identity, marker in zip(CHECKPOINTED, ("landed-a", "landed-b"), strict=True)
    ]
    out_path.write_text(json.dumps(_checkpoint(landed)))
    producer = _StubProducer(
        [_bank_row(21978, 35), _bank_row(21990, 35), _bank_row(21983, 35)]
    )
    monkeypatch.setattr(probe, "_load_module", lambda path, name: producer)

    status = probe._emit(
        "main",
        tmp_path,
        out_path,
        0,
        None,
        None,
        tmp_path / "cache",
    )

    assert status == 0
    assert producer.asked == [UNLANDED], (
        "a resumed campaign re-solved an identity the checkpoint had already landed"
    )
    payload = json.loads(out_path.read_text())
    assert payload["rows"][:2] == landed, (
        "the checkpointed rows were replaced rather than kept"
    )
    identities = [row["identity"] for row in payload["rows"]]
    assert identities == [*CHECKPOINTED, UNLANDED]
    assert len(identities) == len(set(identities))
