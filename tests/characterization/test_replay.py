"""Replay lane -- recorded fits re-run and re-checked against their goldens.

Tier-1 (implemented here): for every run record whose inputs resolve against
the local canonical corpus and whose entry point is runnable, re-run the fit
with the current code and confirm it still matches the golden the record
pinned, at the record's tolerance classes.

Tier-2 (opt-in): reconstruct each record's historical environment -- check out
its git revision, restore its dependency lock into an isolated venv, and replay
under that interpreter to detect drift the current environment hides. It is a
full dependency sync per record, so it is gated behind ``NOVA_TIER2_REPLAY=1``;
otherwise it is a single visible opt-in skip. When the historical environment
cannot be rebuilt, the result degrades visibly (reproduced by current code,
historical env unreconstructable, with the failing stage) rather than passing
or skipping silently.
"""

from __future__ import annotations

import os

import pytest

from . import _registry, _replay

_RECORDS = _replay.run_records()
_ENTRIES = {entry.id: entry for entry in _registry.registry()}

# The record the Tier-2 pilot exercises when opted in.
_TIER2_PILOT_FIT_ID = "sector.fit.ssat"


def _record_ids(records):
    return [path.name for path, _ in records]


def test_run_records_present_and_valid():
    """The run-record registry exists and every record parses and validates.

    ``RunRecord.load`` validates on read, so a malformed record raises during
    collection above; here we assert the pilot record set is non-empty so a
    silently empty registry cannot masquerade as success.
    """
    assert _RECORDS, "no run records found under data/Assembly/run_records"


@pytest.mark.parametrize("path,record", _RECORDS, ids=_record_ids(_RECORDS))
def test_record_replays_to_golden(path, record):
    entry = _ENTRIES.get(record.fit_id)
    if entry is None:
        pytest.skip(f"no registered entry point for fit_id {record.fit_id!r}")

    reason = entry.skip_reason()
    if reason is not None:
        pytest.skip(f"entry point not runnable here: {reason}")

    missing = _replay.unresolved_inputs(record)
    if missing:
        pytest.skip(f"recorded inputs absent from local corpus: {missing}")

    result = _replay.replay(record)
    assert result.checked > 0, f"{record.fit_id}: record pins no outputs to replay"
    assert result.passed, f"{record.fit_id} replay failed:\n" + "\n".join(
        result.failures
    )


def _tier2_pilot_record():
    """Return the (path, record) the Tier-2 pilot replays, or None if absent."""
    for path, record in _RECORDS:
        if record.fit_id == _TIER2_PILOT_FIT_ID:
            return path, record
    return None


def test_record_replays_in_historical_environment():
    """Tier-2 pilot: replay the sector fit under its reconstructed environment.

    Expensive (a full ``uv sync`` of the historical closure), so it runs only
    under ``NOVA_TIER2_REPLAY=1``; otherwise a single visible skip says how to
    opt in. When opted in, the pilot either passes end-to-end or degrades
    visibly with the precise stage that could not be reconstructed -- never a
    silent pass. Genuine drift (a clean record whose historical run diverges)
    fails.
    """
    if os.environ.get(_replay.TIER2_ENV_FLAG) != "1":
        pytest.skip(
            f"Tier-2 historical-environment replay is opt-in (a full dependency "
            f"sync per record); set {_replay.TIER2_ENV_FLAG}=1 to run it"
        )

    found = _tier2_pilot_record()
    if found is None:
        pytest.skip(f"no run record for Tier-2 pilot {_TIER2_PILOT_FIT_ID!r}")
    _, record = found

    entry = _ENTRIES.get(record.fit_id)
    if entry is None:
        pytest.skip(f"no registered entry point for fit_id {record.fit_id!r}")
    reason = entry.skip_reason()
    if reason is not None:
        pytest.skip(f"entry point not runnable here: {reason}")
    missing = _replay.unresolved_inputs(record)
    if missing:
        pytest.skip(f"recorded inputs absent from local corpus: {missing}")

    result = _replay.replay_historical(record)

    if result.degraded:
        pytest.skip(result.message())
    assert result.passed, result.message()
