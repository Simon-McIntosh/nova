"""Replay lane -- recorded fits re-run and re-checked against their goldens.

Tier-1 (implemented here): for every run record whose inputs resolve against
the local canonical corpus and whose entry point is runnable, re-run the fit
with the current code and confirm it still matches the golden the record
pinned, at the record's tolerance classes.

Tier-2 (not implemented): reconstruct each record's historical environment --
its git revision and dependency lock -- and replay under that to detect drift
the current environment hides. A marker test names it so the gap is visible.
"""

from __future__ import annotations

import pytest

from . import _registry, _replay

_RECORDS = _replay.run_records()
_ENTRIES = {entry.id: entry for entry in _registry.registry()}


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


@pytest.mark.skip(
    reason=(
        "Tier-2 historical-environment replay not implemented: would check out "
        "each record's code_git_sha, restore its uv_lock_sha256 dependency set, "
        "and replay under that environment to surface drift the current "
        "environment masks. Out of scope for the Tier-1 foundation."
    )
)
def test_record_replays_in_historical_environment():
    """Placeholder marking Tier-2 historical-environment replay as future work."""
