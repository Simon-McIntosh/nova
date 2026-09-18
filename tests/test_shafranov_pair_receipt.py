"""The Shafranov row receipt never states a combination the row did not reach.

A row the solve refuses has no terminal state, so the target it was refused
against is not an emission of the row: it belongs in ``target_combination`` and
must not appear in ``terminal_profile_combination``, which is what a reader
takes as the combination the row terminated at.  The refused rows of the bank
measurement state the refusal and a null terminal combination, and both the
measurement and the replay from a banked lane log take the field through one
rule, so a receipt cannot carry the target whichever path is not taken.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from benchmarks.shafranov_pair_receipt import (
    EMISSION_PREFIX,
    ROW_FIELDS,
    _emission_entry,
    _terminal_combination,
    regenerate,
)

#: The committed bank receipts, which are what a reader of the plan sees.
DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "docs/figures/constraint-augmented-newton-krylov/shafranov"
)


def _banked_emission(identity: str) -> dict:
    """Return one committed row receipt's fields as a lane emission carries them."""
    document = json.loads(
        (DIRECTORY / f"row-{identity.replace('/', '-')}.json").read_text(
            encoding="utf-8"
        )
    )
    return {field: document[field] for field in ROW_FIELDS}


def _refused_identities(receipt: dict) -> list[str]:
    return [
        entry["identity"]
        for entry in receipt["rows_receipt"]
        if entry["status"] == "refused"
    ]


def _committed_identities() -> list[str]:
    receipt = json.loads((DIRECTORY / "receipt.json").read_text(encoding="utf-8"))
    return [entry["identity"] for entry in receipt["rows_receipt"]]


def _emissions_log(path: Path, identities: list[str]) -> Path:
    """Write the row emissions a banked lane log carries for ``identities``."""
    path.write_text(
        "\n".join(
            EMISSION_PREFIX + json.dumps(_banked_emission(identity), sort_keys=True)
            for identity in identities
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_a_refused_row_replays_without_the_target_as_its_terminal_combination():
    """The seeded field the defect wrote is not the field the receipt reports.

    The banked emission of a refused row carries the target in the terminal
    field when the driver seeded it there, so the replay has to overwrite it
    rather than pass it through unchanged.
    """
    emission = _banked_emission("21978/35")
    assert emission["status"] == "refused"
    target = emission["target_combination"]
    assert target is not None
    emission["terminal_profile_combination"] = target

    entry = _emission_entry(emission, figure=None)

    assert entry["terminal_profile_combination"] is None
    assert entry["terminal_profile_combination"] != target
    assert entry["target_combination"] == target
    for field in (
        "status",
        "refusal_reason",
        "observed_combination_at_reference",
        "reference_combination_gap",
        "minor_radius_m",
        "plasma_current_a",
    ):
        assert entry[field] == emission[field]


def test_an_imposed_row_replays_the_emission_it_carries():
    """The positive control: the rule is not a blanket refusal of the field.

    A row that terminated must report the combination its own profiles read,
    so an emission carrying one replays carrying the same value.
    """
    emission = _banked_emission("21978/35")
    settled = float(emission["target_combination"])
    emission["status"] = "imposed"
    emission["terminal_profile_combination"] = settled

    entry = _emission_entry(emission, figure=None)

    assert entry["terminal_profile_combination"] == settled
    assert _terminal_combination("refused", settled) is None
    assert _terminal_combination("imposed", settled) == settled


def test_every_committed_refused_row_states_a_null_terminal_combination():
    """The committed receipts, read as a reader of the plan reads them.

    The denominator is asserted rather than assumed: the receipt holds six
    rows and the row set is what the bank qualifies, so a receipt that
    silently lost rows cannot pass this test by holding no refused row.
    """
    receipt = json.loads((DIRECTORY / "receipt.json").read_text(encoding="utf-8"))
    refused = _refused_identities(receipt)
    assert len(receipt["rows_receipt"]) == 6
    assert len(refused) == 6
    for entry in receipt["rows_receipt"]:
        assert entry["terminal_profile_combination"] is None
        assert entry["terminal_profile_combination"] != entry["target_combination"]
        completed = json.loads(
            (DIRECTORY / f"row-{entry['identity'].replace('/', '-')}.json").read_text(
                encoding="utf-8"
            )
        )
        for field in ROW_FIELDS:
            assert completed[field] == entry[field]
        assert completed["target_combination"] is not None
        assert completed["observed_combination_at_reference"] is not None


def test_a_lane_log_that_drops_a_row_is_refused_before_anything_is_written(tmp_path):
    """The replay's row-set guard fires, and the refusal is not silently partial.

    A lane log holding fewer rows than the receipt names must be refused
    outright: were the guard dropped, a truncated nine-row lane would rewrite
    the receipt from a partial row set, so the guard is exercised here rather
    than left to the absence of a caller.  The check is the write itself --
    every committed file is compared byte for byte across the refusal.
    """
    directory = tmp_path / "shafranov"
    shutil.copytree(DIRECTORY, directory)
    identities = _committed_identities()
    assert len(identities) == 6
    emissions = _emissions_log(tmp_path / "truncated.log", identities[:-1])

    before = {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }

    with pytest.raises(ValueError, match="different row set"):
        regenerate(emissions=emissions, directory=directory)

    after = {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }
    assert after == before


def test_a_lane_log_matching_the_receipt_regenerates_the_null_terminal(tmp_path):
    """The positive control for that guard: a replay it admits must write.

    The same call with the full row set replays every row, so the refusal
    above is the row-set check and not a mechanism that never writes.  Each
    refused row still carries a null terminal combination after the replay.
    """
    directory = tmp_path / "shafranov"
    shutil.copytree(DIRECTORY, directory)
    identities = _committed_identities()
    emissions = _emissions_log(tmp_path / "complete.log", identities)

    receipt = regenerate(emissions=emissions, directory=directory)

    assert len(receipt["rows_receipt"]) == len(identities)
    assert sorted(entry["identity"] for entry in receipt["rows_receipt"]) == sorted(
        identities
    )
    for entry in receipt["rows_receipt"]:
        assert entry["status"] == "refused"
        assert entry["terminal_profile_combination"] is None
        assert entry["target_combination"] is not None
        written = json.loads(
            (directory / f"row-{entry['identity'].replace('/', '-')}.json").read_text(
                encoding="utf-8"
            )
        )
        assert written["terminal_profile_combination"] is None
        assert written["figure"] is not None
