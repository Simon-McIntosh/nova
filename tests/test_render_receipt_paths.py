"""Render receipts record project-relative paths, not worktree paths.

Every driver whose receipt names a rendered artifact must record that path
relative to the repository root, so the committed evidence still locates its
figure once the worktree that produced it is reclaimed.  The directory scan
reads the receipt JSON files the render drivers write; the helper check imports
each driver and asserts its path-recording helper relativises a source path.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Each render driver, paired with the receipt it writes.
RECEIPTS = {
    "benchmarks/sol_ledger_current_census.py": (
        "docs/figures/forward-solve-api/sol-ledger-census/sol-ledger-render.json",
    ),
    "docs/figures/cut-cell-current-attribution/outboard-hole/measure.py": (
        "docs/figures/cut-cell-current-attribution/outboard-hole/figure-receipt.json",
    ),
    "benchmarks/shafranov_pair_receipt.py": (
        "docs/figures/constraint-augmented-newton-krylov/shafranov/receipt.json",
    ),
    "benchmarks/shafranov_combination_discriminator.py": (
        "docs/figures/constraint-augmented-newton-krylov/"
        "shafranov-discriminator/receipt.json",
    ),
}

#: Directories holding the receipts, scanned for any JSON that still records a
#: worktree path.
RECEIPT_DIRECTORIES = tuple(
    sorted({Path(receipt).parent for paths in RECEIPTS.values() for receipt in paths})
)

#: JSON that is not written by a render route here: a persisted SLURM
#: submission script from an earlier run, which legitimately quotes the paths
#: it was launched with.
NON_RENDER_RECEIPTS = {
    "docs/figures/cut-cell-current-attribution/outboard-hole/preflight-receipt.json",
}


def _scanned_json():
    """Return every receipt-directory JSON the worktree-path scan inspects."""
    for directory in RECEIPT_DIRECTORIES:
        for json_path in sorted((ROOT / directory).glob("*.json")):
            relative = json_path.relative_to(ROOT).as_posix()
            if relative not in NON_RENDER_RECEIPTS:
                yield relative, json_path


def test_the_scan_reaches_every_receipt():
    """Positive control: the scan enumerates the receipts it exists to guard."""
    scanned = {relative for relative, _ in _scanned_json()}
    expected = {receipt for paths in RECEIPTS.values() for receipt in paths}
    missing = expected.difference(scanned)
    assert missing == set()
    for relative in expected:
        assert (ROOT / relative).read_text() != ""


def test_no_receipt_directory_json_records_a_worktree_path():
    offenders = [
        relative
        for relative, json_path in _scanned_json()
        if ".reckon-worktrees" in json_path.read_text()
    ]
    assert offenders == []


def _driver_module(relative_path, name):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_driver_helper_returns_a_project_relative_path():
    for index, relative_path in enumerate(RECEIPTS):
        module = _driver_module(relative_path, f"render_receipt_driver_{index}")
        helper = getattr(module, "repository_relative")
        recorded = helper(ROOT / relative_path)
        assert not os.path.isabs(recorded)
        assert recorded == relative_path
