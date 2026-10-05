"""Render receipts record project-relative paths, not worktree paths.

Every driver whose receipt names a rendered artifact must record that path
relative to the repository root, so the committed evidence still locates its
figure once the worktree that produced it is reclaimed.  The directory scan
reads the receipt JSON files the render drivers write; the render-route checks
run each driver's solve-free render path into a temporary directory that carries
its own ``.git`` marker and assert that every filesystem path the written receipt
records resolves inside that root.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

from nova.database.filepath import repository_relative

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

#: The receipt directories, scanned for any JSON that still records a worktree
#: path.  A historical SLURM preflight record once lived here and was excluded
#: as a non-render receipt; it now records repository-relative paths too, so the
#: scan covers every JSON in these directories without exception.
RECEIPT_DIRECTORIES = tuple(
    sorted({Path(receipt).parent for paths in RECEIPTS.values() for receipt in paths})
)


def _scanned_json():
    """Return every receipt-directory JSON the worktree-path scan inspects."""
    for directory in RECEIPT_DIRECTORIES:
        for json_path in sorted((ROOT / directory).glob("*.json")):
            yield json_path.relative_to(ROOT).as_posix(), json_path


def test_the_scan_reaches_every_receipt():
    """Positive control: the scan enumerates the receipts it exists to guard."""
    scanned = {relative for relative, _ in _scanned_json()}
    expected = {receipt for paths in RECEIPTS.values() for receipt in paths}
    missing = expected.difference(scanned)
    assert missing == set()
    # The preflight record carries no worktree path now, so the scan that is
    # meant to guard it must reach it rather than exclude it.
    assert (
        "docs/figures/cut-cell-current-attribution/outboard-hole/"
        "preflight-receipt.json" in scanned
    )
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


def test_each_driver_uses_the_shared_path_helper():
    """Each driver re-exports the one helper, rather than defining its own copy."""
    for index, relative_path in enumerate(RECEIPTS):
        module = _driver_module(relative_path, f"render_receipt_driver_{index}")
        helper = getattr(module, "repository_relative")
        assert helper is repository_relative
        recorded = helper(ROOT / relative_path)
        assert not os.path.isabs(recorded)
        assert recorded == relative_path


def _git_marked_root(tmp_path):
    """Return a temporary directory that carries its own ``.git`` marker.

    The marker makes the temporary directory the repository root a recorded path
    resolves against, so a path recorded relative to it cannot be an absolute
    worktree path that happens to exist elsewhere.
    """
    (tmp_path / ".git").mkdir(exist_ok=True)
    return tmp_path


def _assert_recorded_paths_are_relative(root, values):
    """Assert each recorded filesystem path is relative to ``root``."""
    for value in values:
        assert not Path(value).is_absolute(), value
        assert (root / value).exists(), value


def test_render_route_records_relative_paths_sol_ledger(tmp_path):
    """The ledger render route records the panel and receipt paths relative."""
    root = _git_marked_root(tmp_path)
    import benchmarks.sol_ledger_current_census as ledger

    shutil.copy(ledger.RECEIPT, root / "sol-ledger-census.json")
    metrics_path = root / "sol-ledger-render.json"
    ledger.render_from_receipt(
        receipt_path=root / "sol-ledger-census.json",
        output=root / "sol-ledger-current.png",
        metrics_path=metrics_path,
    )
    recorded = json.loads(metrics_path.read_text())
    _assert_recorded_paths_are_relative(
        root, (recorded["png"], recorded["svg"], recorded["receipt"])
    )


def test_render_route_records_relative_paths_measure(tmp_path):
    """The outboard-hole render route records its figures relative."""
    root = _git_marked_root(tmp_path)
    module = _driver_module(
        "docs/figures/cut-cell-current-attribution/outboard-hole/measure.py",
        "outboard_hole_measure",
    )
    module.render(output=root)
    receipt = json.loads((root / "figure-receipt.json").read_text())
    _assert_recorded_paths_are_relative(root, receipt["figures"])


def test_render_route_records_relative_paths_shafranov_pair(tmp_path):
    """The Shafranov pair render route records its row panels relative."""
    root = _git_marked_root(tmp_path)
    import benchmarks.shafranov_pair_receipt as pair

    for source in pair.DEFAULT_DIRECTORY.glob("*.json"):
        shutil.copy(source, root / source.name)
    pair.render_row_panels(directory=root)
    receipt = json.loads((root / "receipt.json").read_text())
    _assert_recorded_paths_are_relative(
        root,
        (entry["figure"]["filesystem_path"] for entry in receipt["rows_receipt"]),
    )


def test_render_route_records_relative_paths_shafranov_discriminator(tmp_path):
    """The discriminator render route records its state panels and strip relative."""
    root = _git_marked_root(tmp_path)
    import benchmarks.shafranov_combination_discriminator as discriminator

    for source in discriminator.DEFAULT_DIRECTORY.glob("*.json"):
        shutil.copy(source, root / source.name)
    discriminator.render_state_panels(directory=root)
    receipt = json.loads((root / "receipt.json").read_text())
    recorded = [
        entry["state_panel"][key]["filesystem_path"]
        for entry in receipt["rows_receipt"]
        for key in ("png", "svg")
    ]
    recorded += [receipt["figure"][key]["filesystem_path"] for key in ("png", "svg")]
    _assert_recorded_paths_are_relative(root, recorded)
