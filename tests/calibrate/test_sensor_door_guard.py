"""Keep MAST field signals behind the corrected waveform reader."""

from __future__ import annotations

import ast
from pathlib import Path

from nova.imas.mast_vacuum_cohort import RawArchiveReader

PACKAGE = Path(__file__).parents[2] / "nova" / "imas"
DOOR = PACKAGE / "mast_vacuum_cohort.py"
SWEEP = PACKAGE.parent / "scripts" / "mast_acquisition_sweep.py"


def _field_subscripts(path: Path) -> list[int]:
    """Return lines that index the archive field group directly."""

    tree = ast.parse(path.read_text(), filename=str(path))
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        key = node.slice
        if (isinstance(key, ast.Name) and key.id == "FIELD_GROUP") or (
            isinstance(key, ast.Constant) and key.value == "amb"
        ):
            lines.append(node.lineno)
    return lines


def test_no_mast_module_opens_the_sensor_group_outside_the_door():
    """Any direct field-group indexing outside the door is a read-path bypass."""

    bypasses = {
        path.relative_to(PACKAGE.parent.parent).as_posix(): _field_subscripts(path)
        for path in sorted(PACKAGE.glob("mast_*.py"))
        if path != DOOR and _field_subscripts(path)
    }
    assert bypasses == {}


def test_the_acquisition_sweep_names_its_raw_archive_reader():
    """Calibration reads raw by naming the object, never by disabling correction."""

    source = SWEEP.read_text()
    assert "RAW_ARCHIVE.read_shot_waveforms" in source
    assert "block_scale=RAW" not in source
    assert RawArchiveReader.__name__ == "RawArchiveReader"
