"""Assert the topology-batch arm figures are titled and readable as served.

The probe under ``benchmarks/topology_batch_probe.py`` measures the batched
connectivity kernels once per arm and leaves a receipt beside each figure. The
probe's ``render`` operand rebuilds every arm figure from those committed
receipts and writes ``render-receipt.json`` in the figure directory, which is
what these checks read: no device and no solve is built.

Each arm receipt records the row-index pool it measured in its ``identity``
field. That list is a data record, and the suptitle must name the arm and the
revision instead, because a bare integer list runs off both canvas edges and
names neither. The checks below also require the title to agree with the arm
receipt it was drawn from, require the middle wall panel to be drawn on a log
axis (its series span an order of magnitude, so a linear axis buries all but the
slowest at the axis floor), and require no served file name to end in a digit
suffix.

Point ``NOVA_TOPOLOGY_BATCH_RENDER_RECEIPT`` at another render receipt to audit
it, and at a copy whose titles carry the row-index list to watch the checks
fail.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRECTORY = ROOT / "docs/figures/playable-forward-solve/topology-batch"
DEFAULT_RENDER_RECEIPT = FIGURE_DIRECTORY / "render-receipt.json"
ARMS = {"before", "after", "corrected"}
BATCHES = (1, 8, 16, 64)
COMMA_INTEGER_RUN = re.compile(r"\d+(?:\s*,\s*\d+){2,}")
DIGIT_SUFFIX = re.compile(r"\d$")


def _render_receipt_path() -> Path:
    return Path(
        os.environ.get(
            "NOVA_TOPOLOGY_BATCH_RENDER_RECEIPT", str(DEFAULT_RENDER_RECEIPT)
        )
    )


@pytest.fixture(scope="module")
def render_receipt() -> dict:
    path = _render_receipt_path()
    return json.loads(path.read_text(encoding="utf-8"))


def _title_defects(record: dict) -> list[str]:
    """Return the reasons a figure title is unusable; empty when sound."""
    title = record["title"]
    defects = []
    if record["arm"] not in title:
        defects.append("title omits the arm " + record["arm"])
    if record["revision"][:8] not in title:
        defects.append("title omits the revision")
    if COMMA_INTEGER_RUN.search(title):
        defects.append("comma-separated integer list: " + title)
    return defects


def test_the_render_entry_point_records_a_completed_figure_set(render_receipt):
    assert render_receipt["schema"] == "nova.topology-batch-render-receipt"
    assert render_receipt["completed"] is True
    entry = render_receipt["render_entry_point"]
    assert entry == "benchmarks/topology_batch_probe.py render"
    arms = {rec["arm"] for rec in render_receipt["figures"]}
    assert arms == ARMS


def test_each_title_names_its_arm_and_revision(render_receipt):
    for record in render_receipt["figures"]:
        assert _title_defects(record) == []


def test_no_title_carries_a_completed_row_index_list(render_receipt):
    for record in render_receipt["figures"]:
        assert COMMA_INTEGER_RUN.search(record["title"]) is None


def test_the_title_check_rejects_a_row_index_list_title(render_receipt):
    record = copy.deepcopy(render_receipt["figures"][0])
    record["title"] = "5, 6, 7, 8, 9, 10, 11"
    defects = _title_defects(record)
    assert defects
    joined = " ".join(defects)
    assert "comma-separated integer list" in joined
    assert "revision" in joined


def test_each_title_agrees_with_the_receipt(render_receipt):
    directory = _render_receipt_path().parent
    for record in render_receipt["figures"]:
        name = record["source_receipt"]
        source = json.loads((directory / name).read_text(encoding="utf-8"))
        assert source["tag"] == record["arm"]
        assert source["source_commit"] == record["revision"]
        assert record["arm"] in record["title"]
        assert record["revision"][:8] in record["title"]


def test_the_middle_panel_is_drawn_on_a_log_axis(render_receipt):
    directory = _render_receipt_path().parent
    for record in render_receipt["figures"]:
        panels = record["panels"]
        assert len(panels) == 3
        assert panels[1]["yscale"] == "log"
        assert len(panels[1]["series"]) == 5
        name = record["source_receipt"]
        source = json.loads((directory / name).read_text(encoding="utf-8"))
        serial = []
        for entry in source["kernels"].values():
            if "batches" not in entry:
                continue
            for batch in BATCHES:
                serial.append(entry["batches"][str(batch)]["serial_per_element_s"])
        assert len(serial) == 5 * len(BATCHES)
        assert max(serial) > 8 * min(serial)


def test_the_files_the_receipt_names_are_on_disk(render_receipt):
    directory = _render_receipt_path().parent
    for record in render_receipt["figures"]:
        for key in ("figure", "svg", "source_receipt"):
            path = (directory / record[key]).resolve()
            assert path.is_file(), path
            assert path.stat().st_size > 0


def test_no_served_file_name_ends_in_a_digit_suffix():
    names = sorted(p.name for p in FIGURE_DIRECTORY.iterdir() if p.is_file())
    assert names
    offenders = [n for n in names if DIGIT_SUFFIX.search(Path(n).stem)]
    assert offenders == []
    assert "topology-batch-corrected.png" in names
    assert not any("after2" in n for n in names)
