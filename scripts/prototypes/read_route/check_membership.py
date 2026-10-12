"""Assert the membership envelope on a named persisted Bell read receipt."""

import json
import os
from pathlib import Path


def test_bell_saddle_membership():
    row = json.loads(Path(os.environ["BELL_MEMBERSHIP_ROW"]).read_text())
    assert row["valid"] and row["qualified"], row
    assert row["saddle_membership_error"] <= row["membership_bound"], row
