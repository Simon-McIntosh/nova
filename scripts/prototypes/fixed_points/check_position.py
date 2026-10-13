"""Assert the declared stationary-point bound on a measured moving sweep."""

import json
import os
from pathlib import Path

import numpy as np


def test_cached_xpoint_position_bound():
    rows = json.loads(Path(os.environ["FIXED_POINT_CONTROL"]).read_text())
    errors = np.asarray([row["error_pitches"][-1] for row in rows])
    assert np.all(np.isfinite(errors)) and np.max(errors) <= 1.0, errors
    assert all(row["valid"][-1] for row in rows), rows
