"""Read back measured integration arms against the centroid registration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


def test_seed_current_matches_original_registration() -> None:
    path = Path(os.environ["NOVA_SEED_CONTROL_RECEIPT"])
    receipt = json.loads(path.read_text())
    arm = os.environ.get("NOVA_SEED_CONTROL_ARM", "centroid")
    values = np.asarray(receipt[f"{arm}_current"], dtype=float)
    assert values.shape == (1089,)
    assert np.isfinite(values).all()
    assert np.count_nonzero(values) > 100
    current = float(np.sum(values))
    registered = float(receipt["registered_seed_current_a"])
    print(f"ARM={arm} CURRENT_A={current:.12f} REGISTERED_A={registered:.12f}")
    assert current == pytest.approx(registered, rel=2.0e-6, abs=1.0e-9)
