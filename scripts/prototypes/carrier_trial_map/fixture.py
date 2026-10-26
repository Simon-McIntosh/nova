"""Load an explicitly named, validated carrier for a data-only ablation."""

import json
import os
from dataclasses import replace
from pathlib import Path

import xarray as xr

from scripts.analytic_oracle_fixtures import measure


def pinned_carrier():
    """Preserve the stored identity while selecting the requested fixture."""
    path = Path(os.environ["MEASURE_CARRIER"])
    with xr.open_dataset(path) as data:
        key = data.attrs["cache_key"]
        identity = json.loads(data.attrs["semantic_identity"])
        machine = measure._from_dataset(data, identity, key)
    machine = replace(
        machine,
        cache={
            "hit": True,
            "semantic_key": key,
            "store": str(path),
            "selection": "explicit data ablation",
        },
    )
    print("PINNED_CARRIER", str(path), key, flush=True)
    return machine
