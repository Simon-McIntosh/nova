"""Refuse missing, inconsistent or empty captured support evidence."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.topology import Topology


def require_support(record, arrays):
    """Require a positive, consistent observation at every support stage."""
    material = arrays["inside_material"].astype(bool)
    closed = arrays["closed"].astype(bool)
    connected = arrays["connected"].astype(bool)
    actual = {
        "material": int(material.sum()),
        "closed_flux_test": int(closed.sum()),
        "material_and_closed": int((material & closed).sum()),
        "axis_connected": int(connected.sum()),
    }
    assert actual == record["counts"], (actual, record["counts"])
    assert all(actual.values()), f"empty stage: {actual}"
    assert np.all(~connected | (material & closed)), "flood escaped its candidates"
    assert record["axis_polish_shift_m"] < record["lattice_spacing_m"]
    assert record["axis_cell_material"] == bool(material[record["axis_cell"]])
    assert record["axis_cell_closed"] == bool(closed[record["axis_cell"]])
    assert record["state_node_count"] > record["lattice_node_count"]
    assert arrays["flux"].shape == (arrays["radius"].size, arrays["height"].size)
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--force-empty", action="store_true")
    args = parser.parse_args()
    print(f"measurement_module={inspect.getfile(Topology)}", flush=True)
    print(f"measurement_cwd={Path.cwd().resolve()}", flush=True)
    source = inspect.getsource(ForwardProfile.solve)
    assert "history.active_set_mask_differences" in source
    print(
        "PROVENANCE: receipt mask_history reads active_set_mask_differences", flush=True
    )
    for frame, branch, trips in (
        (179, "limited", 1),
        (146, "diverted", 3),
        (137, "diverted", 1),
    ):
        stem = f"frame-{frame}-{branch}-trip-{trips}"
        record = json.loads((args.input / f"{stem}.json").read_text())
        with np.load(args.input / f"{stem}.npz") as data:
            arrays = {key: data[key] for key in data.files}
        if args.force_empty:
            arrays["connected"] = np.zeros_like(arrays["connected"])
            record["counts"]["axis_connected"] = 0
        counts = require_support(record, arrays)
        print(f"{stem}: {counts}; axis-cell comparison passes", flush=True)
    print("CAPTURE CHECK OK: three nonempty production support reads", flush=True)


if __name__ == "__main__":
    main()
