"""Refresh lower-root diagnostics on saved states without building a forward map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=Path, required=True)
    args = parser.parse_args()
    import jax
    import numpy as np

    from scripts.prototypes.diiid_diverted_branch.measure import lower_saddle_reader
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64
    record = json.loads((args.row / "receipt.json").read_text())
    with np.load(args.row / "terminal.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    state_hash = hashlib.sha256(data["state"].tobytes()).hexdigest()
    reference = np.asarray(record["efit_x_point_rz_m"])
    terminal_trip = record["inner"][-1]["index"]
    with np.load(args.row / f"inner-{terminal_trip}.npz") as archive:
        normalized = archive["state"][: data["state"].size]
        scale = np.dot(data["state"], normalized) / np.dot(normalized, normalized)

    read_lower = lower_saddle_reader(data, reference)
    observations = []

    def refresh(topology, state):
        # Older receipts retain only the terminal axis; record that fallback.
        axis = topology.get("axis_rz_m", data["axis"])
        lower = read_lower(state, topology["selected_saddle_rz_m"], axis)
        print("LOWER_ROOT_PROBE " + json.dumps(lower), flush=True)
        assert lower["lower_saddle_qualified"], (
            "known lower saddle was not recovered by either probe"
        )
        topology.update(lower)
        observations.append(
            dict(
                distance_m=lower["lower_saddle_distance_m"],
                probes=lower["lower_saddle_probe_qualified"],
                axis_source="trip" if "axis_rz_m" in topology else "terminal fallback",
            )
        )

    for trip in record["trip_topology"]:
        with np.load(args.row / f"inner-{trip['trip']}.npz") as archive:
            refresh(trip["initial"], archive["initial"] * scale)
            refresh(trip["returned"], archive["state"] * scale)
    refresh(record["terminal_topology"], data["state"])
    assert observations[0]["distance_m"] < 1e-12
    record["recovery"]["lower_root_diagnostic_refresh"] = observations
    data["continued_saddle"] = np.r_[
        record["terminal_topology"]["lower_saddle_rz_m"],
        record["terminal_topology"]["lower_saddle_flux_wb"],
    ]
    assert hashlib.sha256(data["state"].tobytes()).hexdigest() == state_hash
    np.savez_compressed(args.row / "terminal.npz", **data)
    (args.row / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.row / "trips.json").write_text(
        json.dumps(record["trip_topology"], indent=2) + "\n"
    )
    print("REFRESHED_LOWER_ROOTS " + json.dumps(observations), flush=True)
    print("FORWARD_SOLVES_EXECUTED=0; STATE_SHA256=" + state_hash, flush=True)


if __name__ == "__main__":
    main()
