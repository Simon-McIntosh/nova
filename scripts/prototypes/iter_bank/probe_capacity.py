"""Measure device storage admission for a published ITER operator bank."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from nova.database import content_store
from nova.equilibrium import fixed_points, support_read, topology
from scripts.prototypes.iter_bank.measure import capacity, emit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    receipt = json.loads((args.bank / "build-receipt.json").read_text())
    with np.load(args.bank / "geometry.npz", allow_pickle=False) as archive:
        cells = tuple(
            p[:n]
            for p, n in zip(archive["vertices"], archive["vertex_count"], strict=True)
        )
        geometry = topology.TopologyGeometry.from_cells(
            cells, archive["sample_points"][:, 1:], (archive["wall"],)
        )
    store = receipt["bank"]["store"]
    digest = receipt["bank"]["digest"]
    root = content_store.verified_object_root(store, create=False)
    destination = content_store.verified_destination(
        root, content_store.digest_hex(digest)
    )
    with np.load(destination / "bank.npz", allow_pickle=False) as archive:
        nodes = archive["nodes"]
        sample_index = archive["sample_index"]
    mesh = fixed_points.build_mesh(
        nodes[: receipt["bank"]["active_nodes"]], sample_index, geometry
    )
    bank = support_read.read_bank(store, digest, geometry, mesh)
    jax.block_until_ready(bank)
    emit(
        "DEVICE",
        kind=jax.devices()[0].device_kind,
        bucket=bank.bucket,
        dense_bytes=receipt["bank"]["dense_bytes"],
    )
    result = capacity(bank)
    result["device"] = jax.devices()[0].device_kind
    result["bucket"] = bank.bucket
    result["dense_bytes"] = receipt["bank"]["dense_bytes"]
    args.output.write_text(json.dumps(result, indent=2))
    emit("CAPACITY", **{k: v for k, v in result.items() if k != "trials"})


if __name__ == "__main__":
    main()
