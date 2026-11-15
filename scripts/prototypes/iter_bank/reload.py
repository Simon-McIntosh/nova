"""Reload a published operator bank and verify its persisted dense operands."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import time

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from nova.database import content_store
from nova.equilibrium import fixed_points, support_read, topology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    receipt = json.loads((args.bank / "build-receipt.json").read_text())["bank"]
    with np.load(args.bank / "geometry.npz", allow_pickle=False) as archive:
        cells = tuple(
            p[:n]
            for p, n in zip(archive["vertices"], archive["vertex_count"], strict=True)
        )
        geometry = topology.TopologyGeometry.from_cells(
            cells, archive["sample_points"][:, 1:], (archive["wall"],)
        )
    root = content_store.verified_object_root(receipt["store"], create=False)
    destination = content_store.verified_destination(
        root, content_store.digest_hex(receipt["digest"])
    )
    with np.load(destination / "bank.npz", allow_pickle=False) as archive:
        mesh = fixed_points.build_mesh(
            archive["nodes"][: receipt["active_nodes"]],
            archive["sample_index"],
            geometry,
        )
        started = time.perf_counter()
        bank = support_read.read_bank(
            receipt["store"], receipt["digest"], geometry, mesh
        )
        jax.block_until_ready(bank)
        seconds = time.perf_counter() - started
        verified = []
        for name in (
            "rows",
            "exterior",
            "coil_exterior",
            "coil_count",
            "nodes",
            "sample_index",
            "bucket",
            "active_nodes",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(bank, name)), archive[name]
            )
            verified.append(name)
    assert bank.coil_exterior is not None and int(bank.coil_count) == 12
    result = dict(
        digest=receipt["digest"],
        backend=jax.default_backend(),
        device=jax.devices()[0].device_kind,
        seconds=seconds,
        bucket=bank.bucket,
        active_nodes=int(bank.active_nodes),
        coil_count=int(bank.coil_count),
        verified_arrays=verified,
        module=support_read.__file__,
        x64=jax.config.jax_enable_x64,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("RELOAD " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
