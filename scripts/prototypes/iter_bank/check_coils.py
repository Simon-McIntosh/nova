"""Check driven-coil jets independently of the plasma derivative program."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path

from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.database import content_store
from nova.equilibrium import topology
from scripts.prototypes.iter_bank.measure import assert_accuracy, emit, pack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert jax.config.jax_enable_x64
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    receipt = json.loads((args.bank / "build-receipt.json").read_text())
    source = receipt["source"]
    with np.load(args.bank / "geometry.npz", allow_pickle=False) as archive:
        turns = archive["turns"]
        owners = archive["owners"]
        polygons = tuple(archive[f"coil_{i}"] for i in range(len(turns)))
    root = content_store.verified_object_root(receipt["bank"]["store"], create=False)
    dest = content_store.verified_destination(
        root, content_store.digest_hex(receipt["bank"]["digest"])
    )
    with np.load(dest / "bank.npz", allow_pickle=False) as archive:
        active = int(archive["active_nodes"])
        points = archive["nodes"][:active]
        rows = archive["coil_exterior"][:active]
    coupling = topology.BiotMomentCoupling.from_polygons(polygons)

    def direct(point, coupling, moment):
        return pack(coupling.evaluate(point, tuple(moment.T)))

    program = jax.jit(jax.vmap(direct, in_axes=(0, None, None)))
    records = []
    for state in source["states"]:
        currents = np.asarray(state["coil_currents_A"])
        moment = np.zeros((len(turns), 3))
        moment[:, 0] = turns * currents[owners]
        expected = []
        for first in range(0, len(points), 4):
            p = points[first : first + 4]
            n = len(p)
            p = np.pad(p, ((0, 4 - n), (0, 0)), mode="edge")
            expected.append(
                np.asarray(program(jnp.asarray(p), coupling, jnp.asarray(moment)))[:n]
            )
        expected = np.concatenate(expected)
        actual = np.einsum("nkq,k->nq", rows, currents)
        dropped = (
            actual
            - rows[:, source["coil_names"].index("PF1")]
            * currents[source["coil_names"].index("PF1")]
        )
        np.savez(
            args.output / f"coil-accuracy-{state['index']}.npz",
            expected=expected,
            actual=actual,
            dropped=dropped,
        )
        errors = (
            np.linalg.norm(actual - expected, axis=0) / np.linalg.norm(expected, axis=0)
        ).tolist()
        item = dict(index=state["index"], relative_errors=errors)
        records.append(item)
        emit("COIL_ACCURACY", **item)
        (args.output / "receipt.json").write_text(
            json.dumps(dict(kind="coil-only", accuracy=records), indent=2)
        )
        assert_accuracy(actual, expected)
        try:
            assert_accuracy(dropped, expected)
        except AssertionError as error:
            emit("COIL_NEGATIVE_CONTROL", index=state["index"], refusal=str(error))
        else:
            raise AssertionError("dropped PF1 was not refused")


if __name__ == "__main__":
    main()
