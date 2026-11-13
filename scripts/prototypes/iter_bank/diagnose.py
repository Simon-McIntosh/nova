"""Localize nodal operator disagreement using an already-published bank."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import time

from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_points, support_read, topology
from scripts.prototypes.iter_bank.measure import (
    direct_map,
    emit,
    read_scenario,
    sampled_moments,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert jax.config.jax_enable_x64
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    receipt = json.loads((args.bank / "build-receipt.json").read_text())
    wall, polygons, turns, owners, names, states = read_scenario()
    with np.load(args.bank / "geometry.npz", allow_pickle=False) as archive:
        cells = tuple(
            p[:n]
            for p, n in zip(archive["vertices"], archive["vertex_count"], strict=True)
        )
        geometry = topology.TopologyGeometry.from_cells(
            cells, archive["sample_points"][:, 1:], wall
        )
    destination = (
        Path(receipt["bank"]["store"])
        / "sha256"
        / receipt["bank"]["digest"].split(":")[1]
    )
    # Resolve the existing content-store layout through its verified owner.
    from nova.database import content_store

    root = content_store.verified_object_root(receipt["bank"]["store"], create=False)
    destination = content_store.verified_destination(
        root, content_store.digest_hex(receipt["bank"]["digest"])
    )
    with np.load(destination / "bank.npz", allow_pickle=False) as archive:
        nodes = archive["nodes"]
        sample_index = archive["sample_index"]
    mesh = fixed_points.build_mesh(
        nodes[: receipt["bank"]["active_nodes"]], sample_index, geometry
    )
    started = time.perf_counter()
    bank = support_read.read_bank(
        receipt["bank"]["store"], receipt["bank"]["digest"], geometry, mesh
    )
    jax.block_until_ready(bank)
    emit(
        "RELOAD",
        seconds=time.perf_counter() - started,
        device=jax.devices()[0].device_kind,
    )
    points = np.asarray(bank.nodes)[: int(bank.active_nodes)]
    plasma = topology.BiotMomentCoupling.from_polygons(cells)
    coils = topology.BiotMomentCoupling.from_polygons(polygons)
    records = []
    for state in states:
        moment = sampled_moments(geometry, state[2])
        coil_moment = np.zeros((len(polygons), 3))
        coil_moment[:, 0] = turns * state[3][owners]
        expected = direct_map(points, plasma, moment, coils, coil_moment)
        plasma_actual = np.asarray(
            support_read.nodal_jets(bank, tuple(jnp.asarray(moment.T)))
        )[: len(points)]
        coil_actual = np.einsum(
            "nkq,k->nq", np.asarray(bank.coil_exterior)[: len(points)], state[3]
        )
        actual = plasma_actual + coil_actual
        dropped = (
            actual
            - np.asarray(bank.coil_exterior)[: len(points), names.index("PF1")]
            * state[3][names.index("PF1")]
        )
        np.savez(
            args.output / f"accuracy-{state[0]}.npz",
            actual=actual,
            expected=expected,
            dropped=dropped,
            moments=moment,
            coil_currents=state[3],
            points=points,
        )
        diff = actual - expected
        rank = np.argsort(np.abs(diff[:, 5]))[-10:][::-1]
        item = dict(
            index=state[0],
            relative_errors=(
                np.linalg.norm(diff, axis=0) / np.linalg.norm(expected, axis=0)
            ).tolist(),
            overall_relative=float(np.linalg.norm(diff) / np.linalg.norm(expected)),
            maximum_absolute=np.max(np.abs(diff), axis=0).tolist(),
            worst_nodes=[
                dict(
                    index=int(n),
                    point=points[n].tolist(),
                    actual=actual[n].tolist(),
                    expected=expected[n].tolist(),
                )
                for n in rank
            ],
        )
        if not records:
            plasma_expected = direct_map(
                points, plasma, moment, coils, np.zeros_like(coil_moment)
            )
            coil_expected = direct_map(
                points, plasma, np.zeros_like(moment), coils, coil_moment
            )
            item["plasma_relative"] = (
                np.linalg.norm(plasma_actual - plasma_expected, axis=0)
                / np.linalg.norm(plasma_expected, axis=0)
            ).tolist()
            item["coils_relative"] = (
                np.linalg.norm(coil_actual - coil_expected, axis=0)
                / np.linalg.norm(coil_expected, axis=0)
            ).tolist()
            np.savez(
                args.output / "components.npz",
                plasma_actual=plasma_actual,
                plasma_expected=plasma_expected,
                coil_actual=coil_actual,
                coil_expected=coil_expected,
            )
        records.append(item)
        emit("DIAGNOSIS", **item)
        (args.output / "diagnosis.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
