"""Retain a verified ITER bank and consecutive native flattop inputs."""

# ruff: noqa: E402
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from nova.jax.config import configure_dtypes

configure_dtypes()

import imas
import jax
import jax.numpy as jnp
import numpy as np

from nova.database import content_store
from nova.equilibrium import fixed_points, support_read, topology
from scripts.prototypes.iter_bank.measure import sampled_moments

SOURCE = "/work/imas/shared/imasdb/ITER/3/105027/1"
DD = "3.38.1"


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: np.asarray(x).tolist()) + "\n"
    )


def load_bank(directory):
    record = json.loads((directory / "build-receipt.json").read_text())["bank"]
    store = directory / "store"
    with np.load(directory / "geometry.npz", allow_pickle=False) as archive:
        cells = tuple(
            p[:n]
            for p, n in zip(archive["vertices"], archive["vertex_count"], strict=True)
        )
        geometry = topology.TopologyGeometry.from_cells(
            cells, archive["sample_points"][:, 1:], (archive["wall"],)
        )
    root = content_store.verified_object_root(store, create=False)
    destination = content_store.verified_destination(
        root, content_store.digest_hex(record["digest"])
    )
    with np.load(destination / "bank.npz", allow_pickle=False) as archive:
        mesh = fixed_points.build_mesh(
            archive["nodes"][: record["active_nodes"]],
            archive["sample_index"],
            geometry,
        )
        bank = support_read.read_bank(store, record["digest"], geometry, mesh)
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
    cache = fixed_points.RowCache(
        bank.mesh, bank.nodes, jnp.arange(len(bank.nodes)), bank.rows, bank.exterior
    )
    assert bank.coil_exterior is not None and int(bank.coil_count) == 12
    return bank, cache


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    args.output.mkdir(parents=True, exist_ok=True)
    retained = args.output / "bank"
    retained.mkdir(exist_ok=True)
    for name in ("geometry.npz", "source.json", "build-receipt.json"):
        shutil.copy2(args.bank / name, retained / name)
    shutil.copytree(args.bank / "store", retained / "store", dirs_exist_ok=True)
    bank, _ = load_bank(retained)
    with imas.DBEntry(f"imas:hdf5?path={SOURCE}", "r", dd_version=DD) as entry:
        eq = entry.get("equilibrium", 0, autoconvert=False, lazy=False)
        pf = entry.get("pf_active", 0, autoconvert=False, lazy=False)
    assert int(eq.ids_properties.homogeneous_time) == 1
    assert str(eq.ids_properties.version_put.data_dictionary) == DD
    assert str(pf.ids_properties.version_put.data_dictionary) == DD
    names = [str(pf.coil[i].name) for i in range(len(pf.coil))]
    bank_source = json.loads((retained / "source.json").read_text())
    assert names == bank_source["coil_names"]
    times = np.asarray(eq.time)
    ip = np.asarray(
        [float(eq.time_slice[i].global_quantities.ip) for i in range(len(times))]
    )
    indices = np.arange(129, 149)
    assert np.all(np.diff(times[indices]) > 0)
    assert np.all((times[indices] >= 30) & (times[indices] <= 100))
    assert np.all(np.abs(ip[indices]) >= 0.95 * np.max(np.abs(ip)))
    records = []
    for index in indices:
        s = eq.time_slice[int(index)]
        q, p, field = s.global_quantities, s.profiles_1d, s.profiles_2d[0]
        psi = np.asarray(p.psi)
        normalized = (psi - psi[0]) / (psi[-1] - psi[0])
        assert np.all(np.diff(normalized) > 0)
        knots = np.linspace(0, 1, 33)
        pressure = np.interp(knots, normalized, np.asarray(p.dpressure_dpsi))
        ff = np.interp(knots, normalized, np.asarray(p.f_df_dpsi))
        currents = []
        for coil in pf.coil:
            base = (
                np.asarray(pf.time)
                if int(pf.ids_properties.homogeneous_time) == 1
                else np.asarray(coil.current.time)
            )
            currents.append(
                np.interp(times[index], base, np.asarray(coil.current.data))
            )
        moments = sampled_moments(bank.geometry, s)
        centroid = (
            np.sum(
                moments[:, 0] * np.asarray(bank.geometry.centre)[:, 1] + moments[:, 2]
            )
            / moments[:, 0].sum()
        )
        axis = np.array([float(q.magnetic_axis.r), float(q.magnetic_axis.z)])
        nulls = s.boundary_separatrix.x_point
        xpoints = np.array(
            [[float(nulls[i].r), float(nulls[i].z)] for i in range(len(nulls))]
        ).reshape(-1, 2)
        assert len(xpoints) > 0
        row = dict(
            index=int(index),
            time_s=float(times[index]),
            ip_a=float(ip[index]),
            centroid_m=float(centroid),
            axis=axis,
            xpoints=xpoints,
            axis_flux_wb=-float(q.psi_axis),
            boundary_flux_wb=-float(q.psi_boundary),
            integrated_ip_a=float(moments[:, 0].sum()),
        )
        np.savez_compressed(
            args.output / f"source-{index}.npz",
            moments=moments,
            pressure=pressure,
            ff=ff,
            currents=currents,
            axis=axis,
            xpoints=xpoints,
            radius=np.asarray(field.grid.dim1),
            height=np.asarray(field.grid.dim2),
            flux=-np.asarray(field.psi),
            centroid=centroid,
        )
        records.append(row)
        print(
            "SOURCE_SLICE",
            json.dumps(row, default=lambda x: np.asarray(x).tolist()),
            flush=True,
        )
    write(
        args.output / "selection.json",
        dict(
            source=SOURCE,
            dd=DD,
            slices=records,
            coil_names=names,
            rule="twenty consecutive stored slices from the established 30 s flattop onset; abs(Ip) >= 95% peak",
            bank_digest=json.loads((retained / "build-receipt.json").read_text())[
                "bank"
            ]["digest"],
            bank_origin=str(args.bank),
            bank_sha256=hashlib.sha256(
                (retained / "geometry.npz").read_bytes()
            ).hexdigest(),
            flux_convention="Nova total flux is negative of stored DD3 flux; raw gauges retained, no offset fitted",
        ),
    )
    print("INPUTS_VERIFIED consecutive=20 bank_arrays=8 coils=12", flush=True)


if __name__ == "__main__":
    main()
