"""Signal-store ingest benchmark: warm zarr re-read vs the AL full-get.

The store's value proposition is that a re-read of an ingested pulse serves
dense, standard-name-keyed signals far faster than re-reading the pulse through
the Access Layer. This benchmark writes a synthetic equilibrium once, then
compares:

* baseline -- open the pulse through imas-python, ``get()`` the whole IDS, and
  tensorize the tracked signals (what a consumer pays without the store);
* warm re-read -- load the same signals from the zarr signal store.

Run:  python -m benchmarks.io_ingest

Requires the ``io`` extra (imas-standard-names / imas-standard-names-catalog)
and imas-python.
"""

from __future__ import annotations

import tempfile
import timeit

import numpy as np

import imas

from nova.imas.ids_entry import IdsEntry
from nova.imas.ids_index import IdsIndex
from nova.io.ingest import IdsIngest, SignalSpec
from nova.io.signalstore import SignalStore
from nova.io.standardname import StandardNameResolver

DD_VERSION = "3.42.0"
N_TIME = 500
N_RHO = 101

SPECS = [
    SignalSpec("plasma_current", "time_slice", "global_quantities.ip"),
    SignalSpec("magnetic_axis", "time_slice", "global_quantities.magnetic_axis.r"),
    SignalSpec(
        "minor_radius_of_plasma_boundary",
        "time_slice",
        "boundary_separatrix.minor_radius",
    ),
    SignalSpec(
        "elongation_of_plasma_boundary",
        "time_slice",
        "boundary_separatrix.elongation",
    ),
    SignalSpec("safety_factor", "time_slice", "profiles_1d.q"),
    SignalSpec("electron_pressure", "time_slice", "profiles_1d.pressure"),
]


def build_pulse(directory: str) -> str:
    """Write a synthetic equilibrium pulse to HDF5 and return its URI."""
    entry = IdsEntry(name="equilibrium", dd_version=DD_VERSION)
    time = np.linspace(0.0, 100.0, N_TIME)
    entry.ids.time = time.tolist()
    entry.ids.time_slice.resize(N_TIME)
    entry.ids.ids_properties.homogeneous_time = 1
    rng = np.random.default_rng(0)
    with entry.node("time_slice:global_quantities.*"):
        entry["ip", :] = -15e6 * np.sin(time / 100.0 * np.pi)
    with entry.node("time_slice:global_quantities.magnetic_axis.*"):
        entry["r", :] = 6.2 + 0.1 * np.cos(time / 20.0)
    with entry.node("time_slice:boundary_separatrix.*"):
        entry["minor_radius", :] = 2.0 + 0.05 * np.cos(time / 30.0)
        entry["elongation", :] = 1.8 + 0.1 * np.sin(time / 40.0)
    psi = np.linspace(0.0, 1.0, N_RHO)
    with entry.node("time_slice:profiles_1d.*"):
        entry["psi", :] = np.tile(psi, (N_TIME, 1))
        entry["q", :] = 1.0 + 3.0 * rng.random((N_TIME, N_RHO))
        entry["pressure", :] = 1e5 * rng.random((N_TIME, N_RHO))
    uri = f"imas:hdf5?path={directory}"
    database = imas.DBEntry(uri, "w", dd_version=DD_VERSION)
    try:
        database.put(entry.ids)
    finally:
        database.close()
    return uri


def al_full_get(uri: str) -> None:
    """Baseline: AL full get() + tensorize the tracked signals."""
    database = imas.DBEntry(uri, "r", dd_version=DD_VERSION)
    try:
        ids = database.get("equilibrium")
        for spec in SPECS:
            IdsIndex(ids, spec.node).array(spec.path)
    finally:
        database.close()


def main(repeats: int = 5) -> float:
    """Run the benchmark and return the warm-reread speedup factor."""
    with (
        tempfile.TemporaryDirectory() as pulse_dir,
        tempfile.TemporaryDirectory() as store_dir,
    ):
        uri = build_pulse(pulse_dir)

        # Warm the DD cache so the baseline measures data access, not parsing.
        al_full_get(uri)
        baseline = min(
            timeit.repeat(lambda: al_full_get(uri), number=1, repeat=repeats)
        )

        store = SignalStore(
            filename="signals",
            dirname=store_dir,
            parents=2,
            uri=uri,
            dd_version=DD_VERSION,
        )
        ingest = IdsIngest(resolver=StandardNameResolver())
        database = imas.DBEntry(uri, "r", dd_version=DD_VERSION)
        try:
            ingest.ingest(database.get("equilibrium"), SPECS, store)
        finally:
            database.close()

        store.read()  # warm the store handle
        warm = min(timeit.repeat(lambda: store.read(), number=1, repeat=repeats))

    speedup = baseline / warm
    print(f"AL full-get baseline : {baseline * 1e3:8.2f} ms")
    print(f"warm zarr re-read    : {warm * 1e3:8.2f} ms")
    print(f"speedup              : {speedup:8.1f}x  (target >= 100x)")
    return speedup


if __name__ == "__main__":
    main()
