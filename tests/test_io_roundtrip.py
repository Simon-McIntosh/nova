"""zarr signal-store round-trip: IDS -> store -> IDS -> validate(), no DB."""

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("imas_standard_names"):
    import imas_standard_names  # noqa: F401

    from nova.io.egress import IdsEgress
    from nova.io.ingest import IdsIngest, SignalSpec
    from nova.io.signalstore import SignalStore
    from nova.io.standardname import StandardNameResolver

# nova.imas.dataset guards the imas import; importing here is safe because
# imas-python is a core dependency.
from nova.imas.ids_entry import IdsEntry
from nova.imas.ids_index import IdsIndex

DD_VERSION = "3.42.0"

# Facility-signal -> standard-name map for a handful of equilibrium global
# quantities. In production codex discovers this map; the test supplies it
# explicitly. Names are resolved against ISN/ISNC, never minted here.
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
]

TIME = np.array([1.5, 19.0, 110.0, 600.0])
IP = 1e6 * np.array([-0.4, -5.1, -15.0, -1.5])
R_AXIS = np.array([5.8, 6.2, 6.2, 6.1])
MINOR_RADIUS = np.array([1.7, 2.0, 2.0, 1.9])
ELONGATION = np.array([1.1, 1.8, 1.9, 1.1])


def build_equilibrium():
    """Return a synthetic equilibrium IDS pinned to the target DD version."""
    entry = IdsEntry(name="equilibrium", dd_version=DD_VERSION)
    entry.ids.time = TIME.tolist()
    entry.ids.time_slice.resize(len(TIME))
    entry.ids.ids_properties.homogeneous_time = 1
    with entry.node("time_slice:global_quantities.*"):
        entry["ip", :] = IP
    with entry.node("time_slice:global_quantities.magnetic_axis.*"):
        entry["r", :] = R_AXIS
    with entry.node("time_slice:boundary_separatrix.*"):
        entry["minor_radius", :] = MINOR_RADIUS
        entry["elongation", :] = ELONGATION
    return entry.ids


@pytest.fixture
def store(tmp_path):
    """Return a signal store rooted in a tmp directory."""
    return SignalStore(
        filename="signals",
        dirname=str(tmp_path),
        parents=2,
        uri="test:synthetic/equilibrium",
        dd_version=DD_VERSION,
    )


def test_tensorize_keys_by_standard_name():
    ingest = IdsIngest(resolver=StandardNameResolver())
    dataset = ingest.tensorize(
        build_equilibrium(),
        SPECS,
        uri="test:synthetic/equilibrium",
        dd_version=DD_VERSION,
    )
    assert "plasma_current" in dataset
    assert np.allclose(dataset["plasma_current"].values, IP)
    assert dataset["plasma_current"].dims == ("time",)
    assert dataset.attrs["ids_name"] == "equilibrium"
    assert dataset.attrs["cocos"] == 11
    # plasma_current is a catalog entry, so it carries an authoritative unit.
    assert dataset["plasma_current"].attrs.get("units")


def test_ingest_is_cache_keyed(store):
    ingest = IdsIngest(resolver=StandardNameResolver())
    assert not store.cached()
    ingest.ingest(build_equilibrium(), SPECS, store)
    assert store.cached()
    reread = store.read()
    assert np.allclose(reread["plasma_current"].values, IP)
    assert np.allclose(reread["time"].values, TIME)


def test_roundtrip_zarr_to_ids_validate(store, tmp_path):
    resolver = StandardNameResolver()
    IdsIngest(resolver=resolver).ingest(build_equilibrium(), SPECS, store)
    dataset = store.read()

    uri = f"imas:hdf5?path={tmp_path / 'egress'}"
    written = IdsEgress().egress(dataset, SPECS, uri)

    # Re-read the egressed pulse with the Access Layer and confirm values.
    import imas

    database = imas.DBEntry(written, "r", dd_version=DD_VERSION)
    try:
        equilibrium = database.get("equilibrium")
    finally:
        database.close()
    assert equilibrium.ids_properties.homogeneous_time == 1
    index = IdsIndex(equilibrium, "time_slice")
    assert np.allclose(index.array("global_quantities.ip"), IP)
    assert np.allclose(index.array("boundary_separatrix.minor_radius"), MINOR_RADIUS)
    assert np.allclose(index.array("global_quantities.magnetic_axis.r"), R_AXIS)
