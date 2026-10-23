"""The signal store keys its cache on the COCOS convention of the stored fields.

A store's group name follows the source identity *and* the convention its fields
are in, so two reads of one source in different conventions never share a group
and an entry written before the convention entered the key cannot be served as
if it were the convention-keyed one.
"""

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("imas_standard_names"):
    import imas_standard_names  # noqa: F401

    from nova.io.ingest import IdsIngest, SignalSpec
    from nova.io.signalstore import SignalStore, StaleConventionError
    from nova.io.standardname import StandardNameResolver

from nova.imas.ids_entry import IdsEntry

DD3 = "3.42.0"
DD4 = "4.1.0"

SPECS = [
    SignalSpec("plasma_current", "time_slice", "global_quantities.ip"),
    SignalSpec("magnetic_axis", "time_slice", "global_quantities.magnetic_axis.r"),
]

TIME = np.array([1.5, 19.0, 110.0, 600.0])
IP = 1e6 * np.array([-0.4, -5.1, -15.0, -1.5])
R_AXIS = np.array([5.8, 6.2, 6.2, 6.1])


def build_equilibrium(dd_version):
    """Return a synthetic equilibrium IDS written at the given DD version."""
    entry = IdsEntry(name="equilibrium", dd_version=dd_version)
    entry.ids.time = TIME.tolist()
    entry.ids.time_slice.resize(len(TIME))
    entry.ids.ids_properties.homogeneous_time = 1
    with entry.node("time_slice:global_quantities.*"):
        entry["ip", :] = IP
    with entry.node("time_slice:global_quantities.magnetic_axis.*"):
        entry["r", :] = R_AXIS
    return entry.ids


def make_store(tmp_path, dd_version, **kwargs):
    """Return a signal store for one source identity and DD version."""
    return SignalStore(
        filename="signals",
        dirname=str(tmp_path),
        parents=2,
        uri="test:synthetic/equilibrium",
        dd_version=dd_version,
        **kwargs,
    )


def tensorize(tmp_path, dd_version):
    """Tensorize a synthetic equilibrium read for a store identity."""
    ingest = IdsIngest(resolver=StandardNameResolver())
    return ingest.tensorize(
        build_equilibrium(dd_version),
        SPECS,
        uri="test:synthetic/equilibrium",
        dd_version=dd_version,
    )


def test_dd3_and_dd4_stores_key_and_declare_their_convention(tmp_path):
    store3 = make_store(tmp_path, DD3)
    store4 = make_store(tmp_path, DD4)

    assert store3.cocos == 11
    assert store4.cocos == 17
    assert store3.cache_key != store4.cache_key

    dataset3 = tensorize(tmp_path, DD3)
    dataset4 = tensorize(tmp_path, DD4)
    assert dataset3.attrs["cocos"] == 11
    assert dataset4.attrs["cocos"] == 17


def test_convention_alone_changes_the_key(tmp_path):
    store3 = make_store(tmp_path, DD3)
    rekeyed = make_store(tmp_path, DD3, cocos=17)

    assert store3.dd_version == rekeyed.dd_version
    assert store3.cocos != rekeyed.cocos
    assert store3.cache_key != rekeyed.cache_key


def test_entry_written_under_the_old_key_is_refused(tmp_path):
    dataset = tensorize(tmp_path, DD3)

    # An older nova wrote the group under a key carrying no convention.
    stale = make_store(tmp_path, DD3)
    stale.group = stale.legacy_cache_key
    stale.write(dataset)
    assert stale.legacy_cache_key != stale.cache_key

    fresh = make_store(tmp_path, DD3)
    with pytest.raises(StaleConventionError):
        fresh.read()
    assert not fresh.cached()

    # Re-ingesting writes the convention-keyed group, which is then served.
    IdsIngest(resolver=StandardNameResolver()).ingest(
        build_equilibrium(DD3), SPECS, fresh
    )
    assert fresh.cached()
    assert np.allclose(fresh.read()["plasma_current"].values, IP)
