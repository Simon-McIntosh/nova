import matplotlib.pylab
import numpy as np
import pytest

from nova.imas.database import Database
from nova.imas.ids_entry import IdsEntry
from nova.imas.pulsedesign import PulseDesign
from nova.imas.equilibrium import EquilibriumData
from nova.imas.sample import Sample
from nova.imas.test_utilities import ids_attrs, mark, mark_imas

biot_attrs = {
    "dplasma": -1,
    "nwall": None,
    "nlevelset": None,
    "ngrid": None,
    "nfield": None,
    "nforce": None,
    "tplasma": "h",
}


@pytest.fixture()
def ids():
    ids_entry = IdsEntry(name="equilibrium")
    time = [1.5, 19, 110, 600, 670]
    ids_entry.ids.time = time
    ids_entry.ids.time_slice.resize(len(time))
    ids_entry.ids.ids_properties.homogeneous_time = 1
    with ids_entry.node("time_slice:boundary_separatrix.*"):
        ids_entry["type", :] = [0, 1, 1, 1, 1]
        ids_entry["psi", :] = [107.8, 73.5, 17.4, -13.7, -7.5]
        ids_entry["minor_radius", :] = [1.7, 2.0, 2.0, 2.0, 1.9]
        ids_entry["elongation", :] = [1.1, 1.8, 1.8, 1.9, 1.1]
        ids_entry["elongation_upper", :] = [0.0, 0.2, 0.1, 0.1, 0.1]
        ids_entry["elongation_lower", :] = [0.0, 0.3, 0.2, 0.3, 0.3]
        ids_entry["triangularity_upper", :] = [0.0, 0.3, 0.4, 0.5, 0.3]
        ids_entry["triangularity_lower", :] = [0.1, 0.6, 0.5, 0.6, 0.6]
    with ids_entry.node("time_slice:boundary_separatrix.geometric_axis.*"):
        ids_entry["r", :] = [5.8, 6.2, 6.2, 6.2, 6.1]
        ids_entry["z", :] = [0.0, 0.1, 0.3, 0.3, -1.0]

    with ids_entry.node("time_slice:global_quantities.*"):
        ids_entry["ip", :] = 1e6 * np.array([-0.4, -5.1, -15, -15, -1.5])
    with ids_entry.node("time_slice:profiles_1d.*"):
        ids_entry["dpressure_dpsi", :] = 1e3 * np.array(
            [
                [0.2, 0.2, 0.2, 0.1, 0.1],
                [0.0, 0.7, 0.5, 0.4, 0.3],
                [0.4, 6.4, 5.7, 5.6, 5.7],
                [0.4, 7.2, 6.9, 6.5, 6.2],
                [0.0, 0.3, 0.3, 0.2, 0.1],
            ]
        )
        ids_entry["f_df_dpsi", :] = np.array(
            [
                [0.0, 0.1, 0.1, 0.1, 0.0],
                [1.4, 0.4, 0.3, 0.2, 0.2],
                [2.0, 1.5, 0.7, 0.3, 0.0],
                [2.0, 1.0, 0.6, 0.3, 0.2],
                [1.7, 0.6, 0.1, -0.0, -0.1],
            ]
        )
    return ids_entry.ids


@mark_imas
def test_ids_file_cache(ids):
    ids.time_slice[0].boundary_separatrix.psi = 66
    design_a = PulseDesign(ids=ids, **biot_attrs)
    design_a.time_index = 0

    ids.time_slice[0].boundary_separatrix.psi = 77
    design_b = PulseDesign(ids=ids, **biot_attrs)
    design_b.time_index = 0

    assert design_a["psi_boundary"] == 66
    assert design_b["psi_boundary"] == 77


@mark_imas
def test_pf_active_ids_input(ids):
    design = PulseDesign(ids=ids, **biot_attrs)
    pf_active_ids = design.geometry["pf_active"](**design.pf_active, lazy=False).ids
    design = PulseDesign(ids=ids, pf_active={"ids": pf_active_ids}, **biot_attrs)
    ids_entry = IdsEntry(name="pf_active")
    design.update_metadata(ids_entry)


@mark_imas
def test_pf_active_ids_input_cache(ids):
    pf_active_103 = Database(
        111001, 103, "pf_active", machine="iter_md", lazy=False
    ).ids
    pf_active_203 = Database(
        111001, 203, "pf_active", machine="iter_md", lazy=False
    ).ids
    design_103 = PulseDesign(ids=ids, pf_active={"ids": pf_active_103}, **biot_attrs)
    design_203 = PulseDesign(ids=ids, pf_active={"ids": pf_active_203}, **biot_attrs)
    assert design_103.group_attrs["pf_active"] != design_203.group_attrs["pf_active"]


@mark_imas
def test_make_frame(ids):
    design = PulseDesign(
        ids=ids, **dict(biot_attrs | {"nlevelset": 1e3, "nwall": 3, "dplasma": -1e3})
    )
    design.itime = 0
    design.add_animation("time", 10, ramp=100)
    with matplotlib.pylab.ioff():
        design.make_frame(20)


@mark["equilibrium_pds"]
def test_sample_pds():
    equilibrium = EquilibriumData(**ids_attrs["equilibrium_pds"])
    sample = Sample(equilibrium.data, epsilon=0.75, savgol=None)
    design = PulseDesign(ids=sample.equilibrium_ids(), **biot_attrs)
    assert design.data.sizes["time"] == 16


if __name__ == "__main__":
    pytest.main([__file__])
