import tempfile

import numpy as np
import pytest

from nova.imas.database import IdsEntry
from nova.imas.pulsedesign import PulseDesign


@pytest.fixture
def ids():
    ids_entry = IdsEntry(name='equilibrium')
    time = [1.5, 19, 110, 600, 670]
    ids_entry.ids_data.time = time
    ids_entry.ids_data.time_slice.resize(len(time))
    ids_entry.ids_data.ids_properties.homogeneous_time = 1
    with ids_entry.node('time_slice:boundary_separatrix.*'):
        ids_entry['type', :] = [0, 1, 1, 1, 1]
        ids_entry['psi', :] = [107.8, 73.5, 17.4, -13.7, -7.5]
        ids_entry['minor_radius', :] = [1.7, 2., 2., 2., 1.9]
        ids_entry['elongation', :] = [1.1, 1.8, 1.8, 1.9, 1.1]
        ids_entry['elongation_upper', :] = [0., 0.2, 0.1, 0.1, 0.1]
        ids_entry['elongation_lower', :] = [0., 0.3, 0.2, 0.3, 0.3]
        ids_entry['triangularity_upper', :] = [0., 0.3, 0.4, 0.5, 0.3]
        ids_entry['triangularity_lower', :] = [0.1, 0.6, 0.5, 0.6, 0.6]
    with ids_entry.node('time_slice:boundary_separatrix.geometric_axis.*'):
        ids_entry['r', :] = [5.8, 6.2, 6.2, 6.2, 6.1]
        ids_entry['z', :] = [0., 0.1, 0.3, 0.3, -1.]

    with ids_entry.node('time_slice:global_quantities.*'):
        ids_entry['ip', :] = 1e6 * np.array([-0.4, -5.1, -15, -15, -1.5])
    with ids_entry.node('time_slice:profiles_1d.*'):
        ids_entry['dpressure_dpsi', :] = 1e3 * np.array(
            [[0.2, 0.2, 0.2, 0.1, 0.1],
             [0., 0.7, 0.5, 0.4, 0.3],
             [0.4, 6.4, 5.7, 5.6, 5.7],
             [0.4, 7.2, 6.9, 6.5, 6.2],
             [0., 0.3, 0.3, 0.2, 0.1]])
        ids_entry['f_df_dpsi', :] = np.array(
            [[0., 0.1, 0.1, 0.1, 0.],
             [1.4, 0.4, 0.3, 0.2, 0.2],
             [2., 1.5, 0.7, 0.3, 0.],
             [2., 1., 0.6, 0.3, 0.2],
             [1.7, 0.6, 0.1, -0., -0.1]])
    return ids_entry.ids_data


def test_ids_file_cache(ids):
    ids.time_slice[0].boundary_separatrix.psi = 66
    design_a = PulseDesign(ids=ids, dplasma=-1, nwall=None, nlevelset=None)
    ids.time_slice[0].boundary_separatrix.psi = 77
    design_b = PulseDesign(ids=ids, dplasma=-1, nwall=None, nlevelset=None)

    assert False



if __name__ == '__main__':

    pytest.main([__file__])
