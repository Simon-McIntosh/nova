"""Manage access to dynamic coil data data."""
from dataclasses import dataclass

import numpy as np

from nova.frame.baseplot import Plot
from nova.imas.scenario import Scenario


@dataclass
class PulseSchedule(Plot, Scenario):
    """Manage access to pf_active ids."""

    name: str = 'pulse_schedule'
    ids_node: str = ''

    def time_coordinate(self, path: str, attr: str):
        """Return time coordinate."""
        if self.data.attrs['homogeneous_time'] == 1:
            return ('time',)
        coord = f'{attr}_time'
        self.data.coords[coord] = self.ids_index.get(path + '.time')
        return (coord,)

    def build_0d(self, ids_node: str, attrs: list[str]):
        """Extract position control attributes from ids."""
        with self.ids_index.node(ids_node):
            for attr in attrs:
                path = self.ids_index.get_path('*.reference', attr)
                if len(data := self.ids_index.get(path + '.data')) == 0:
                    continue
                self.data[attr] = self.time_coordinate(path, attr), data

    def build_point(self, ids_node: str):
        """Extract rz points."""
        name = ids_node.split('.')[-1]
        with self.ids_index.node(ids_node):
            if self.ids_index.empty('r.reference.data'):
                return
            time = self.time_coordinate('r.reference', name)
            shape = self.ids_index.shape('r.reference.data')[::-1] + (2,)
            data = np.zeros(shape, dtype=float)
            for i, attr in enumerate(['r', 'z']):
                data[..., i] = self.ids_index.array(f'{attr}.reference.data')
            self.data.coords['point'] = ['r', 'z']
            self.data[name] = time + (f'{name}_index', 'point'), data

    def build_points(self, ids_node: str, attrs: list[str]):
        """Build point set."""
        for attr in attrs:
            self.build_point('.'.join([ids_node, attr]))

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            self.build_0d(
                'flux_control',
                ['i_plasma', 'loop_voltage', 'li_3', 'beta_normal'])
            self.build_0d(
                'position_control',
                ['minor_radius', 'elongation', 'elongation_upper',
                 'elongation_lower', 'triangularity', 'triangularity_upper',
                 'triangularity_lower', 'active_limiter_point.r',
                 'active_limiter_point.z'])
            self.build_points(
                'position_control',
                ['x_point', 'strike_point', 'boundary_outline'])

        return self



if __name__ == '__main__':

    ids = 135007, 4
    #ids = 135011, 7


    PulseSchedule(*ids)._clear()
    pulse = PulseSchedule(*ids)

    print(pulse.data.data_vars)
