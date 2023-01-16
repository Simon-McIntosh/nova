"""Manage access to dynamic coil data data."""
from dataclasses import dataclass, field

import numpy as np

from nova.frame.baseplot import Plot
from nova.imas.scenario import Scenario


@dataclass
class PF_Active(Plot, Scenario):
    """Manage access to pf_active ids."""

    name: str = 'pf_active'
    coil_attrs: list[str] = field(
        default_factory=lambda: ['current', 'b_field_max_timed'])

    @staticmethod
    def coil_name(coil):
        """Return coil identifier, return coil name if empty."""
        if not coil.identifier:
            return coil.name
        return coil.identifier

    def initalize(self):
        """Create xarray coil data entries."""
        coords = self.data.attrs['pf_active']
        shape = tuple(self.data.dims[coordinate] for coordinate in coords)
        coil = self.ids.coil[0]
        for attr in list(self.coil_attrs):
            if len(getattr(coil, attr).data) > 0:
                self.data[attr] = coords, np.zeros(shape, float)
            else:
                self.coil_attrs.remove(attr)

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            self.data['time'] = self.ids.time
            coil_names = [self.coil_name(coil).strip()
                          for coil in self.ids.coil]
            self.data['coil_name'] = coil_names
            self.data['coil_index'] = 'coil_name', range(len(coil_names))
            self.data.attrs['pf_active'] = ['time', 'coil_name']
            self.initalize()
            for i in range(len(coil_names)):
                coil = self.ids.coil[i]
                for attr in self.coil_attrs:
                    try:
                        self.data[attr][:, i] = getattr(coil, attr).data
                    except ValueError:  # skip missing attributes
                        pass
        return self

    def plot(self, axes=None):
        """Plot current timeseries."""
        self.set_axes(axes, '1d')
        self.axes.plot(self.data.time, self.data.current)


if __name__ == '__main__':

    # pf_active = PF_Active(130506, 403, machine='iter')
    pf_active = PF_Active(135013, 2)
    #pf_active = PF_Active(105007, 9)  # b field max timed 135002, 5
    pf_active.plot()
