"""Manage access to dynamic coil data data."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from nova.imas.scenario import Scenario
from nova.utilities.pyplot import plt


@dataclass
class PF_Active(Scenario):
    """Manage access to pf_active ids."""

    pulse: int
    run: int
    name: str = 'pf_active'
    coil_attrs: list[str] = field(
        default_factory=lambda: ['current'])

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
            coil_names = [self.coil_name(coil) for coil in self.ids.coil]
            self.data['coil_index'] = range(len(coil_names))
            self.data['coil_name'] = 'coil_index', coil_names
            self.data.attrs['pf_active'] = ['time', 'coil_index']
            self.initalize()
            for i in range(len(coil_names)):
                coil = self.ids.coil[i]
                for attr in self.coil_attrs:
                    self.data[attr][:, i] = getattr(coil, attr).data

    def plot(self):
        """Plot current timeseries."""
        plt.plot(self.data.current)
        plt.despine()


if __name__ == '__main__':

    pf_active = PF_Active(135011, 7)
    #pf_active.build()
    pf_active.plot()
