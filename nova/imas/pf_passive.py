"""Manage access to dynamic pf passive data."""
from dataclasses import dataclass, field

import numpy as np

from nova.frame.baseplot import Plot
from nova.imas.scenario import Scenario


@dataclass
class PF_Passive(Plot, Scenario):
    """Manage access to pf_passive ids."""

    name: str = 'pf_passive'
    loop_attrs: list[str] = field(default_factory=lambda: ['current'])

    @staticmethod
    def loop_name(coil):
        """Return coil identifier, return coil name if empty."""
        if not coil.identifier:
            return coil.name
        return coil.identifier

    def initalize(self):
        """Create xarray coil data entries."""
        coords = self.data.attrs['pf_passive']
        shape = tuple(self.data.dims[coordinate] for coordinate in coords)
        '''
        coil = self.ids.coil[0]
        for attr in list(self.coil_attrs):
            if len(getattr(coil, attr).data) > 0:
                self.data[attr] = coords, np.zeros(shape, float)
            else:
                self.coil_attrs.remove(attr)
        '''

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            coords = self.data.attrs['pf_passive'] = ['time', 'loop_name']
            self.data.attrs['pf_passive']
            self.data['time'] = self.ids.time
            self.data['loop_name'] = self.get_ids('loop(:)/name')
            self.data.attrs['pf_passive'] = ['time', 'loop_name']
            #self.initalize()
            for attr in self.loop_attrs:
                try:
                    self.data[attr] = coords, self.get_ids(f'loop(:)/{attr}').T
                except ValueError:  # skip missing attributes
                    pass
            '''
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
            '''
        return self


if __name__ == '__main__':

    pf_passive = PF_Passive(135013, 2).build()
