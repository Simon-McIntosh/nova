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

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        with self.build_scenario():
            coords = self.data.attrs['pf_passive'] = ['time', 'loop_name']
            self.data.attrs['pf_passive']
            self.data['time'] = self.ids.time
            self.data['loop_name'] = self.get_ids('loop(:)/name')
            self.data.attrs['pf_passive'] = ['time', 'loop_name']
            for attr in self.loop_attrs:
                try:
                    self.data[attr] = coords, self.get_ids(f'loop(:)/{attr}').T
                except ValueError:  # skip missing attributes
                    pass
        return self


if __name__ == '__main__':

    pf_passive = PF_Passive(105028, 1)
