"""Manage access to dynamic coil data data."""
from dataclasses import dataclass, field

from nova.frame.baseplot import Plot
from nova.imas.scenario import Scenario


@dataclass
class PF_Active(Plot, Scenario):
    """Manage access to pf_active ids."""

    name: str = 'pf_active'
    ids_node: str = 'coil'
    coil_attrs: list[str] = field(
        default_factory=lambda: ['current', 'b_field_max_timed'])

    @staticmethod
    def coil_name(coil):
        """Return coil identifier, return coil name if empty."""
        if not coil.identifier:
            return coil.name
        return coil.identifier

    def build(self):
        """Build netCDF database using data extracted from imasdb."""
        name = [self.coil_name(coil).strip() for coil in self.ids_data.coil]
        with self.build_scenario():
            self.data.coords['coil_name'] = name
            self.data.coords['coil_index'] = 'coil_name', range(len(name))
            self.append(('time', 'coil_name'), self.coil_attrs, '*.data')
            for force in ['radial', 'vertical']:
                with self.ids_index.node(f'{force}_force'):
                    self.append(('time', 'coil_name'), ['force'], '*.data',
                                prefix=f'{force}_')
        return self

    def plot(self, axes=None):
        """Plot current timeseries."""
        self.set_axes('1d', axes=axes)
        self.axes.plot(self.data.time, self.data.current)


if __name__ == '__main__':

    # pf_active = PF_Active(130506, 403, machine='iter')
    pulse, run = 105028, 1
    pulse, run = 105011, 9
    pulse, run = 105007, 9
    pulse, run = 105011, 10
    pulse, run = 135003, 5
    #pulse, run = 115002, 4
    pulse, run = 135007, 4
    pulse, run = 135011, 7
    PF_Active(pulse, run)._clear()
    pf_active = PF_Active(pulse, run)
    #pf_active = PF_Active(105007, 9)  # b field max timed 135002, 5
    pf_active.plot()
