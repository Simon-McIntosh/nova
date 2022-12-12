"""Load magnetics from machine description."""
from dataclasses import dataclass, field
from typing import ClassVar

import pandas

from nova.imas.database import Database


@dataclass
class Magnetics(Database):
    """Manage active poloidal loop ids, pf_passive."""

    pulse: int = 150100
    run: int = 4
    name: str = 'magnetics'
    user: str = 'public'
    machine: str = 'iter_md'
    data: pandas.DataFrame = field(init=False,
                                   default_factory=pandas.DataFrame)

    signal: ClassVar[dict[str, str]] = dict(
        A3='i', A4='i', A5='p', A6='p', )

    diagnostic: ClassVar[dict[str, list[str]]] = dict(
        flux_loop=['toroidal', 'saddle', 'diamagnetic_internal',
                   'diamagnetic_external', 'diamagnetic_compensation',
                   'diamagnetic_differential'],
        b_field_pol_probe=['position', 'mirnov', 'hall', 'flux_gate',
                           'faraday_fiber', 'differential'],
        b_field_tor_probe=['position', 'mirnov', 'hall', 'flux_gate',
                           'faraday_fiber', 'differential'],
        rogowski_coil=[],
        shunt=[])

    def __post_init__(self):
        """Load data from magnetics IDS and build overview."""
        super().__post_init__()
        self.build_data()
        self.build_overview()

    def build_data(self):
        """Extract magnetics data."""
        for diagnostic in self.diagnostic:
            name, identifier, diagnostic_name, diagnostic_type = [], [], [], []
            for ids in self.get_ids(diagnostic):
                name.append(ids.name)
                identifier.append(ids.identifier)
                diagnostic_name.append(diagnostic)
                try:
                    diagnostic_type.append(
                        self.diagnostic[diagnostic][ids.type.index-1])
                except AttributeError:
                    diagnostic_type.append(diagnostic)

            data = pandas.DataFrame(dict(name=name,
                                         diagnostic_name=diagnostic_name,
                                         diagnostic_type=diagnostic_type),
                                    index=identifier)
            self.data = pandas.concat([self.data, data])

    def build_overview(self):
        """Extract overview from dataframe."""
        index, identifier, name, diagnostic_type, number = [], [], [], [], []
        for data_name in self.data.name.unique():
            frame = self.data.loc[self.data.name == data_name, :]
            index.append(data_name.split(' ')[0].split('.')[1])
            identifier.append('-'.join(frame.index[0].split('-')[:-1]))
            name.append(' '.join(data_name.split(' ')[1:]))
            type_array = frame.diagnostic_type.unique()
            if len(type_array) != 1:
                raise ValueError(f'diagnostic type not unique for {data_name} '
                                 f'{frame.diagnostic_type.unique()}')
            diagnostic_type.append(type_array[0])
            number.append(len(frame))
        self.summary = pandas.DataFrame(dict(name=name, identifier=identifier,
                                             diagnostic=diagnostic_type,
                                             number=number), index=index)

    def signal_types(self):
        """Add signal type information."""


if __name__ == '__main__':

    magnetics = Magnetics()
    print(magnetics.summary)
