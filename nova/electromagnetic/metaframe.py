
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from nova.electromagnetic.metadata import MetaData


@dataclass
class MetaFrame(MetaData):
    """Manage CoilFrame metadata - accessed via CoilFrame['attrs']."""

    required: list[str] = field(default_factory=lambda: ['x', 'z', 'dl', 'dt'])
    additional: list[str] = field(default_factory=lambda: ['rms', 'mpc'])
    default: dict[str, Union[float, str, bool, None]] = field(
        repr=False, default_factory=lambda: {
            'dCoil': 0., 'nx': 1, 'nz': 1, 'Nt': 1., 'Nf': 1,
            'rms': 0., 'dx': 0., 'dz': 0., 'dA': 0., 'dl_x': 0., 'dl_z': 0.,
            'm': '', 'R': 0.,  'rho': 0.,
            'turn_fraction': 1., 'skin_fraction': 1.,
            'cross_section': 'rectangle', 'turn_section': 'rectangle',
            'patch': None, 'polygon': None,
            'coil': '', 'part': '',
            'subindex': None, 'material': '', 'mpc': '',
            'active': True, 'optimize': False, 'plasma': False,
            'feedback': False, 'acloss': False,
            'Ic': 0., 'It': 0., 'Psi': 0., 'Bx': 0., 'Bz': 0., 'B': 0.})
    frame: dict[str, Union[str, bool]] = field(
        repr=False, default_factory=lambda: {
            'name': '', 'label': 'Coil', 'delim': '_', 'link': True})

    def validate(self):
        """
        Extend MetaData.validate.

            - Exclude duplicate values from self.required in self.additional.
            - Check that all additional attributes have a default value.

        """
        MetaData.validate(self)
        # exculde duplicate values
        self.additional = [attr for attr in self.additional
                           if attr not in self.required]
        # check defaults
        unset = np.array([attr not in self.default
                          for attr in self.additional])
        if unset.any():
            raise ValueError('Default value not set for additional attributes '
                             f'{np.array(self.additional)[unset]}')

    @property
    def required_number(self):
        """Return number of required arguments."""
        return len(self.required)
