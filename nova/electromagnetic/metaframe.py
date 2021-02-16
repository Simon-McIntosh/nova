"""Manage CoilFrame metadata."""

from dataclasses import dataclass, field, fields
from typing import Union

import numpy as np

# pylint:disable=unsubscriptable-object


@dataclass
class MetaFrame:
    """Manage CoilFrame metadata - accessed via CoilFrame['attrs']."""

    required: list[str] = field(default_factory=lambda: ['x', 'z', 'dl', 'dt'])
    additional: list[str] = field(default_factory=lambda: ['rms', 'mpc'])
    default: dict[str, Union[float, str, ]] = field(
        repr=False, default_factory=lambda: {
            'dCoil': 0., 'nx': 1, 'nz': 1, 'Nt': 1., 'Nf': 1,
            'rms': 0., 'dx': 0., 'dz': 0., 'dA': 0., 'dl_x': 0., 'dl_z': 0.,
            'm': '', 'R': 0.,  'rho': 0.,
            'turn_fraction': 1., 'skin_fraction': 1.,
            'cross_section': 'rectangle', 'turn_section': 'rectangle',
            'patch': None, 'polygon': None,
            'coil': '', 'part': '', 'subindex': None, 'material': '',
            'mpc': '',
            'active': True, 'optimize': False, 'plasma': False,
            'feedback': False, 'acloss': False,
            'Ic': 0., 'It': 0., 'Psi': 0., 'Bx': 0., 'Bz': 0., 'B': 0.})
    coildata: dict = None
    dataframe: dict = None

    mode = {'required': 'replace',
            'additional': 'extend',
            'default': 'update'}

    def __post_init__(self):
        """Check that all additional attributes have default values."""
        self._check_default()

    @property
    def number_required(self):
        """Return number of required arguments."""
        return len(self.required)

    @property
    def metadata(self):
        """Manage metadata."""
        return {field.name: getattr(self, field.name)
                for field in fields(self)}

    @metadata.setter
    def metadata(self, metadata):
        for attribute in metadata:
            fieldname = attribute if attribute in self.mode else 'default'
            mode = self.mode[fieldname]
            if mode == 'replace':
                setattr(self, fieldname, metadata[attribute])
            if mode == 'extend':
                unique = [attr for attr in metadata[attribute]
                          if attr not in getattr(self, fieldname)]
                getattr(self, fieldname).extend(unique)
            elif mode == 'update':
                getattr(self, fieldname).update(metadata[attribute])
            if fieldname == 'additional':
                self._check_default()

    def _check_default(self):
        unset = np.array([attr not in self.default
                          for attr in self.additional])
        if unset.any():
            raise ValueError('Default value not set for additional attributes '
                             f'{np.array(self.additional)[unset]}')


if __name__ == '__main__':

    metaframe = MetaFrame()
    metaframe.metadata = {'additional': ['mpc']}
    print(metaframe.metadata)
