
from dataclasses import dataclass, field, fields
from typing import Union

import numpy as np

from nova.electromagnetic.metadata import MetaData

# pylint:disable=unsubscriptable-object


@dataclass
class MetaFrame(MetaData):
    """Manage Frame metadata - accessed via Frame['attrs']."""

    required: list[str] = field(default_factory=lambda: ['x', 'z', 'dl', 'dt'])
    additional: list[str] = field(default_factory=lambda: [])
    exclude: list[str] = field(default_factory=lambda: [])
    default: dict[str, Union[float, str, bool, None]] = field(
        repr=False, default_factory=lambda: {
            'dCoil': 0., 'nx': 1, 'nz': 1, 'Nt': 1., 'Nf': 1,
            'rms': 0., 'dx': 0., 'dz': 0., 'dA': 0.,
            'dl': 0.1, 'dt': 0.1, 'dl_x': 0., 'dl_z': 0.,
            'm': '', 'R': 0.,  'rho': 0.,
            'turn_fraction': 1., 'skin_fraction': 1.,
            'section': 'rectangle', 'turn': 'rectangle',
            'patch': None, 'poly': None, 'coil': '', 'part': '',
            'subindex': None, 'material': '',
            'link': '', 'factor': 1., 'reference': 0,
            'active': True, 'optimize': False, 'plasma': False,
            'feedback': False, 'acloss': False,
            'Ic': 0., 'It': 0., 'Psi': 0., 'Bx': 0., 'Bz': 0., 'B': 0.})
    index: dict[str, Union[str, bool]] = field(
        repr=False, default_factory=lambda: {
            'name': '', 'label': 'Coil', 'delim': '', 'offset': 0})

    def validate(self):
        """
        Extend MetaData.validate.

            - Exclude duplicate values from self.required in self.additional.
            - Check that all additional attributes have a default value.

        """
        super().validate()
        # exculde duplicate values
        self.additional = [attr for attr in self.additional
                           if attr not in self.required]
        # check for exclude attributes in required
        exclude_required = np.array([attr in self.required
                                     for attr in self.exclude])
        if exclude_required.any():
            raise IndexError('exclude attributes '
                             f'{np.array(self.exclude)[exclude_required]} '
                             'specified in self.required')
        # remove exclude attributes from additional
        self.additional = [attr for attr in self.additional
                           if attr not in self.exclude]
        # check unset defaults
        unset = np.array([attr not in self.default
                          for attr in self.additional])
        if unset.any():
            raise ValueError('default value not set for additional attributes '
                             f'{np.array(self.additional)[unset]}')
        # block index field extension
        index_default = next(field.default_factory() for field in fields(self)
                             if field.name == 'index')
        extend = np.array([attr not in index_default for attr in self.index])
        if extend.any():
            raise IndexError('additional attributes passed to index field '
                             f'{np.array(list(self.index.keys()))[extend]}')

    @property
    def required_number(self):
        """Return number of required arguments."""
        return len(self.required)

    @property
    def columns(self):
        """Return metaframe columns."""
        return self.required + self.additional
