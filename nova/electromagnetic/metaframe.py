"""Manage frame metadata."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from nova.electromagnetic.metadata import MetaData

# pylint:disable=unsubscriptable-object


@dataclass
class MetaFrame(MetaData):
    """
    Manage Frame metadata.

    - required: required column, set as *args
    - additional: additional columns, set as **kwargs
    - default
    """

    required: list[str] = field(default_factory=lambda: ['x', 'z', 'dl', 'dt'])
    additional: list[str] = field(default_factory=lambda: [])
    exclude: list[str] = field(default_factory=lambda: [])
    subspace: list[str] = field(default_factory=lambda: [
        'Ic', 'It', 'Nt', 'active', 'plasma', 'optimize', 'feedback'])
    default: dict[str, Union[float, str, bool, None]] = field(
        repr=False, default_factory=lambda: {
            'x': 0., 'z': 0.,
            'dCoil': 0., 'nx': 1, 'nz': 1, 'Nt': 1., 'Nf': 1,
            'rms': 0., 'dx': 0., 'dz': 0., 'dA': 0.,
            'dl': 0.1, 'dt': 0.1, 'dl_x': 0., 'dl_z': 0.,
            'm': '', 'R': 0.,  'rho': 0.,
            'turn_fraction': 1., 'skin_fraction': 1.,
            'section': 'rectangle', 'turn': 'rectangle',
            'patch': None, 'poly': None, 'coil': '', 'part': '',
            'subindex': None, 'material': '',
            'link': '', 'factor': 1., 'ref': 0, 'subref': 0,
            'active': True, 'optimize': False, 'plasma': False,
            'feedback': False, 'acloss': False,
            'Ic': 0., 'It': 0., 'Psi': 0., 'Bx': 0., 'Bz': 0., 'B': 0.,
            'name': '', 'label': 'Coil', 'delim': '', 'offset': 0})
    _lock: dict[str, bool] = field(default_factory=lambda: {
        'subspace': True, 'energize': False}, init=False)

    def lock(self, key=None):
        """
        Return lock status.

        Parameters
        ----------
        key : str
            Lock label.

        """
        if key is None:
            return self._lock
        else:
            return self._lock[key]

    @contextmanager
    def setlock(self, status, keys=None):
        """
        Manage access to subspace frame variables.

        Parameters
        ----------
        status : Union[bool, None]
            Subset lock status.
        keys : Union[str, list[str]]
            Lock label, if None set all keys in self._lock.

        Returns
        -------
        None.

        """
        if keys is None:
            keys = list(self._lock.keys())
        if isinstance(keys, str):
            keys = [keys]
        _lock = {key: self._lock[key] for key in keys}
        self._lock |= {key: status for key in keys}
        yield
        self._lock |= _lock

    def validate(self):
        """
        Extend MetaData.validate.

            - Raise error if exclude attributes specified as required.
            - Subtract reduce attributes from additional
            - Ensure that all additional attributes have a default value.

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

    @property
    def columns(self):
        """Return metaframe columns."""
        return self.required + self.additional
