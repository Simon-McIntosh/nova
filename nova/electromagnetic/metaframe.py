"""Manage frame metadata."""

from contextlib import contextmanager
from dataclasses import dataclass, field, InitVar
from typing import Iterable, Union

import pandas

import numpy as np

from nova.electromagnetic.metadata import MetaData

# pylint:disable=unsubscriptable-object


@dataclass
class Lock:
    """Manage variable access (subspace, energize, array)."""

    subspace: list[str] = field(init=False, default_factory=lambda: [])
    energize: list[str] = field(init=False, default_factory=lambda: [])
    array: list[str] = field(init=False, default_factory=lambda: [])
    _lock: dict[str, bool] = field(init=False, default_factory=lambda: {
        'subspace': False, 'energize': False, 'array': False,
        'multipoint': False})

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

    def hascol(self, attr, col):
        """Return Ture if col in attr."""
        try:
            return col in getattr(self, attr)
        except (KeyError, TypeError):
            return False

    def assert_hascol(self, attr, col):
        """Check for col in attr, raise error if not found."""
        try:
            assert self.hascol(attr, col)
        except AssertionError as hasnot:
            raise AssertionError(
                f'metaframe does not have {attr} or '
                f'{col} not in metaframe.{attr} '
                f'{getattr(self.metaframe, attr)}') from hasnot


@dataclass
class MetaArray(MetaData):
    """Manage Frame metadata - accessed via Frame['attrs']."""

    index: InitVar[list[str]] = field(default=None)
    data: dict[str, Iterable[Union[str, int, float]]] = field(init=False)

    _internal = ['index', 'data']

    def __post_init__(self, index):
        """Init update flags."""
        self.index = index
        self.data = {}
        super().__post_init__()

    @property
    def dataarray(self):
        """Return DataFrame representation of fast access data arrays."""
        return pandas.DataFrame(self.data, index=self.index,
                                columns=self.array)


@dataclass
class MetaFrame(MetaArray, Lock):
    """
    Manage Frame metadata.

    - required: required column, set as *args
    - additional: additional columns, set as **kwargs
    - default
    """

    required: list[str] = field(default_factory=lambda: [])
    additional: list[str] = field(default_factory=lambda: [])
    exclude: list[str] = field(default_factory=lambda: [])
    available: list[str] = field(default_factory=lambda: [])
    default: dict[str, Union[float, str, bool, None]] = field(
        repr=False, default_factory=lambda: {
            'x': 0., 'z': 0.,
            'delta': 0., 'nx': 1, 'nz': 1, 'Nt': 1., 'Nf': 1,
            'rms': 0., 'dx': 0., 'dz': 0., 'dA': 0.,
            'dl': 0.1, 'dt': 0.1, 'dl_x': 0., 'dl_z': 0.,
            'm': '', 'R': 0.,  'rho': 0.,
            'turn_fraction': 1., 'section': 'rectangle', 'turn': 'rectangle',
            'patch': None, 'poly': None, 'frame': '', 'part': '',
            'subindex': None, 'material': '',
            'link': '', 'factor': 1., 'ref': 0, 'subref': 0,
            'active': True, 'optimize': False, 'plasma': False,
            'feedback': False, 'acloss': False,
            'Ic': 0., 'It': 0., 'Psi': 0., 'Bx': 0., 'Bz': 0., 'B': 0.,
            'name': '', 'label': 'Coil', 'delim': '', 'offset': 0})
    tag: list[str] = field(default_factory=lambda: [
        'name', 'label', 'delim', 'offset'])

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

    @property
    def columns(self):
        """Return metaframe columns."""
        return list(dict.fromkeys(self.required + self.additional))

    def update(self, metadata):
        """
        Update metaframe metadata.

        - Extend metaframe attributes inculded in **metadata.
        - Permit per-item update of default dict.

        """
        if metadata is None:
            metadata = {}
        if 'metadata' in metadata:
            metadata |= metadata.pop('metadata')
        for attr in list(metadata):
            if hasattr(self, attr.lower()):
                self.metadata = {attr: metadata.pop(attr)}
            elif attr in self.default:
                self.default[attr] = metadata.pop(attr)
        if len(metadata) > 0:
            raise IndexError('unreconised attributes set in **metadata: '
                             f'{metadata}.')
