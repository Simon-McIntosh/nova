"""Manage frame metadata."""
from dataclasses import dataclass, field
from typing import Iterable, Union

import pandas

import numpy as np

from nova.frame.metadata import MetaData

# pylint:disable=unsubscriptable-object


@dataclass
class MetaArray(MetaData):
    """Manage DataFrame metadata - accessed via DataFrame['attrs']."""

    index: pandas.Index = field(repr=False, default=pandas.Index([]))
    data: dict[str, Iterable[Union[str, int, float]]] = field(
        init=False, default_factory=dict)
    version: dict[str, int] = field(
        init=False, default_factory=lambda: dict(index=0))

    def __post_init__(self):
        """Set internal data variables and initialize version dict."""
        self.metadata = {'_internal':
                         ['index', 'data', 'lock', 'default']}
        super().__post_init__()

    @property
    def dataframe(self):
        """Return DataFrame representation of fast access data arrays."""
        return pandas.DataFrame(self.data, index=self.index)


@dataclass
class MetaSet(MetaArray):
    """Manage variable access to frame subsets (subspace, energize, array)."""

    subspace: list[str] = field(default_factory=lambda: [])
    energize: list[str] = field(default_factory=lambda: [])
    array: list[str] = field(default_factory=lambda: [])
    lock: dict[str, bool] = field(default_factory=lambda: {
        'subspace': False, 'energize': False, 'array': False,
        'multipoint': False, 'column': False})

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
                f'{col} not in metaframe.{attr}: '
                f'{getattr(self, attr)}') from hasnot


@dataclass
class MetaFrame(MetaSet):
    """
    Manage DataFrame metadata.

    - required: required column, set as *args
    - additional: additional columns, set as **kwargs
    - default
    """

    base: list[str] = field(default_factory=lambda: [])
    required: list[str] = field(default_factory=lambda: [])
    additional: list[str] = field(default_factory=lambda: [])
    exclude: list[str] = field(default_factory=lambda: [])
    available: list[str] = field(default_factory=lambda: [])
    default: dict[str, Union[float, str, bool, None]] = field(
        repr=False, default_factory=lambda: {
            'x': 0., 'y': 0., 'z': 0., 'xo': 0., 'zo': 0., 'segment': 'ring',
            'dl': 0.1, 'dt': 0.1, 'rms': 0.,
            'dx': 0., 'dy': 0., 'dz': 0.,
            'area': 0., 'volume': 0.,
            'delta': 0., 'nturn': 1., 'nfilament': 1.,
            'material': '', 'mass': '', 'rho': 0.,
            'section': 'rectangle', 'turn': 'rectangle', 'body': '',
            'turnturn': 1., 'scale': 1., 'skin': 1.,
            'poly': None, 'vtk': None, 'frame': '', 'part': '',
            'link': '', 'factor': 1., 'ref': 0, 'subref': 0,
            'active': True, 'fix': True,
            'plasma': False, 'filament': False,
            'passive': False, 'free': False, 'coil': True,
            'ionize': False, 'acloss': False,
            'ferritic': False,
            'Ic': 0., 'It': 0.,
            'I': 0., 'Imin': 0., 'Imax': 0.,
            'V': 0., 'Vmin': 0., 'Vmax': 0., 'R': 0.,
            'Psi': 0., 'Bx': 0., 'Bz': 0., 'B': 0.,
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
        # propergate subspace variables to available
        available_unset = [attr for attr in self.subspace
                           if attr not in self.available]
        if available_unset:
            self.available.extend(available_unset)
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
        # remove exclude attributes from available
        self.available = [attr for attr in self.available
                          if attr not in self.exclude]

    @property
    def columns(self):
        """Return metaframe columns."""
        return list(dict.fromkeys(self.base + self.required + self.additional))

    def update(self, metadata):
        """
        Update metaframe metadata.

        - Extend metaframe attributes inculded in **metadata.
        - Permit per-item update of default dict.

        """
        if metadata is None:
            return
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
