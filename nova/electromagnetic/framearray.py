"""Extend pandas.DataFrame to manage coil and subcoil data."""

from typing import Optional, Collection, Any, Union

import pandas
import numpy as np

from nova.electromagnetic.dataarray import DataArray
from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.metaframe import MetaFrame

# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object


class FrameArray(DataArray):
    """Extend dataarray. Manage metadata."""

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, Union[MetaArray, MetaFrame]] = None,
                 metadata: Optional[dict] = None,
                 **default: Optional[dict]):
        super().__init__(data, index, columns)
        self.update_attrs(data, attrs)
        self.update_metaframe()
        self.update_default(default)
        self.update_metadata(metadata)
        self.update_index()
        self.update_columns()

    def update_attrs(self, data, attrs):
        """Update DataFrame attributes, [metaarray, metaframe]'."""
        if hasattr(data, 'attrs'):
            self.attrs |= data.attrs
        if attrs is None:
            attrs = {}
        self.attrs |= attrs
        extra = np.array([attr not in ['metaarray', 'metaframe']
                          for attr in self.attrs], bool)
        if extra.any():
            raise IndexError('unrecognised attributes set in self.attrs '
                             f' {np.array(list(self.attrs.keys()))[extra]}')

    def update_metaframe(self):
        """Update metaframe if not present in self.attrs."""
        if 'metaframe' not in self.attrs:
            self.attrs['metaframe'] = MetaFrame()

    @property
    def metaframe(self):
        """Manage metaframe instance."""
        return self.attrs['metaframe']

    def replace_metadata(self, Metadata):
        """Replace metadata extracted from framearray."""
        if Metadata is None:
            Metadata = {}
        self.metadata = {attr.capitalize(): Metadata[attr]
                         for attr in Metadata}

    def update_default(self, default):
        """Update metaframe.defaults."""
        if default is None:
            default = {}
        extend = np.array([attr not in self.metaframe.default
                           for attr in default])
        if extend.any():
            raise IndexError('additional default attributes set in **default '
                             f'{np.array(list(default.keys()))[extend]} '
                             'extend default set using metadata.')
        self.metaframe.default |= default

    def update_metadata(self, metadata):
        """Update metadata."""
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @property
    def metadata(self):
        """Manage FrameArray metadata via the MetaFrame class."""
        metadata = {}
        for attr in self.attrs:
            metadata |= getattr(self, attr).metadata
        return metadata

    @metadata.setter
    def metadata(self, metadata):
        for attr in self.attrs:
            getattr(self, attr).metadata = metadata
            getattr(self, attr).validate()
        self.validate_metadata()

    def validate_metadata(self):
        """Validate required and additional attributes in FrameArray."""
        # check for additional attributes in metaarray.array
        unset = [attr not in self.metaframe.columns
                 for attr in self.metaarray.array]
        if np.array(unset).any():
            raise IndexError(
                'attributes in metadata.array '
                f'{np.array(self.metaarray.array)[unset]}'
                f' not found in metaframe.required {self.metaframe.required}'
                f'or metaframe.additional {self.metaframe.additional}')
        if not self.empty:
            columns = self.columns.to_list()
            required_unset = [attr not in columns
                              for attr in self.metaframe.required]
            if np.array(required_unset).any():
                unset = np.array(self.metaframe.required)[required_unset]
                raise IndexError(f'required attributes missing {unset}')
            # extend additional
            additional = [attr for attr in columns
                          if attr not in self.metaframe.columns]
            if additional:
                self.metadata = {'additional': additional}
            # set defaults
            additional_unset = [attr not in columns
                                for attr in self.metaframe.additional]
            if np.array(additional_unset).any():
                unset = np.array(self.metaframe.additional)[additional_unset]
                for attr in unset:
                    self.loc[:, attr] = self.metaframe.default[attr]


    def update_index(self):
        """Reset index if self.index is unset."""
        if isinstance(self.index, pandas.RangeIndex):
            self['index'] = self._build_index(self)
            self.set_index('index', inplace=True)
            self.index.name = None

    def update_columns(self):
        if not self.columns.empty:
            '''
            print('columns', self.columns)
            columns_unset = np.array([attr not in self.metaframe.columns
                                      for attr in self.columns])
            if columns_unset.any():
                raise IndexError(
                    'requested FrameArray columns '
                    f'{self.columns.to_numpy()[columns_unset]} '
                    'not found in self.metaframe.columns '
                    f'{self.metaframe.columns}')
            '''
            # intersection of self.columns and self.metaframe.columns
            metadata = {}
            metadata['Required'] = [attr for attr
                                    in self.metaframe.required
                                    if attr in self.columns]
            metadata['Additional'] = [attr for attr in self.columns
                                      if attr not in self.metaframe.required]
            if self.metaframe.required != metadata['Required'] or \
                    self.metaframe.additional != metadata['Additional']:
                self.metadata = metadata  # perform intersection
