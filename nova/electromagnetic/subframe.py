"""Extend pandas.DataFrame to manage coil and subcoil data."""

from typing import Optional, Collection, Any

import pandas
import numpy as np

from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.array import MetaArray, Array


class Series(pandas.Series):
    """Provide series constructor methods."""

    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_expanddim(self):
        return SubFrame


class SubFrame(Array, pandas.DataFrame):
    """
    Extends Pandas.DataFrame.

    Inspiration for DataFrame inheritance taken from GeoPandas
    https://github.com/geopandas.
    """

    _attributes = {'metaframe': MetaFrame, 'metaarray': MetaArray}

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 attrs: Optional[dict] = None,
                 metadata: Optional[dict] = None,
                 **default: Optional[dict]):
        super().__init__(data, index)
        if isinstance(data, pandas.core.internals.managers.BlockManager):
            return
        self.update_attrs(attrs)
        self.extract_metadata(data)
        self.update_default(default)
        self.update_metadata(metadata)
        self.update_index()

    @property
    def _constructor(self):
        return SubFrame

    @property
    def _constructor_sliced(self):
        return Series

    @property
    def metaframe(self):
        """Return metaframe instance."""
        return self.attrs['metaframe']

    def update_attrs(self, attrs=None):
        """Update frame attrs and initialize."""
        super().update_attrs(attrs)
        if attrs is not None:
            self.attrs |= attrs
        self.generate_attribute('metaframe')

    def generate_attribute(self, attribute):
        """Generate meta* attributes. Store in self.attrs."""
        if attribute not in self.attrs:
            self.attrs[attribute] = self._attributes[attribute]()
            if 'metadata' in self.attrs:
                self.attrs['metadata'].append(attribute)
            else:
                self.attrs['metadata'] = [attribute]

    def validate_metadata(self):
        """Validate required and additional attributes in SubFrame."""
        super().validate_metadata()
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

    def extract_metadata(self, data):
        """Replace metadata extracted from frame."""
        if hasattr(data, 'metadata'):
            metadata = {attr.capitalize(): data.metadata[attr]
                        for attr in data.metadata}
            self.metadata = metadata

    def update_metadata(self, metadata):
        """Update metadata."""
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @property
    def metadata(self):
        """Manage SubFrame metadata via the MetaFrame class."""
        metadata = {}
        for attribute in self.attrs['metadata']:
            metadata |= getattr(self, attribute).metadata
        return metadata

    @metadata.setter
    def metadata(self, metadata):
        for attribute in self.attrs['metadata']:
            getattr(self, attribute).metadata = metadata
            getattr(self, attribute).validate()
        self.validate_metadata()

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

    def update_index(self):
        """Reset index if self.index is unset."""
        if isinstance(self.index, pandas.RangeIndex):
            self['index'] = self._build_index(self)
            self.set_index('index', inplace=True)
            self.index.name = None