"""Extend pandas.DataFrame to manage coil and subcoil data."""

from typing import Optional, Collection, Any, Union

import pandas
import numpy as np

from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.dataframearray import DataFrameArray
from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.multipoint import MultiPoint
from nova.electromagnetic.polygon import Polygon


# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object


class SuperFrame(DataFrame):
    """
    Extend DataFrame or DataFrameArray. Frame superclass.

    Manage Frame metadata (metaarray, metaframe).

    Implement current properties.
    """

    _attributes = ['multipoint', 'subspace', 'polygon']

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, Union[MetaArray, MetaFrame]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns)
        self.update_metaframe()
        self.update_attrs(data, attrs)
        self.update_metadata(metadata)
        self.update_index()
        self.multipoint = MultiPoint(self)
        self.polygon = Polygon(self)

    def __getattr__(self, col):
        """Intercept DataFrame.__getattr__ to serve self.attrs."""
        if col in self.attrs:
            return self.attrs[col]
        return super().__getattr__(col)

    def __setattr__(self, col, value):
        """Check lock. Extend DataFrame.__setattr__ (frame.* = *).."""
        if col in self._attributes:
            self.attrs[col] = value
            return None
        self.metaframe.check_lock(col)
        return super().__setattr__(col, value)

    def __setitem__(self, col, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        self.metaframe.check_lock(col)
        return super().__setitem__(col, value)

    def _set_value(self, index, col, value, takeable=False):
        """Check lock. Extend DataFrame._set_value. (frame.at[i, '*'] = *)."""
        self.metaframe.check_lock(col)
        return super()._set_value(index, col, value, takeable)

    def update_attrs(self, data, attrs):
        """Update DataFrame attributes, [metaarray, metaframe]'."""
        if hasattr(data, 'attrs'):
            self.attrs |= data.attrs
        if attrs is None:
            attrs = {}
        self.attrs |= attrs

    def update_metaframe(self):
        """Update metaframe if not present in self.attrs."""
        if 'metaframe' not in self.attrs:
            self.attrs['metaframe'] = MetaFrame()

    def extract_metadata(self, metadata):
        """Extract metadata. Set defaults and meta*.metadata."""
        default = {}
        framearray = {attr: {} for attr in self.metaattrs}
        if metadata is None:
            metadata = {}
        if 'metadata' in metadata:
            metadata |= metadata.pop('metadata')
        for field in list(metadata):
            for attr in framearray:
                if hasattr(getattr(self, attr), field.lower()):
                    framearray[attr][field] = metadata.pop(field)
                    break
            else:
                if field in self.metaframe.default:
                    default[field] = metadata.pop(field)
        if len(metadata) > 0:
            raise IndexError('unreconised attributes set in **metadata: '
                             f'{metadata}.')
        for attr in framearray:
            getattr(self, attr).metadata |= framearray[attr]
        self.metaframe.default |= default

    def update_metadata(self, metadata):
        """Update metadata."""
        columns = not self.columns.empty
        self.extract_metadata(metadata)
        self.reset_metaframe(columns)

    def reset_metaframe(self, columns):
        """Revise metaframe to match self.columns if self.columns not empty."""
        metadata = {}
        if columns:
            for attribute in ['required', 'additional']:
                metadata[attribute.capitalize()] = [
                    attr for attr in getattr(self.metaframe, attribute)
                    if attr in self.columns]
            self.metadata = metadata

    @property
    def metaattrs(self) -> list[str]:
        """Return meta* attrs."""
        return [attr for attr in self.attrs if 'meta' in attr]

    @property
    def metadata(self):
        """Manage FrameArray metadata via the MetaFrame class."""
        metadata = {}
        for attr in self.metaattrs:
            metadata |= getattr(self, attr).metadata
        return metadata

    @metadata.setter
    def metadata(self, metadata):
        for attr in self.metaattrs:
            getattr(self, attr).metadata = metadata
        self.validate_metadata()

    def validate_metadata(self):
        """Validate required and additional attributes in FrameArray."""
        # check for additional attributes in metaarray.array
        if hasattr(self, 'metaarray'):
            unset = [attr not in self.metaframe.columns
                     for attr in self.metaarray.array]
            if np.array(unset).any():
                raise IndexError(
                    'attributes in metadata.array '
                    f'{np.array(self.metaarray.array)[unset]}'
                    ' not found in metaframe.required '
                    f'{self.metaframe.required}'
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
            # update nan
            isnan = [pandas.isna(self.loc[:, attr]).any()
                     for attr in self.metaframe.additional]
            if np.array(isnan).any():
                nan = np.array(self.metaframe.additional)[isnan]
                for attr in nan:
                    if attr in self.metaframe.default:
                        index = pandas.isna(self.loc[:, attr])
                        self.loc[index, attr] = self.metaframe.default[attr]

    def update_index(self):
        """Reset index if self.index is unset."""
        if isinstance(self.index, pandas.RangeIndex):
            self['index'] = self._build_index(self)
            self.set_index('index', inplace=True)
            self.index.name = None
