"""Extend pandas.DataFrame to manage coil and subcoil data."""

import re
import string
from typing import Optional, Collection, Any, Union

import pandas
import numpy as np

from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.multipoint import MultiPoint
from nova.electromagnetic.polygon import Polygon


# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object


class SuperFrame(DataFrame):
    """
    Extend DataFrame. Frame superclass.

    Manage Frame metadata (metaarray, metaframe).

    Implement current properties.
    """

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
        """Extract metadata. Set default, tag, and meta*.metadata."""
        update = {attr: {} for attr in ['default', 'tag']}
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
                for attr in update:
                    if field in getattr(self.metaframe, attr):
                        update[attr][field] = metadata.pop(field)
        for attr in update:
            getattr(self.metaframe, attr).update(update[attr])
        if len(metadata) > 0:
            raise IndexError('unreconised attributes set in **metadata: '
                             f'{metadata}.')
        for attr in framearray:
            getattr(self, attr).metadata |= framearray[attr]

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
            self.format_columns()

    def format_columns(self):
        """
        Format DataFrame columns.

            - required unset: raise error
            - additional unset: insert default
            - isnan: insert default
        """
        with self.metaframe.setlock(None):
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

    def _set_offset(self, metaindex):
        try:  # reverse search through frame index
            match = next(name for name in self.index[::-1]
                         if metaindex['label'] in name)
            offset = re.sub(r'[a-zA-Z]', '', match)
            if isinstance(offset, str):
                offset = offset.replace(metaindex['delim'], '')
                offset = offset.replace('_', '')
                offset = int(offset)
            offset += 1
        except (TypeError, StopIteration):  # unit index, label not present
            offset = 0
        metaindex['offset'] = np.max([offset, metaindex['offset']])

    def _build_index(self, data, **kwargs):
        data_length = self._data_length(data)
        # update metaframe.tag
        self.metaframe.tag |= {key: kwargs[key] for key in kwargs
                               if key in self.metaframe.tag}
        metaindex = self.metaframe.tag
        if kwargs.get('name', metaindex['name']):  # valid name
            name = metaindex['name']
            if pandas.api.types.is_list_like(name) or data_length == 1:
                return self._check_index(name, data_length)
            if metaindex['delim'] and metaindex['delim'] in name:
                split_name = name.split(metaindex['delim'])
                metaindex['label'] = metaindex['delim'].join(split_name[:-1])
                metaindex['offset'] = int(split_name[-1])
            else:
                metaindex['delim'] = ''
                metaindex['label'] = name.rstrip(string.digits)
                metaindex['offset'] = int(name.lstrip(string.ascii_letters))
        self._set_offset(metaindex)
        label_delim = metaindex['label']+metaindex['delim']
        index = [f'{label_delim}{i+metaindex["offset"]:d}'
                 for i in range(data_length)]
        return self._check_index(index, data_length)

    def _check_index(self, index, data_length):
        """
        Return new index.

        Parameters
        ----------
        index : array-like[str]
            new index.
        data_length : TYPE
            Maximum item length in data.

        Raises
        ------
        IndexError
            Missmatch between len(index) and data_length.
        IndexError
            Items in new index mirror those already defined in self.index.

        Returns
        -------
        index : array-like[str]

        """
        if not pandas.api.types.is_list_like(index):
            index = [index]
        if len(index) != data_length:
            raise IndexError(f'missmatch between len(index) {len(index)} and '
                             f'maximum item item in data {data_length}')
        taken = [name in self.index for name in index]
        if np.array(taken).any():
            raise IndexError(f'{np.array(index)[taken]} '
                             f'already defined in self.index: {self.index}')
        return index
