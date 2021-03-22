"""Subclass pandas.DataFrame."""
import re
import string
from typing import Collection, Any

import pandas
import numpy as np

from nova.electromagnetic.metadata import MetaData
from nova.electromagnetic.metaframe import MetaFrame

# pylint: disable=too-many-ancestors


class Series(pandas.Series):
    """Provide series constructor methods."""

    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_expanddim(self):
        return DataFrame


class DataFrame(pandas.DataFrame):
    """
    Extend pandas.DataFrame.

    - Manage Frame metadata (metaarray, metaframe).
    - Add boolean methods (add_frame, drop_frame...).

    """

    def __init__(self,
                 data=None,
                 index: Collection[Any] = None,
                 columns: Collection[Any] = None,
                 attrs: dict[str, Collection[Any]] = None,
                 **metadata: dict[str, Collection[Any]]):
        super().__init__(data, index, columns)
        self.update_metadata(data, columns, attrs, metadata)
        self.update_index()
        self.update_columns()

    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series

    @property
    def metaattrs(self) -> list[str]:
        """Return metadata attrs."""
        return [attr for attr in self.attrs
                if isinstance(self.attrs[attr], MetaData)]

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

    def __getattr__(self, name):
        """Extend pandas.DataFrame.__getattr__. (frame.*)."""
        if name in self.attrs:
            return self.attrs[name]
        return super().__getattr__(name)

    def update_metadata(self, data, columns, attrs, metadata):
        """Update metadata. Set default and meta*.metadata."""
        self.extract_attrs(data, attrs)
        self.trim_columns(columns)
        self.update_available(data, columns)
        self.extract_metadata(metadata)
        self.match_columns()

    def extract_attrs(self, data, attrs):
        """Extract metaframe / metaarray from data / attrs."""
        if data is None:
            data = {}
        if attrs is None:
            attrs = {}
        if hasattr(data, 'attrs'):
            for attr in data.attrs:  # update metadata from data
                if isinstance(data.attrs[attr], MetaData):
                    self.attrs[attr] = data.attrs[attr]
        for attr in attrs:  # update from attrs (replacing data.attrs)
            if isinstance(attrs[attr], MetaData):
                self.attrs[attr] = attrs[attr]
        if not self.hasattrs('metaframe'):
            self.attrs['metaframe'] = MetaFrame()  # init metaframe

    def trim_columns(self, columns):
        """Trim metaframe required / additional to columns."""
        if columns:  # trim to columns
            required = [attr for attr in self.metaframe.required
                        if attr in columns]
            additional = [attr for attr in self.metaframe.additional
                          if attr in columns]
            available = [attr for attr in self.metaframe.available
                         if attr in columns]
            self.metaframe.metadata = {'Required': required,
                                       'Additional': additional,
                                       'Available': available}

    def update_available(self, data, columns):
        """Update metaframe.available."""
        try:
            data_columns = list(data)
        except TypeError:
            data_columns = []
        if columns is None:
            columns = []
        self.metaframe.metadata = {'available': data_columns+list(columns)}

    def extract_metadata(self, metadata):
        """Extract attributes from **metadata."""
        if metadata is None:
            metadata = {}
        if 'metadata' in metadata:
            metadata |= metadata.pop('metadata')
        meta, default = {attr: {} for attr in self.metaattrs}, {}
        for field in list(metadata):
            for attr in meta:
                if hasattr(getattr(self, attr), field.lower()):
                    meta[attr][field] = metadata.pop(field)
                    break
            else:
                if field in self.metaframe.default:
                    default[field] = metadata.pop(field)
        self.metaframe.default |= default
        if len(metadata) > 0:
            raise IndexError('unreconised attributes set in **metadata: '
                             f'{metadata}.')
        for attr in meta:
            getattr(self, attr).metadata |= meta[attr]
        if self.metaframe.columns:
            self.metaframe.metadata = {'available': self.metaframe.columns}

    def match_columns(self):
        """Intersect metaframe.required with self.columns if not empty."""
        if not self.columns.empty:
            required = [attr for attr in self.metaframe.required
                        if attr in self.columns]
            self.metaframe.metadata = {'Required': required}

    def update_index(self):
        """Reset index if self.index is unset."""
        if isinstance(self.index, pandas.RangeIndex):
            self['index'] = self._build_index(self)
            self.set_index('index', inplace=True)
            self.index.name = None

    def _set_offset(self, metatag):
        try:  # reverse search through frame index
            match = next(name for name in self.index[::-1]
                         if metatag['label'] in name)
            offset = re.sub(r'[a-zA-Z]', '', match)
            if isinstance(offset, str):
                offset = offset.replace(metatag['delim'], '')
                offset = offset.replace('_', '')
                offset = int(offset)
            offset += 1
        except (TypeError, StopIteration):  # unit index, label not present
            offset = 0
        metatag['offset'] = np.max([offset, metatag['offset']])

    def _build_index(self, data, **kwargs):
        index_length = self._index_length(data)
        # update metaframe tag defaults with kwargs
        metatag = {key: kwargs.get(key, self.metaframe.default[key])
                   for key in ['name', 'label', 'delim', 'offset']}
        if kwargs.get('name', metatag['name']):  # valid name
            name = metatag['name']
            if pandas.api.types.is_list_like(name) or index_length == 1:
                return self._check_index(name, index_length)
            if metatag['delim'] and metatag['delim'] in name:
                split_name = name.split(metatag['delim'])
                metatag['label'] = metatag['delim'].join(split_name[:-1])
                metatag['offset'] = int(split_name[-1])
            else:
                metatag['delim'] = ''
                metatag['label'] = name.rstrip(string.digits)
                metatag['offset'] = int(name.lstrip(string.ascii_letters))
        self._set_offset(metatag)
        label_delim = metatag['label']+metatag['delim']
        index = [f'{label_delim}{i+metatag["offset"]:d}'
                 for i in range(index_length)]
        return self._check_index(index, index_length)

    @staticmethod
    def _index_length(data):
        """
        Return maximum item length in data.

        Parameters
        ----------
        data : dict[str, Union[float, array-like]]

        Returns
        -------
        index_length : int
            Maximum item item in data.

        """
        try:
            index_length = np.max(
                [len(data[key]) for key in data
                 if pandas.api.types.is_list_like(data[key])])
        except ValueError:
            index_length = 1  # scalar input
        return index_length

    def _check_index(self, index, index_length):
        """
        Return new index.

        Parameters
        ----------
        index : array-like[str]
            new index.
        index_length : int
            Maximum item length in data.

        Raises
        ------
        IndexError
            Missmatch between len(index) and index_length.
        IndexError
            Items in new index mirror those already defined in self.index.

        Returns
        -------
        index : array-like[str]

        """
        if not pandas.api.types.is_list_like(index):
            index = [index]
        if len(index) != index_length:
            raise IndexError(f'missmatch between len(index) {len(index)} and '
                             f'maximum item item in data {index_length}')
        taken = [name in self.index for name in index]
        if np.array(taken).any():
            raise IndexError(f'{np.array(index)[taken]} '
                             f'already defined in self.index: {self.index}')
        return index

    def update_columns(self):
        """
        Format DataFrame columns.

            - required unset: raise error
            - additional unset: insert default
            - isnan: insert default
        """
        if self.columns.empty:
            return
        with self.metaframe.setlock(None):
            columns = self.columns.to_list()
            # check required
            required_unset = [attr not in columns
                              for attr in self.metaframe.required]
            if np.array(required_unset).any():
                unset = np.array(self.metaframe.required)[required_unset]
                raise IndexError(f'required attributes missing {unset}')
            # fill nan
            isnan = [pandas.isna(self.loc[:, attr]).any() for attr in columns]
            if np.array(isnan).any():
                nan = np.array(columns)[isnan]
                for attr in nan:
                    if attr in self.metaframe.default:
                        index = pandas.isna(self.loc[:, attr])
                        self.loc[index, attr] = self.metaframe.default[attr]
            # extend additional
            additional = [attr for attr in columns
                          if attr not in self.metaframe.columns]
            if additional:
                self.metaframe.metadata = {'additional': additional}
            # set defaults
            additional_unset = [attr not in columns
                                for attr in self.metaframe.additional]
            if np.array(additional_unset).any() and not self.index.empty:
                unset = np.array(self.metaframe.additional)[additional_unset]
                for attr in unset:
                    self.loc[:, attr] = self.metaframe.default[attr]
                turn_set = np.array([attr in self.columns
                                     for attr in ['It', 'Nt']])
                if 'Ic' in unset and turn_set.all():
                    self.loc[:, 'Ic'] = self.loc[:, 'It'] / self.loc[:, 'Nt']

    @staticmethod
    def isframe(obj, frame=True):
        """
        Return isinstance(arg[0], Frame | DataFrame) flag.

        Parameters
        ----------
        obj : Any
            Input.
        frame : bool, optional
            Accept pandas.DataFrame. The default is True.

        Returns
        -------
        isframe: bool
            Frame / pandas.DataFrame isinstance flag.

        """
        if isinstance(obj, DataFrame):
            return True
        if isinstance(obj, pandas.DataFrame) and frame:
            return True
        return False

    def hasattrs(self, attr):
        """Return True if attr in self.attrs."""
        return attr in self.attrs

    #@profile
    def in_field(self, col, field):
        """Return Ture if col in metaframe.{field} and hasattr(self, field)."""
        try:
            return col in self.attrs[field].columns
        except (KeyError, TypeError):
            return False
        #if not isinstance(col, str):
        #    return False
        #if self.hasattrs('metaframe') and self.hasattrs(field):
        #    if hasattr(self.attrs[field], 'columns'):
        #        return col in self.attrs[field].columns
        #return False

    def assert_in_field(self, col, field):
        """Check for col in metaframe.{field}, raise error if not found."""
        try:
            self.in_field(col, field)
        except AssertionError as in_field_assert:
            raise AssertionError(
                f'\'{col}\' not specified in metaframe.subspace '
                f'{self.metaframe.subspace}') from in_field_assert

    def format_value(self, col, value):
        """Return vector with dtype as type(metaframe.default[col])."""
        if not self.hasattrs('metaframe') or col == 'link':
            return value
        try:
            dtype = type(self.metaframe.default[col])
        except (KeyError, TypeError):  # no default type, isinstance(col, list)
            return value
        try:
            if pandas.api.types.is_list_like(value):
                return np.array(value, dtype)
            return dtype(value)
        except (ValueError, TypeError):  # NaN conversion error
            return value


if __name__ == '__main__':

    dataframe = DataFrame(Required=['x', 'dCoil'], Additional=['Ic'],
                          Subspace=[])
    print(dataframe.metaframe.required)
    print(dataframe.metaframe.additional)
    print(dataframe)
