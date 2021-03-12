"""Extend pandas.DataFrame to manage coil and subcoil data."""

import re
import string
from typing import Optional, Collection, Any

import pandas
import numpy as np
import shapely

from nova.electromagnetic.indexer import IndexerMixin
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.current import Current
from nova.electromagnetic.multipoint import MultiPoint
from nova.electromagnetic.polygon import Polygon


# pylint: disable=too-many-ancestors


class SuperFrame(DataFrame):
    """
    Extend DataFrame. Frame superclass.

    - Manage Frame metadata (metaarray, metaframe).
    - Add boolean methods (add_frame, drop_frame...).

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: Optional[dict[str, Collection[Any]]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, mixin=IndexerMixin)
        self.extract_attrs(data, attrs)
        self.update_metadata(metadata)
        self.update_index()
        self.update_attrs()

    def update_attrs(self):
        """Compose additional attributes."""
        self.attrs['current'] = Current(self)
        self.attrs['multipoint'] = MultiPoint(self)
        self.attrs['polygon'] = Polygon(self)

    def extract_attrs(self, data, attrs):
        """Extract frame attrs from data and update."""
        if hasattr(data, 'attrs'):
            self.attrs |= data.attrs
        if attrs is None:
            attrs = {}
        self.attrs |= attrs

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

    def update_metadata(self, metadata):
        """Update metadata."""
        columns = not self.columns.empty
        self.extract_metadata(metadata)
        self.reset_metaframe(columns)

    def extract_metadata(self, metadata):
        """Extract metadata. Set default and meta*.metadata."""
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
        self.metaframe.default |= default
        if len(metadata) > 0:
            raise IndexError('unreconised attributes set in **metadata: '
                             f'{metadata}.')
        for attr in framearray:
            getattr(self, attr).metadata |= framearray[attr]

    def reset_metaframe(self, columns):
        """Revise metaframe to match self.columns if self.columns not empty."""
        metadata = {}
        if columns:
            for attribute in ['required', 'additional']:
                metadata[attribute.capitalize()] = [
                    attr for attr in getattr(self.metaframe, attribute)
                    if attr in self.columns]
            self.metadata = metadata

    def validate_metadata(self):
        """Validate required and additional attributes in FrameArray."""
        # check for additional attributes in metaarray.array
        if 'metaarray' in self.attrs:
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
        with self.metaframe.setlock('subspace', None):
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
        data_length = self._data_length(data)
        # update metaframe tag defaults with kwargs
        metatag = {key: kwargs.get(key, self.metaframe.default[key])
                   for key in ['name', 'label', 'delim', 'offset']}
        if kwargs.get('name', metatag['name']):  # valid name
            name = metatag['name']
            if pandas.api.types.is_list_like(name) or data_length == 1:
                return self._check_index(name, data_length)
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

    def add_frame(self, *args, iloc=None, **kwargs):
        """
        Build frame from *args, **kwargs and concatenate with Frame.

        Parameters
        ----------
        *args : Union[float, array-like]
            Required arguments listed in self.metaframe.required.
        iloc : int, optional
            Row locater for inserted coil. The default is None (-1).
        **kwargs : dict[str, Union[float, array-like]]
            Optional keyword as arguments listed in self.metaframe.additional.

        Returns
        -------
        index : pandas.Index
            built frame.index.

        """
        self.metadata = kwargs.pop('metadata', {})
        insert = self._build_frame(*args, **kwargs)
        self.concat(insert, iloc=iloc)

    def concat(self, insert, iloc=None, sort=False):
        """Concatenate insert with DataFrame(self)."""
        dataframe = pandas.DataFrame(self)
        if not isinstance(insert, pandas.DataFrame):
            insert = pandas.DataFrame(insert)
        if iloc is None:  # append
            dataframes = [dataframe, insert]
        else:  # insert
            dataframes = [dataframe.iloc[:iloc, :],
                          insert,
                          dataframe.iloc[iloc:, :]]
        dataframe = pandas.concat(dataframes, sort=sort)  # concatenate
        self.__init__(dataframe, attrs=self.attrs)

    def _build_frame(self, *args, **kwargs):
        """
        Return Frame constructed from required and optional input.

        Parameters
        ----------
        *args : Union[Frame, pandas.DataFrame, array-like]
            Required arguments listed in self.metaframe.required.
        **kwargs : dict[str, Union[float, array-like, str]]
            Optional keyword arguments listed in self.metaframe.additional.

        Returns
        -------
        insert : pandas.DataFrame

        """
        args, kwargs = self._extract(*args, **kwargs)
        data = self._build_data(*args, **kwargs)
        index = self._build_index(data, **kwargs)
        insert = SuperFrame(data, index=index, attrs=self.attrs)
        return pandas.DataFrame(insert)

    def _extract(self, *args, **kwargs):
        """
        Return *args and **kwargs with data extracted from frame.

        If args[0] is a *frame, replace *args and update **kwargs.
        Else pass *args, **kwargs.

        Parameters
        ----------
        *args : Union[Frame, DataFrame, list[float], list[array-like]]
            Arguments.
        **kwargs : dict[str, Union[float, array-like]]
            Keyword arguments.

        Raises
        ------
        IndexError
            Input argument length must be greater than 0.
        ValueError
            Required arguments not present in *frame.
        IndexError
            Output argument number len(args) != len(self.metaframe.required).

        Returns
        -------
        args : list[Any]
            Return argument list, replaced input arg[0] is *frame.
        kwargs : dict[str, Any]
            Return keyword arquments, updated if input arg[0] is *frame.

        """
        if len(args) == 0:
            raise IndexError('len(args) == 0, argument number must be > 0')
        if self.isframe(args[0], dataframe=True) and len(args) == 1:
            dataframe = args[0]
            missing = [arg not in dataframe for arg in self.metaframe.required]
            if np.array(missing).any():
                required = np.array(self.metaframe.required)[missing]
                raise ValueError(f'required arguments {required} '
                                 'not specified in dataframe '
                                 f'{dataframe.columns}')
            args = [dataframe.loc[:, col] for col in self.metaframe.required]
            if not isinstance(dataframe.index, pandas.RangeIndex):
                kwargs['name'] = dataframe.index
            kwargs |= {col: dataframe.loc[:, col] for col in
                       self.metaframe.additional if col in dataframe}
        if len(args) != len(self.metaframe.required):
            raise IndexError(
                'incorrect required argument number (*args)): '
                f'{len(args)} != {len(self.metaframe.required)}\n'
                f'required *args: {self.metaframe.required}\n'
                f'additional **kwargs: {self.metaframe.additional}')
        return args, kwargs

    def _build_data(self, *args, **kwargs):
        """Return data dict built from *args and **kwargs."""
        data = {}  # python 3.6+ assumes dict is insertion ordered
        for key, arg in zip(self.metaframe.required, args):
            data[key] = np.array(arg, dtype=float)  # add required arguments
        # current_label = self._current_label(**kwargs)
        for key in self.metaframe.additional:
            if key in kwargs:
                data[key] = kwargs.pop(key)
            else:
                data[key] = self.metaframe.default[key]
        additional = []
        for key in kwargs:
            if key in self.metaframe.default:
                additional.append(key)
                data[key] = kwargs[key]
        for key in additional:
            del kwargs[key]
        if len(additional) > 0:  # extend aditional arguments
            self.metaframe.metadata = {'additional': additional}
        # self._propogate_current(current_label, data)
        unset = [key for key in kwargs]
        if np.array(unset).any():
            unset_kwargs = np.array(list(kwargs.keys()))[unset]
            default = {key: '_default_value_' for key in unset_kwargs}
            raise IndexError(
                f'unset kwargs: {unset_kwargs}\n'
                'enter default value in self.metaframe.defaults\n'
                f'set as self.metaframe.meatadata = {{default: {default}}}')
        return data

    @staticmethod
    def _data_length(data):
        """
        Return maximum item length in data.

        Parameters
        ----------
        data : dict[str, Union[float, array-like]]

        Returns
        -------
        data_length : int
            Maximum item item in data.

        """
        try:
            data_length = np.max(
                [len(data[key]) for key in data
                 if pandas.api.types.is_list_like(data[key])])
        except ValueError:
            data_length = 1  # scalar input
        return data_length

    def drop_frame(self, index=None):
        """Drop frame(s)."""
        if index is None:
            index = self.index
        self.multipoint.drop(index)
        self.drop(index, inplace=True)

    def translate(self, index=None, xoffset=0, zoffset=0):
        """Translate coil(s)."""
        if index is None:
            index = self.index
        elif not pandas.api.types.is_list_like(index):
            index = [index]
        if xoffset != 0:
            self.loc[index, 'x'] += xoffset
        if zoffset != 0:
            self.loc[index, 'z'] += zoffset
        for name in index:
            self.loc[name, 'poly'] = \
                shapely.affinity.translate(self.loc[name, 'poly'],
                                           xoff=xoffset, yoff=zoffset)
            self.loc[name, 'patch'] = None  # re-generate coil patch


if __name__ == '__main__':

    superframe = SuperFrame(Required=['x', 'z'], Additional=['Ic'])

    superframe.add_frame(4.987878, range(3), link=True)
    superframe.add_frame(5, range(2), link=False)
    superframe.add_frame(7, range(4), link=True)

    def set_current():
        superframe.Ic = np.random.rand(len(superframe))

    print(superframe)
