"""Subclass pandas.DataFrame."""
import re
import string
from typing import Collection, Any

import pandas
import numpy as np
import shapely

from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.indexer import Indexer
from nova.electromagnetic.energize import Energize
from nova.electromagnetic.multipoint import MultiPoint
from nova.electromagnetic.polygon import Polygon

# pylint: disable=too-many-ancestors


class SubSpaceError(IndexError):
    """Prevent direct access to frame's subspace variables."""

    def __init__(self, name, col):
        super().__init__(
            f'{name} access is restricted for subspace attributes. '
            f'Use frame.subspace.{name}[:, {col}] = *.\n\n'
            'Lock may be overridden via the following context manager '
            'but subspace will still overwrite (Cavieat Usor):\n'
            'with frame.metaframe.setlock(None):\n'
            f'    frame.{name}[:, {col}] = *')


class IndexerMixin:
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        col = self.obj._get_col(key)
        value = self.obj._format_value(col, value)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace') is True:
                raise SubSpaceError(self.name, col)
        if self.obj.in_field(col, 'energize'):
            if self.obj.metaframe.lock('energize') is False:
                return self.obj.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Refresh subspace items prior to return."""
        col = self.obj._get_col(key)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace') is True:
                self.obj.set_frame(col)
        if self.obj.in_field(col, 'energize'):
            print(self.obj.metaframe.lock('energize'))
            if self.obj.metaframe.lock('energize') is not None:
                return self.obj.energize._get_item(super(), key)
        return super().__getitem__(key)


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
        self.attrs['indexer'] = Indexer(IndexerMixin)
        self.attrs['metaframe'] = MetaFrame()
        self.extract_attrs(data, attrs)
        self.update_metadata(metadata)
        self.update_index()
        self.attrs['energize'] = Energize(self)
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
        self._validate_metadata()

    def update_metadata(self, metadata):
        """Update metadata. Set default and meta*.metadata."""
        columns = not self.columns.empty
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
        if columns:
            self._update_metaframe()

    def _update_metaframe(self):
        """Update metaframe to match self.columns if self.columns not empty."""
        metadata = {}
        for attribute in ['required', 'additional']:
            metadata[attribute.capitalize()] = [
                attr for attr in getattr(self.metaframe, attribute)
                if attr in self.columns]
        self.metadata = metadata

    def _validate_metadata(self):
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
            self._format_columns()

    def _format_columns(self):
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
                turn_set = np.array([attr in self.columns
                                     for attr in ['It', 'Nt']])
                if 'Ic' in unset and turn_set.all():
                    self.loc[:, 'Ic'] = self.loc[:, 'It'] / self.loc[:, 'Nt']
            # update nan
            print(self.metaframe.lock())
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
        Build frame from *args, **kwargs and concatenate with DataFrame.

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
        self._concat(insert, iloc=iloc)

    def _concat(self, insert, iloc=None, sort=False):
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
        insert = DataFrame(data, index=index, attrs=self.attrs)
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
        for attr, arg in zip(self.metaframe.required, args):
            data[attr] = np.array(arg, dtype=float)  # add required arguments
        additional = []
        for attr in list(kwargs.keys()):
            if attr in self.metaframe.default:
                data[attr] = kwargs.pop(attr)  # add keyword arguments
                if attr not in self.metaframe.additional:
                    additional.append(attr)
        if len(kwargs) > 0:  # ckeck for unset kwargs
            unset_kwargs = np.array(list(kwargs.keys()))
            default = {key: '_default_value_' for key in unset_kwargs}
            raise IndexError(
                f'unset kwargs: {unset_kwargs}\n'
                'enter default value in self.metaframe.defaults\n'
                f'set as self.metaframe.meatadata = {{default: {default}}}')
        if 'It' in data and 'Ic' not in data:  # patch line current
            data['Ic'] = \
                data['It'] / data.get('Nt', self.metaframe.default['Nt'])
        if len(additional) > 0:  # extend aditional arguments
            self.metaframe.metadata = {'additional': additional}
        additional_unset = [attr for attr in self.metaframe.additional
                            if attr not in data]
        for attr in additional_unset:  # set default attributes
            data[attr] = self.metaframe.default[attr]
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

    def __repr__(self):
        """Propagate frame subspace variables prior to display."""
        self.update_frame()
        return super().__repr__()

    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series

    @property
    def loc(self):
        """Extend DataFrame.loc, restrict subspace access."""
        return self.indexer.loc("loc", self)

    @property
    def iloc(self):
        """Extend DataFrame.iloc, restrict subspace access."""
        return self.indexer.iloc("iloc", self)

    @property
    def at(self):
        """Extend DataFrame.at, restrict subspace access."""
        return self.indexer.at("at", self)

    @property
    def iat(self):
        """Extend DataFrame.iat, restrict subspace access."""
        return self.indexer.iat("iat", self)

    @staticmethod
    def isframe(obj, dataframe=True):
        """
        Return isinstance(arg[0], Frame | DataFrame) flag.

        Parameters
        ----------
        obj : Any
            Input.
        dataframe : bool, optional
            Accept pandas.DataFrame. The default is True.

        Returns
        -------
        isframe: bool
            Frame / pandas.DataFrame isinstance flag.

        """
        if isinstance(obj, DataFrame):
            return True
        if isinstance(obj, pandas.DataFrame) and dataframe:
            return True
        return False

    def _get_col(self, key):
        """Return column label."""
        if isinstance(key, tuple):
            col = key[-1]
        else:
            col = key
        if isinstance(col, int):
            col = self.columns[col]
        return col

    def _hasattr(self, attr):
        """Return True if attr in self.attrs."""
        return attr in self.attrs

    def in_field(self, col, field):
        """Return Ture if col in metaframe.{field} and hasattr(self, field)."""
        if not isinstance(col, str):
            return False
        if self._hasattr('metaframe') and self._hasattr(field):
            if hasattr(self.attrs[field], 'columns'):
                return col in self.attrs[field].columns
        #if self._hasattr('metaframe') and field == 'subspace':
        return False

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if self._hasattr('subspace'):
            for col in self.subspace:
                self.set_frame(col)

    def set_frame(self, col):
        """Inflate subspace variable and setattr in frame."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(True, 'subspace'):
            value = getattr(self, col).to_numpy()
            if hasattr(self, 'subref'):  # inflate
                value = value[self.subref]
        with self.metaframe.setlock(None, 'subspace'):
            super().__setitem__(col, value)

    def get_frame(self, col):
        """Return inflated subspace variable."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(False, 'subspace'):
            return super().__getitem__(col)

    def assert_in_field(self, col, field):
        """Check for col in metaframe.{field}, raise error if not found."""
        try:
            self.in_field(col, field)
        except AssertionError as in_field_assert:
            raise AssertionError(
                f'\'{col}\' not specified in metaframe.subspace '
                f'{self.metaframe.subspace}') from in_field_assert

    def __getattr__(self, name):
        """Extend DataFrame.__getattr__. (frame.*)."""
        if name in self.attrs:
            return self.attrs[name]
        return super().__getattr__(name)

    def __getitem__(self, key):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        col = self._get_col(key)
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__getattr__(col)
            if self.metaframe.lock('subspace') is False:
                self.set_frame(col)
        if self.in_field(col, 'energize'):
            if self.metaframe.lock('energize') is not None:
                return self.energize._get_item(super(), key)
        return super().__getitem__(col)

    def __setitem__(self, key, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        col = self._get_col(key)
        value = self._format_value(col, value)
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__setitem__(key, value)
            if self.metaframe.lock('subspace') is False:
                raise SubSpaceError('setitem', col)
        if self.in_field(col, 'energize'):
            if self.metaframe.lock('energize') is False:
                return self.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)

    def _format_value(self, col, value):
        if not pandas.api.types.is_numeric_dtype(type(value)) \
                or not self._hasattr('metaframe'):
            return value
        try:
            dtype = type(self.metaframe.default[col])
        except KeyError:  # no default type
            return value
        try:
            if pandas.api.types.is_list_like(value):
                return np.array(value, dtype)
            return dtype(value)
        except (ValueError, TypeError):  # NaN conversion error
            return value


if __name__ == '__main__':

    dataframe = DataFrame({'x': [1, 2, 3], 'z': 6.7})
    dataframe.add_frame(1, 3, It=5)
    dataframe.loc[:, 'x'] = 7
    print(dataframe)
