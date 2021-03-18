"""Subclass pandas.DataFrame."""
import re
import string
from typing import Collection, Any

import pandas
import numpy as np
import shapely

from nova.electromagnetic.metadata import MetaData
from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.metamethod import MetaMethod
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
            if self.obj.metaframe.lock('energize') is False:
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
        self.update_metadata(data, columns, attrs, metadata)
        self.update_attrs()
        self.update_index()
        self.update_columns()
        self.init_attrs()

    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series

    def __repr__(self):
        """Propagate frame subspace variables prior to display."""
        self.update_frame()
        return super().__repr__()

    def update_metadata(self, data, columns, attrs, metadata):
        """Update metadata. Set default and meta*.metadata."""
        self._extract_attrs(data, attrs)
        self._trim_columns(columns)
        self._update_avalible(data, columns)
        self._extract_metadata(metadata)
        self._match_columns()

    def _extract_attrs(self, data, attrs):
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
        if not self._hasattr('metaframe'):
            self.attrs['metaframe'] = MetaFrame()  # init metaframe

    def _trim_columns(self, columns):
        """Trim metaframe required / additional to columns."""
        if columns:  # trim to columns
            required = [attr for attr in self.metaframe.required
                        if attr in columns]
            additional = [attr for attr in self.metaframe.additional
                          if attr in columns]
            avalible = [attr for attr in self.metaframe.avalible
                        if attr in columns]
            self.metaframe.metadata = {'Required': required,
                                       'Additional': additional,
                                       'Avalible': avalible}

    def _update_avalible(self, data, columns):
        """Update metaframe.avalible."""
        try:
            data_columns = list(data)
        except TypeError:
            data_columns = []
        if columns is None:
            columns = []
        self.metaframe.metadata = {'avalible': data_columns+list(columns)}

    def _extract_metadata(self, metadata):
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

    def _match_columns(self):
        """Intersect metaframe.required with self.columns if not empty."""
        if not self.columns.empty:
            required = [attr for attr in self.metaframe.required
                        if attr in self.columns]
            self.metaframe.metadata = {'Required': required}

    def update_attrs(self):
        """Extract frame attrs from data and update."""
        self.attrs['indexer'] = Indexer(IndexerMixin)
        self.attrs['energize'] = Energize(self)
        self.attrs['multipoint'] = MultiPoint(self)
        self.attrs['polygon'] = Polygon(self)

    def init_attrs(self):
        """Initialize attributes (metamethods)."""
        for attr in self.attrs:
            attribute = self.attrs[attr]
            if isinstance(attribute, MetaMethod):
                if attribute.generate:
                    attribute.initialize()

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if self._hasattr('subspace'):
            for col in [col for col in self.subspace if col in self]:
                self.set_frame(col)

    def set_frame(self, col):
        """Inflate subspace variable and setattr in frame."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(True, 'subspace'):
            value = getattr(self, col).to_numpy()
        with self.metaframe.setlock(None):
            if hasattr(self, 'subref'):  # inflate
                value = value[self.subref]
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

    def update_columns(self):
        """
        Format DataFrame columns.

            - required unset: raise error
            - additional unset: insert default
            - isnan: insert default
        """
        if self.columns.empty:
            return
            #self.__init__(DataFrame(columns=self.metaframe.columns,
            #                        index=self.index, attrs=self.attrs))
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
        frame = pandas.DataFrame(self)
        if not isinstance(insert, pandas.DataFrame):
            insert = pandas.DataFrame(insert)
        if iloc is None:  # append
            frames = [frame, insert]
        else:  # insert
            frames = [frame.iloc[:iloc, :], insert, frame.iloc[iloc:, :]]
        frame = pandas.concat(frames, sort=sort)  # concatenate
        self.__init__(frame, attrs=self.attrs)

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
        return DataFrame(data, index=index, attrs=self.attrs)

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
        if self.isframe(args[0], frame=True) and len(args) == 1:
            frame = args[0]
            missing = [arg not in frame for arg in self.metaframe.required]
            if np.array(missing).any():
                required = np.array(self.metaframe.required)[missing]
                raise ValueError(f'required arguments {required} '
                                 'not specified in frame '
                                 f'{frame.columns}')
            args = [frame.loc[:, col] for col in self.metaframe.required]
            if not isinstance(frame.index, pandas.RangeIndex):
                kwargs['name'] = frame.index
            kwargs |= {col: frame.loc[:, col] for col in
                       self.metaframe.additional if col in frame}
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
        attrs = list(data.keys()) + list(kwargs.keys())  # record passed attrs
        for attr in self.metaframe.additional:  # set additional to default
            data[attr] = self.metaframe.default[attr]
        additional = []
        for attr in list(kwargs.keys()):
            if attr in self.metaframe.default:
                data[attr] = kwargs.pop(attr)  # add keyword arguments
                if attr not in self.metaframe.additional:
                    additional.append(attr)
        if 'It' in attrs and 'Ic' not in attrs:  # patch line current
            data['Ic'] = \
                data['It'] / data.get('Nt', self.metaframe.default['Nt'])
        if len(additional) > 0:  # extend aditional arguments
            self.metaframe.metadata = {'additional': additional}
        if len(kwargs) > 0:  # ckeck for unset kwargs
            unset_kwargs = np.array(list(kwargs.keys()))
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
        return False

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
            if self.metaframe.lock('energize') is False:
                return self.energize._get_item(super(), key)
        return super().__getitem__(key)

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

    dataframe = DataFrame(Required=['x', 'dCoil'], Additional=['Ic'],
                          Subspace=[])
    print(dataframe.metaframe.required)
    print(dataframe.metaframe.additional)
    print(dataframe.metaframe.subspace)
