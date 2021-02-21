"""Extend pandas.DataFrame to manage coil and subcoil data."""

from dataclasses import dataclass, field, fields
import re
import string
from typing import Optional, Collection, Any, Union

import pandas
import numpy as np
import shapely

from nova.electromagnetic.multipoint import MultiPoint
from nova.electromagnetic.polygon import Polygon
from nova.electromagnetic.metadata import MetaData
from nova.electromagnetic.array import MetaArray, Array

# pylint:disable=unsubscriptable-object
# pylint: disable=too-many-ancestors


@dataclass
class MetaFrame(MetaData):
    """Manage Frame metadata - accessed via Frame['attrs']."""

    required: list[str] = field(default_factory=lambda: ['x', 'z'])
    additional: list[str] = field(default_factory=lambda: [])
    default: dict[str, Union[float, str, bool, None]] = field(
        repr=False, default_factory=lambda: {
            'dCoil': 0., 'nx': 1, 'nz': 1, 'Nt': 1., 'Nf': 1,
            'rms': 0., 'dx': 0., 'dz': 0., 'dA': 0.,
            'dl': 0.1, 'dt': 0.1, 'dl_x': 0., 'dl_z': 0.,
            'm': '', 'R': 0.,  'rho': 0.,
            'turn_fraction': 1., 'skin_fraction': 1.,
            'cross_section': 'rectangle', 'turn_section': 'rectangle',
            'patch': None, 'poly': None, 'coil': '', 'part': '',
            'subindex': None, 'material': '', 'mpc': '',
            'active': True, 'optimize': False, 'plasma': False,
            'feedback': False, 'acloss': False,
            'Ic': 0., 'It': 0., 'Psi': 0., 'Bx': 0., 'Bz': 0., 'B': 0.})
    frame: dict[str, Union[str, bool]] = field(
        repr=False, default_factory=lambda: {
            'name': '', 'label': 'Coil', 'delim': ''})

    def validate(self):
        """
        Extend MetaData.validate.

            - Exclude duplicate values from self.required in self.additional.
            - Check that all additional attributes have a default value.

        """
        MetaData.validate(self)
        # exculde duplicate values
        self.additional = [attr for attr in self.additional
                           if attr not in self.required]
        # check unset defaults
        unset = np.array([attr not in self.default
                          for attr in self.additional])
        if unset.any():
            raise ValueError('Default value not set for additional attributes '
                             f'{np.array(self.additional)[unset]}')
        # block frame extension
        frame_default = next(field.default_factory() for field in fields(self)
                             if field.name == 'frame')
        extend = np.array([attr not in frame_default for attr in self.frame])
        if extend.any():
            raise IndexError('additional attributes passed to frame field '
                             f'{np.array(list(self.frame.keys()))[extend]}')

    @property
    def required_number(self):
        """Return number of required arguments."""
        return len(self.required)

    @property
    def columns(self):
        """Return metaframe columns."""
        return self.required + self.additional


class Series(pandas.Series):
    """Provide series constructor methods."""

    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_expanddim(self):
        return Frame


class Frame(pandas.DataFrame):
    """
    Extends Pandas.DataFrame.

    Inspiration for DataFrame inheritance taken from GeoPandas
    https://github.com/geopandas.
    """

    _attributes = {'multipoint': MultiPoint, 'polygon': Polygon}

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 attrs: Optional[dict] = None,
                 metadata: Optional[dict] = None,
                 **default: Optional[dict]):
        super().__init__(data, index)
        self.update_attrs(attrs)
        self.update_attributes()
        self.update_metadata(metadata)
        self.update_default(default)
        self.update_index()
        self.multipoint.link()

    @property
    def _constructor(self):
        return Frame

    @property
    def _constructor_sliced(self):
        return Series

    @property
    def metaframe(self):
        """Return metaframe instance."""
        return self.attrs['metaframe']

    @property
    def multipoint(self):
        """Return multipoint instance."""
        return self.attrs['multipoint']

    @property
    def polygon(self):
        """Return section instance."""
        return self.attrs['polygon']

    def update_attrs(self, attrs=None):
        """Update frame attrs and initialize."""
        if attrs is not None:
            self.attrs |= attrs
        if 'metaframe' not in self.attrs:
            self.attrs['metaframe'] = MetaFrame()
            self.attrs['metadata'] = ['metaframe']

    def update_attributes(self):
        """Update Frame attributes."""
        for attr in self._attributes:
            self.attrs[attr] = self._attributes[attr](self)

    def validate_frame(self):
        """Validate required and additional attributes in Frame."""
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

    def update_metadata(self, metadata):
        """Update metadata."""
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @property
    def metadata(self):
        """Manage Frame metadata via the MetaFrame class."""
        metadata = {}
        for attribute in self.attrs['metadata']:
            metadata |= getattr(self, attribute).metadata
        return metadata

    @metadata.setter
    def metadata(self, metadata):
        for attribute in self.attrs['metadata']:
            getattr(self, attribute).metadata = metadata
            getattr(self, attribute).validate()

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

    def add_frame(self, *args, iloc=None, **kwargs):
        """
        Build frame from *args, **kwargs and concatenate with Frame.

        Parameters
        ----------
        *args : Union[float, array-like]
            Required arguments listed in self.metaframe.required.
        iloc : int, optional
            DESCRIPTION. The default is None.
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
        Frame.__init__(self, dataframe, attrs=self.attrs)
        #self.rebuild_CoilArray()  # rebuild fast index

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
        insert = Frame(data, index=index, attrs=self.attrs)
        insert.polygon.generate()
        insert.multipoint.link()
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
            Output argument number len(args) != self.metaframe.required_number.

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
        if len(args) != self.metaframe.required_number:
            raise IndexError('incorrect output argument number: '
                             f'{len(args)} != '
                             f'{self.metaframe.required_number}\n')
        return args, kwargs

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
        if isinstance(obj, Frame):
            return True
        if isinstance(obj, pandas.DataFrame) and dataframe:
            return True
        return False

    def _build_data(self, *args, **kwargs):
        """Return data dict built from *args and **kwargs."""
        data = {}  # python 3.6+ assumes dict is insertion ordered
        for key, arg in zip(self.metaframe.required, args):
            data[key] = np.array(arg, dtype=float)  # add required arguments
        current_label = self._current_label(**kwargs)
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
        self._propogate_current(current_label, data)
        unset = [key not in self.metaframe.frame for key in kwargs]
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
        metaindex = self.metaframe.frame | kwargs
        metaindex['offset'] = 0
        if kwargs.get('name', ''):  # valid name
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

    def _current_label(self, **kwargs):
        """Return current label, Ic or It."""
        current_label = None
        if 'Ic' in self.metaframe.required or 'Ic' in kwargs:
            current_label = 'Ic'
        elif 'It' in self.metaframe.required or 'It' in kwargs:
            current_label = 'It'
        return current_label

    @staticmethod
    def _propogate_current(current_label, data):
        """
        "Propogate current data, Ic->It or It->Ic.

        Parameters
        ----------
        current_label : str
            Current label, Ic or It.
        data : Union[pandas.DataFrame, dict]
            Current / turn data.

        Returns
        -------
        None.

        """
        if current_label == 'Ic':
            data['It'] = data['Ic'] * data['Nt']
        elif current_label == 'It':
            data['Ic'] = data['It'] / data['Nt']

    def drop_frame(self, index=None):
        """Drop frame(s)."""
        if index is None:
            index = self.index
        self.drop_mpc(index)
        self.drop(index, inplace=True)
        #self.rebuild_CoilArray()

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
        # TODO regenerate Biot arrays...


class FrameArray(Frame, Array):
    """Extends Frame with fast attribute access provided by Array."""

    @property
    def metaarray(self):
        """Return metaarray instance."""
        return self.attrs['metaarray']

    def update_attrs(self, attrs=None):
        """Extend Frame.update_attrs with metaarray instance."""
        super().update_attrs(attrs)
        if 'metaarray' not in self.attrs:
            self.attrs['metaarray'] = MetaArray()
            self.attrs['metadata'].append('metaarray')

    def validate_array(self):
        """Validate metaarray."""
        unset = [attr not in self.metaframe.columns
                 for attr in self.metaarray.array]
        if np.array(unset).any():
            raise IndexError(
                f'metaarray attributes {np.array(self.metaarray.array)[unset]}'
                f' already set in metaframe.required {self.metaframe.required}'
                f'or metaframe.additional {self.metaframe.additional}')


if __name__ == '__main__':

    frame = Frame(mpc=True)
    # implement antiattribute (exclude) field in metaframe

    frame.add_frame(4, [5, 7, 12], name='coil1')
    print(frame)

    #framearray = FrameArray(frame)
    #print(framearray)
