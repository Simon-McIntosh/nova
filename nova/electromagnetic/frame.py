"""Extend pandas.DataFrame to manage coil and subcoil data."""

from dataclasses import dataclass, field, fields
import re
import string
from typing import Optional, Collection, Any, Union

import pandas
import numpy as np
import shapely

from nova.electromagnetic.metadata import MetaData
from nova.electromagnetic.array import Array
from nova.electromagnetic.polygen import polygen, root_mean_square

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
            'rms': 0., 'dx': 0., 'dz': 0., 'dA': 0., 'dl_x': 0., 'dl_z': 0.,
            'm': '', 'R': 0.,  'rho': 0.,
            'turn_fraction': 1., 'skin_fraction': 1.,
            'cross_section': 'rectangle', 'turn_section': 'rectangle',
            'patch': None, 'polygon': None,
            'coil': '', 'part': '',
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
        # block frame mutable keys
        frame_default = next(field.default_factory() for field in fields(self)
                             if field.name == 'frame')
        mutate = np.array([attr not in frame_default for attr in self.frame])
        if mutate.any():
            raise IndexError('non-default attributes set in frame field '
                             f'{np.array(list(self.frame.keys()))[mutate]}')

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


class Frame(Array, pandas.DataFrame):
    """
    Frame instance inherits from Pandas DataFrame and CoilArray.

    Inspiration for DataFrame inheritance taken from GeoPandas
    https://github.com/geopandas.
    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: Optional[dict] = {},
                 metadata: Optional[dict] = {},
                 **default: Optional[dict]):
        super().__init__(data, index, columns)
        self.update_attrs(attrs)
        self.metadata = metadata
        self.metaframe.default |= default
        self.update_index()
        self.link_frame()

    @property
    def _constructor(self):
        return Frame

    @property
    def _constructor_sliced(self):
        return Series

    def update_attrs(self, attrs):
        """Update metaframe and metaarray attributes."""
        self.attrs |= attrs
        for instance in ['frame', 'array']:
            if hasattr(self, f'_init_{instance}'):
                getattr(self, f'_init_{instance}')()

    def _init_frame(self):
        if 'metaframe' not in self.attrs:
            self.attrs['metaframe'] = MetaFrame()

    def validate_metadata(self):
        """Validate metaframe and metaarray."""
        for instance in ['frame', 'array']:
            if hasattr(self, f'validate_{instance}'):
                getattr(self, f'validate_{instance}')()

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

    @property
    def metaframe(self):
        """Return metaframe."""
        return self.attrs['metaframe']

    @property
    def metadata(self):
        """Manage Frame metadata via the MetaFrame class."""
        metadata = {}
        for instance in ['metaframe', 'metaarray']:
            try:
                metadata |= getattr(self, instance).metadata
            except AttributeError:
                pass
        return metadata

    @metadata.setter
    def metadata(self, metadata):
        for instance in ['metaframe', 'metaarray']:
            try:
                getattr(self, instance).metadata = metadata
            except AttributeError:
                pass
        self.validate_metadata()

    def update_index(self):
        """Reset index if self.index is unset."""
        if isinstance(self.index, pandas.RangeIndex):
            self['index'] = self._build_index(self)
            self.set_index('index', inplace=True)
            self.index.name = None

    def link_frame(self):
        """
        Apply multi-point constraints to frame.

            - if 'mpc' in self.columns
                Format mpc attribute:

                - mpc is none or NaN: mpc = ''
                - mpc is bool: mpc = '' if False else 1
                - mpc is int or float: mpc[0] = '', mpc[1:] = factor

            - elif self.link
                Add mpc attrbute and set mpc vector to Ture

            - else
                mpc unset.

        """
        if 'mpc' in self.columns:
            isnan = np.array([pandas.isna(mpc)
                              for mpc in self.mpc], dtype=bool)
            self.loc[isnan, 'mpc'] = self.metaframe.default['mpc']
            isnumeric = np.array([isinstance(mpc, (int, float)) &
                                  ~isinstance(mpc, bool)
                                  for mpc in self.mpc], dtype=bool)
            istrue = np.array([mpc is True for mpc in self.mpc], dtype=bool)
            istuple = np.array([isinstance(mpc, tuple)
                                for mpc in self.mpc], dtype=bool)
            self.loc[~istrue & ~isnumeric & ~istuple, 'mpc'] = ''
            index = self.index[istrue | isnumeric]
            if index.empty:
                return
            factor = np.ones(len(self))
            factor[isnumeric] = self.mpc[isnumeric]
            factor = factor[istrue | isnumeric][1:]
            if len(index) > 1:
                self.add_mpc(index, factor)

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
        frames = self._build_frame(*args, **kwargs)
        self.concat(frames, iloc=iloc)

    def concat(self, *frames, iloc=None, sort=False):
        """Concatenate *frames with Frame."""
        dataframe = pandas.DataFrame(self)
        dataframes = [pandas.DataFrame(frame) for frame in frames]
        if iloc is None:  # append
            frames = [dataframe, *dataframes]
        else:  # insert
            frames = [dataframe.iloc[:iloc, :],
                      *dataframes,
                      dataframe.iloc[iloc:, :]]
        frame = pandas.concat(frames, sort=sort)  # concatenate
        Frame.__init__(self, frame, attrs=self.attrs)
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
        frame : Frame

        """
        args, kwargs = self._extract(*args, **kwargs)
        data = self._build_data(*args, **kwargs)
        index = self._build_index(data, **kwargs)
        frame = Frame(data, index=index, attrs=self.attrs)
        frame.generate_polygon()
        frame.link_frame()
        return frame

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
            frame = args[0]
            missing = [arg not in frame for arg in self.metaframe.required]
            if np.array(missing).any():
                required = np.array(self.metaframe.required)[missing]
                raise ValueError(f'required arguments {required} '
                                 f'not specified in frame {frame.columns}')
            args = [frame.loc[:, col] for col in self.metaframe.required]
            if not isinstance(frame.index, pandas.RangeIndex):
                kwargs['name'] = frame.index
            kwargs |= {col: frame.loc[:, col] for col in
                       self.metaframe.additional if col in frame}
        if len(args) != self.metaframe.required_number:
            raise IndexError('incorrect output argument number: '
                             f'{len(args)} != '
                             f'{self.metaframe.required_number}\n')
        return args, kwargs

    @staticmethod
    def isframe(frame, dataframe=True):
        """
        Return isinstance(arg[0], Frame | DataFrame) flag.

        Parameters
        ----------
        frame : Any
            Input.
        dataframe : bool, optional
            Accept pandas.DataFrame. The default is True.

        Returns
        -------
        isframe: bool
            Frame / pandas.DataFrame isinstance flag.

        """
        if isinstance(frame, Frame):
            return True
        if isinstance(frame, pandas.DataFrame) and dataframe:
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

    def add_column(self, label):
        """Add column to Frame initializing values to default."""
        if label not in self.metaframe.columns:
            self.metadata = {'additional': [label]}
            if len(self) > 0:  # initialize with default value
                print(label, self.metaframe.default[label])
                self[label] = self.metaframe.default[label]

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

    def add_mpc(self, index, factor=1):
        """
        Define multi-point constraint linking a set of coils.

        Parameters
        ----------
        index : list[str]
            List of coil names (present in self.index).
        factor : float, optional
            Inter-coil coupling factor. The default is 1.

        Raises
        ------
        IndexError

            - index must be list-like
            - len(index) must be greater than l
            - len(factor) must equal 1 or len(name)-1.

        Returns
        -------
        None.

        """
        if not pandas.api.types.is_list_like(index):
            raise IndexError(f'index: {index} is not list like')
        index_number = len(index)
        if index_number == 1:
            raise IndexError(f'len({index}): {index_number} '
                             'is not greater > 1')
        if not pandas.api.types.is_list_like(factor):
            factor = factor * np.ones(index_number-1)
        elif len(factor) != index_number-1:
            raise IndexError(f'len(factor={factor}) must == 1 '
                             f'or == len(index={index})-1')
        self.at[index[0], 'mpc'] = ''
        for i in np.arange(1, index_number):
            self.at[index[i], 'mpc'] = (index[0], factor[i-1])
        #self.rebuild_CoilArray()

    def drop_mpc(self, index):
        """Drop multi-point constraints referancing dropped coils."""
        if 'mpc' in self.columns:
            if not pandas.api.types.is_list_like(index):
                index = [index]
            name = [mpc[0] if mpc else '' for mpc in self.mpc]
            drop = [n in index for n in name]
            self.remove_mpc(drop)

    def remove_mpc(self, index):
        """Remove multi-point constraint on indexed coils."""
        if not pandas.api.types.is_list_like(index):
            index = [index]
        self.loc[index, 'mpc'] = ''

    def reduce_mpc(self, matrix):
        """Apply mpc constraints to coupling matrix."""
        _matrix = matrix[:, self._mpc_iloc]  # extract primary coils
        if len(self._mpl_index) > 0:  # add multi-point links
            _matrix[:, self._mpl_index[:, 0]] += \
                matrix[:, self._mpl_index[:, 1]] * \
                np.ones((len(matrix), 1)) @ self._mpl_factor.reshape(-1, 1)
        return _matrix

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
            self.loc[name, 'polygon'] = \
                shapely.affinity.translate(self.loc[name, 'polygon'],
                                           xoff=xoffset, yoff=zoffset)
            self.loc[name, 'patch'] = None  # re-generate coil patch

    def limit(self, index):
        """Return coil limits [xmin, xmax, zmin, zmax]."""
        geom = self.loc[index, ['x', 'z', 'dx', 'dz']]
        limit = [min(geom['x'] - geom['dx'] / 2),
                 max(geom['x'] + geom['dx'] / 2),
                 min(geom['z'] - geom['dz'] / 2),
                 max(geom['z'] + geom['dz'] / 2)]
        return limit

    def generate_polygon(self):
        """Generate polygons based on coil geometroy and cross section."""
        if 'polygon' in self.columns:
            for index in self.index[self.polygon.isna()]:
                cross_section = self.loc[index, 'cross_section']
                polygon = polygen(cross_section)(
                    *self.loc[index, ['x', 'z', 'dl', 'dt']])
                self.loc[index, 'polygon'] = polygon
            self.update_polygon()

    def update_polygon(self, index=None):
        """
        Update polygon derived attributes.

        Derived attributes:
            - x, z, dx, dz : float
                coil centroid and bounding box

        Parameters
        ----------
        index : str or array-like or Index, optional
            Frame subindex. The default is None (all coils).

        Raises
        ------
        ValueError
            Zero cross-sectional area.

        Returns
        -------
        None.

        """
        if index is None:
            index = self.index[(self.rms == 0) & (~self.polygon.isna())]
        elif not pandas.api.types.is_list_like(index):
            index = [index]
        for key in index:
            i = self.index.get_loc(key)
            polygon = self.at[key, 'polygon']
            cross_section = self.at[key, 'cross_section']
            length, thickness = self.dl[i], self.dt[i]
            area = polygon.area  # update polygon area
            if area == 0:
                raise ValueError(
                    f'zero area polygon entered for coil {index}\n'
                    f'cross section: {cross_section}\n'
                    f'length {length}\nthickness {thickness}')
            x_center = polygon.centroid.x  # update x centroid
            z_center = polygon.centroid.y  # update z centroid
            self.x[i] = x_center
            self.z[i] = z_center
            self.loc[key, 'dA'] = area
            bounds = polygon.bounds
            self.dx[i] = bounds[2] - bounds[0]
            self.dz[i] = bounds[3] - bounds[1]
            self.rms[i] = root_mean_square(cross_section, x_center,
                                           length, thickness, polygon)
        if len(index) != 0:
            self.update_dataframe = ['x', 'z', 'dx', 'dz', 'rms']


if __name__ == '__main__':

    frame = Frame(mpc=True, metadata={'Additional': ['mpc']})
    frame.add_frame(4, [5, 7, 12], name='coil1')
    print(frame)

