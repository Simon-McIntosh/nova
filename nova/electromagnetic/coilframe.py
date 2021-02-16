"""Extend pandas.DataFrame to manage coil and subcoil data."""

import re
import string
from typing import Optional, Collection, Any
import warnings

import pandas
import numpy as np
import shapely

from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.coildata import CoilData
from nova.electromagnetic.polygen import polygen, root_mean_square

# pylint:disable=unsubscriptable-object
# pylint: disable=too-many-ancestors


class CoilSeries(pandas.Series):
    """Provide series constructor methods."""

    @property
    def _constructor(self):
        return CoilSeries

    @property
    def _constructor_expanddim(self):
        return CoilFrame


class CoilFrame(pandas.DataFrame, CoilData):
    """
    CoilFrame instance inherits from Pandas DataFrame and Coildata.

    Inspiration for DataFrame inheritance taken from GeoPandas
    https://github.com/geopandas.
    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 metadata: Optional[dict] = None):
        pandas.DataFrame.__init__(self, data, index, columns)
        self.metadata = metadata
        #CoilData.__init__(self)  # fast access attributes

    @property
    def _constructor(self):
        return CoilFrame

    @property
    def _constructor_sliced(self):
        return CoilSeries

    @property
    def metaframe(self):
        """Return MetaFrame instance."""
        try:
            return self.attrs['metaframe']
        except KeyError:
            self.attrs['metaframe'] = MetaFrame()
            return self.attrs['metaframe']

    @property
    def metadata(self):
        """Manage CoilFrame metadata via the MetaFrame class."""
        return self.metaframe.metadata

    @metadata.setter
    def metadata(self, metadata):
        if metadata is None:
            metadata = {}
        if 'metaframe' in metadata:
            self.attrs['metaframe'] = metadata.pop('metaframe')
        self.metaframe.metadata = metadata

    @property
    def coil_number(self):
        """Return coil number."""
        return len(self.index)

    def add_coil(self, *args, iloc=None, **kwargs):
        """
        Add coil(s) to CoilFrame, return updated index.

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
            Updated CoilFrame index.

        """
        coil = self._build_coil(*args, **kwargs)
        self.concatenate(coil, iloc=iloc)
        return coil.index

    def _build_coil(self, *args, **kwargs):
        """
        Return CoilFrame constructed from required and optional input.

        Parameters
        ----------
        *args : Union[float, array-like]
            Required arguments listed in self.metaframe.required.
        **kwargs : dict[str, Union[float, array-like, str]]
            Optional keyword arguments listed in self.metaframe.additional.

        Returns
        -------
        coil : TYPE
            DESCRIPTION.

        """
        # unpack optional keyword arguments
        mpc = kwargs.pop('mpc', False)
        delim = kwargs.pop('delim', '_')
        label = kwargs.pop('label', 'Coil')
        name = kwargs.pop('name', None)
        args, kwargs = self._format_arguments(*args, **kwargs)
        data = self._build_data(*args, **kwargs)
        index = self._build_index(data, delim, label, name)
        coil = CoilFrame(data, index=index, metadata=self.metadata)
        coil.generate_polygon()
        if mpc and coil.nC > 1:
            coil.add_mpc(coil.index.to_list())
        return coil

    def _format_arguments(self, *args, **kwargs):
        """
        Return formated args and kwargs.

            - Extract coilframe if self.isframe(args[0]).
            - Ensure len(args) == len(self.metaframe.required).



        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        IndexError
            DESCRIPTION.
        KeyError
            DESCRIPTION.

        Returns
        -------
        args : TYPE
            DESCRIPTION.
        kwargs : TYPE
            DESCRIPTION.

        """
        if self.isframe(*args):  # data passed as CoilFrame
            args, kwargs = self._extract_coilframe(args[0], **kwargs)
        elif len(self._required_columns) != len(args):  # set from kwargs
            raise IndexError(f'\nincorrect argument number: {len(args)}\n'
                             f'input *args as {self._required_columns} '
                             '\nor set _default_columns=[*] in kwarg')
        for key in self._additional_columns:
            if key not in kwargs and key not in self._default_attributes:
                raise KeyError(f'default_attributes not set for {key} in '
                               f' {self._default_attributes.keys()}')
        return args, kwargs

    def concatenate(self, *coil, iloc=None, sort=False):
        """Return concatenated CoilFrames."""
        if iloc is None:  # append
            coils = [self, *coil]
        else:  # insert
            coils = [self.iloc[:iloc, :], *coil, self.iloc[iloc:, :]]
        coil = pandas.concat(coils, sort=sort)  # concatenate
        CoilFrame.__init__(self, coil, metadata=self.metadata)
        self.rebuild_coildata()  # rebuild fast index

    def drop_coil(self, index=None):
        """Drop coil(s)."""
        if index is None:
            index = self.index
        self.drop_mpc(index)
        self.drop(index, inplace=True)
        self.rebuild_coildata()

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
        for i in np.arange(1, index_number):
            self.at[index[i], 'mpc'] = (index[0], factor[i])
        self.rebuild_coildata()

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

    def limit(self, index):
        """Return coil limits [xmin, xmax, zmin, zmax]."""
        geom = self.loc[index, ['x', 'z', 'dx', 'dz']]
        limit = [min(geom['x'] - geom['dx'] / 2),
                 max(geom['x'] + geom['dx'] / 2),
                 min(geom['z'] - geom['dz'] / 2),
                 max(geom['z'] + geom['dz'] / 2)]
        return limit

    @staticmethod
    def isframe(*args, accept_dataframe=True):
        """
        Return isinstance(arg[0], CoilFrame) flag.

        Parameters
        ----------
        *args : Any
            Input arguments.
        accept_dataframe : bool, optional
            Accept pandas.DataFrame. The default is True.

        Returns
        -------
        is_coilframe: bool
            Coilframe isinstance flag.

        """
        if len(args) == 1:
            if isinstance(args[0], CoilFrame):
                return True
            if isinstance(args[0], pandas.DataFrame) and accept_dataframe:
                return True
        return False

    def _extract_coilframe(self, coilframe, **kwargs):
        """Extract data from coilframe and set as args / kwargs."""
        args = [coilframe.loc[:, col] for col in self._required_columns]
        kwargs['name'] = coilframe.index
        for col in coilframe.columns:
            if col not in self._required_columns:
                if col in self._additional_columns:
                    kwargs[col] = coilframe.loc[:, col]
        return args, kwargs

    def _extract_data(self, *args, **kwargs):
        data = {}  # python 3.6+ assumes dict is insertion ordered
        for key, arg in zip(self._required_columns, args):
            data[key] = np.array(arg, dtype=float)  # add required arguments
        current_label = self._extract_current_label(**kwargs)
        for key in self._additional_columns:
            if key in kwargs:
                data[key] = kwargs.pop(key)
            else:
                data[key] = self._default_attributes[key]
        for key in self._default_attributes:
            additional_columns = []
            if key in kwargs:
                additional_columns.append(key)
                data[key] = kwargs.pop(key)
        self._update_coilframe_metadata(additional_columns=additional_columns)
        self._propogate_current(current_label, data)
        if len(kwargs.keys()) > 0:
            warnings.warn(f'\n\nunset kwargs: {list(kwargs.keys())}'
                          '\nto use include within additional_columns:\n'
                          f'{self._additional_columns}'
                          '\nor within default_attributes:\n'
                          f'{self._default_attributes}\n')
        return data

    def _extract_current_label(self, **kwargs):
        current_label = None
        if 'Ic' in self._required_columns or 'Ic' in kwargs:
            current_label = 'Ic'
        elif 'It' in self._required_columns or 'It' in kwargs:
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

    def _extract_index(self, data, delim, label, name):
        try:
            nCol = np.max([len(data[key]) for key in data
                           if pandas.api.types.is_list_like(data[key])])
        except ValueError:
            nCol = 1  # scalar input
        if isinstance(name, pandas.RangeIndex):
            name = None
        if pandas.api.types.is_list_like(name):
            if len(name) != nCol:
                raise IndexError(f'missmatch between name {name} and '
                                 f'column number: {nCol}')
            index = name
        else:
            if name is None:
                try:  # reverse search through coilframe index
                    offset = next(
                        int(re.sub(r'[a-zA-Z]', '',
                                   index).replace(delim, '').replace('_', ''))
                        for index in self.index[::-1] if label in index) + 1
                except StopIteration:  # label not present in index
                    offset = 0
            else:
                if delim:
                    label = name.split(delim)[0]
                    try:
                        index = name.split(delim)[1]
                    except IndexError:
                        index = ''
                else:
                    label = name.rstrip(string.digits)  # trailing  number
                    index = name.rstrip(string.ascii_letters)
                try:  # build list taking starting index from name
                    offset = int(re.sub(r'[a-zA-Z]', '', index))
                except ValueError:
                    offset = 0
            if nCol > 1 or name is None:
                index = [f'{label}{delim}{i+offset:d}' for i in range(nCol)]
            else:
                index = [name]
        self._check_index(index)
        return index

    def _check_index(self, index):
        for name in index:
            if name in self.index:
                raise IndexError(f'\ncoil: {name} already defined in index\n'
                                 f'index: {self.index}')

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
            CoilFrame subindex. The default is None (all coils).

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
            dl, dt = self.dl[i], self.dt[i]
            dA = polygon.area  # update polygon area
            if dA == 0:
                raise ValueError(
                    f'zero area polygon entered for coil {index}\n'
                    f'cross section: {cross_section}\n'
                    f'dl {dl}\ndt {dt}')
            x = polygon.centroid.x  # update x centroid
            z = polygon.centroid.y  # update z centroid
            self.x[i] = x
            self.z[i] = z
            self.loc[key, 'dA'] = dA
            bounds = polygon.bounds
            self.dx[i] = bounds[2] - bounds[0]
            self.dz[i] = bounds[3] - bounds[1]
            self.rms[i] = root_mean_square(cross_section, x, dl, dt, polygon)
        if len(index) != 0:
            self.update_dataframe = ['x', 'z', 'dx', 'dz', 'rms']


if __name__ == '__main__':

    frame = CoilFrame()
