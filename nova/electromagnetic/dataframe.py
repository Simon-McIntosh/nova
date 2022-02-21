"""Subclass pandas.DataFrame."""
import re
import string

import copy
import json
import pandas
import numpy as np
import xarray

from nova.electromagnetic.frameattrs import FrameAttrs
from nova.geometry.geoframe import GeoFrame
from nova.geometry.polyframe import PolyFrame
from nova.geometry.vtkgen import VtkFrame

# pylint: disable=too-many-ancestors


class Series(pandas.Series):
    """Provide series constructor methods."""

    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_expanddim(self):
        return DataFrame


class DataFrame(FrameAttrs):
    """
    Extend pandas.DataFrame.

    - DataFrame singleton (no subspace, select, geometory, multipoint,
                           energize or plot methods)

    """

    geoframe = dict(Polygon=PolyFrame, VTK=VtkFrame, Geo=GeoFrame, Json=str)

    def __init__(self, data=None, index=None, columns=None,
                 attrs=None, **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.update_index()
        self.update_columns()

    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series

    def update_index(self):
        """Reset index if self.index is unset."""
        if not self.index.is_unique:  # rebuild index
            self.index = pandas.RangeIndex(len(self))
        if isinstance(self.index, pandas.RangeIndex):
            self['index'] = self._build_index(self)
            self.set_index('index', inplace=True)
            self.index.name = None
        self.metaframe.index = self.index

    def _set_offset(self, metatag):
        try:  # reverse search through frame index
            match = next(name for name in self.index[::-1]
                         if metatag['label'] in name)
            if metatag['delim'] and metatag['delim'] in match:
                offset = int(match.split(metatag['delim'])[-1])
            else:
                offset = re.sub(r'[a-zA-Z]', '', match)
            if isinstance(offset, str):
                offset = offset.replace(metatag['delim'], '')
                offset = offset.replace('_', '')
                offset = int(offset)
            offset += 1
        except (TypeError, StopIteration):  # unit index, label not present
            offset = 0
        metatag['offset'] = np.max([offset, metatag['offset']])

    def _build_index(self, data: pandas.DataFrame, **kwargs):
        """Wrap public build_index with dataframe input."""
        index_length = self._index_length(data)
        return self.build_index(index_length, **kwargs)

    def build_index(self, index_length: int, **kwargs):
        """Return index constructed from length and kwargs."""
        # update metaframe tag defaults with kwargs
        metatag = {key: kwargs.get(key, self.metaframe.default[key])
                   for key in self.metaframe.tag}
        if isinstance(name := metatag['name'], pandas.Index) or len(name) > 0:
            if pandas.api.types.is_list_like(name) or index_length == 1:
                return self._check_index(name, index_length)
            if metatag['delim'] and metatag['delim'] in name:
                split_name = name.split(metatag['delim'])
                metatag['label'] = metatag['delim'].join(split_name[:-1])
                metatag['offset'] = int(split_name[-1])
            else:
                metatag['delim'] = ''
                metatag['label'] = name.rstrip(string.digits)
                try:
                    metatag['offset'] = int(name.lstrip(string.ascii_letters))
                except ValueError:  # no trailing number, use default
                    pass
        self._set_offset(metatag)
        label_delim = metatag['label']+metatag['delim']
        index = [f'{label_delim}{i+metatag["offset"]:d}'
                 for i in range(index_length)]
        if 'frame' in self.columns and metatag['label'] not in self.index:
            index[0] = metatag['label']
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
        if len(data) == 0:
            return 0
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
                             f'maximum length data column {index_length}')
        if len(index) != len(np.unique(index)):
            raise IndexError(f'index not unique {index}')
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
        with self.setlock(None):
            if self.columns.empty:
                for attr in self.metaframe.columns:
                    self[attr] = None
                return
            columns = self.columns.to_list()
            # check required
            required_unset = [attr not in columns
                              for attr in self.metaframe.required]
            if np.array(required_unset).any():
                unset = np.array(self.metaframe.required)[required_unset]
                raise IndexError(f'required attributes missing {unset}')
            # fill nan
            frame = pandas.DataFrame(self)
            if frame.isna().values.any():
                for attr in frame:
                    isna = pandas.isna(frame[attr])
                    if isna.any() and not any([value is None for value
                                               in frame[attr].values]):
                        self.loc[isna, attr] = self.metaframe.default[attr]
            # extend additional
            additional = [attr for attr in columns
                          if attr not in self.metaframe.columns]
            if additional:
                self.metaframe.metadata = {'additional': additional}
            # set defaults
            additional_unset = [attr not in columns
                                for attr in self.metaframe.columns]
            if np.array(additional_unset).any():
                unset = np.array(self.metaframe.columns)[additional_unset]
                if self.index.empty:  # insert additional columns
                    for attr in unset:
                        self[attr] = None
                    return
                for attr in unset:
                    self.loc[:, attr] = self.metaframe.default[attr]
                turn_set = np.array([attr in self.columns
                                     for attr in ['It', 'nturn']])
                if 'Ic' in unset and turn_set.all():
                    self.loc[:, 'Ic'] = \
                        self.loc[:, 'It'] / self.loc[:, 'nturn']

    def _dumps(self, col: str):
        """Return poly as list of json strings."""
        return [geom.dumps() if geom else '' for geom in self[col]]

    def _loads(self, col: str):
        """Load json strings and convert to shapely polygons."""
        geotype = [json.loads(geom)['type'] if geom else None
                   for geom in self[col]]
        self.loc[:, col] = [self.geoframe[geo].loads(geom)
                            for geom, geo in zip(self[col], geotype)]

    def geotype(self, geo: str, col: str):
        """Return boolean list of matching geoframe types."""
        return np.array([isinstance(geom, self.geoframe[geo])
                         for geom in self[col]], dtype=bool)

    def extract_metadata(self) -> dict:
        """Return metadata with version as attribute list."""
        metadata = copy.deepcopy(self.metaframe.metadata)
        if 'version' in metadata:
            metadata['version'] = list(metadata['version'])
        return metadata

    def insert_metadata(self, attrs: dict):
        """Return metadata wirk version as dict[str id(None)]."""
        metadata = copy.deepcopy(attrs)
        if 'version' in metadata:
            if isinstance(metadata['version'], str):
                metadata['version'] = [metadata['version']]
            metadata['version'] = {attr: id(None)
                                   for attr in metadata['version']}
        return metadata

    def store(self, file, group=None, mode='w', vtk=False):
        """Store dataframe as group in netCDF4 hdf5 file."""
        xframe = self.to_xarray()
        xframe.attrs = self.extract_metadata()
        for col in ['poly', 'vtk']:
            if col == 'vtk' and not vtk:
                xframe = xframe.drop_vars('vtk', errors='ignore')
                continue
            try:
                xframe[col].values = self._dumps(col)
            except KeyError:
                pass
        xframe.to_netcdf(file, group=group, mode=mode)

    def load(self, file, group=None):
        """Load dataframe from hdf file."""
        with xarray.open_dataset(file, group=group) as data:
            metadata = self.insert_metadata(data.attrs)
            self.__init__(data.to_dataframe(), **metadata)
        if 'index' in self.version:
            self.version['index'] = id(self.index)
        for col in ['poly', 'vtk']:
            try:
                self._loads(col)
            except KeyError:
                pass
        return self


if __name__ == '__main__':

    dataframe = DataFrame(base=['x', 'y', 'z'],
                          required=['x'], additional=['Ic', 'z'],
                          Subspace=[], label='PF')

    dataframe.store('tmp.h5', 'frame')
    dataframe.load('tmp.h5', 'frame')
