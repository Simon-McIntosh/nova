"""Manage inport and export of SpacialAnalyzer point groups."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar
import os

import numpy as np
import pandas
from scipy.optimize import minimize
import scipy.spatial.transform
import warnings
import xarray

from nova.assembly.transform import Rotate
from nova.definitions import root_dir


@dataclass
class SpacialAnalyzer:
    """Manage SpacialAnalyzer point group datasets."""

    sector: int = 6
    subpath: str = 'input/Assembly/Magnets/TFC'
    files: dict[str, str] = field(default_factory=dict)
    datum: tuple = field(init=False, default=(0, 0, -0.7, 0))
    rotate: Rotate = field(init=False, default_factory=Rotate)

    header: ClassVar[list[str]] = ['version', 'frame', 'axes', 'format']
    columns: ClassVar[list[str]] = ['collection', 'group', 'point',
                                    'x', 'y', 'z']

    def __post_init__(self):
        """Load sector data."""
        self.path = os.path.join(root_dir, self.subpath, f'SM{self.sector}/')
        self.files = dict(nominal='nominal', nominal_ccl='nominal_ccl',
                          reference_ccl='reference_ccl') | self.files
        self.extract_datum()

    def rotate_array(self, dataarray: xarray.DataArray, angle: float):
        """Rotate dataarray."""
        rotate = scipy.spatial.transform.Rotation.from_euler('z', angle)
        for i in range(2):
            dataarray[i] = rotate.apply(dataarray[i])

    def datum_error(self, datum, nominal_ccl):
        """Return datum error."""
        nominal = nominal_ccl.copy()
        nominal[..., :2] -= datum[:2]
        self.rotate_array(nominal, -datum[-1])
        return np.mean(self.rotate.clock(nominal[0])[..., 1]**2 +
                       self.rotate.anticlock(nominal[1])[..., 1]**2)

    def extract_datum(self):
        """Fit nominal ccl points to extract datum offset."""
        nominal_ccl = self.read_ccl(self.files['nominal_ccl'])
        opt = minimize(self.datum_error, (0, 0, 0),  args=(nominal_ccl,),
                       method='SLSQP')
        if not opt.success:
            warnings.warn(f'datum extraction failed {opt}')
        self.datum = tuple(opt.x[:2]) + (self.datum[2],) + (opt.x[-1],)

    def to_datum(self, dataarray: xarray.DataArray):
        """Return dataarray shifted from nominal frame into datum frame."""
        dataarray = dataarray.copy()
        dataarray[:] -= self.datum[:3]
        self.rotate_array(dataarray, -self.datum[-1])
        return dataarray

    def from_datum(self, dataarray: xarray.DataArray):
        """Return dataarray shifted from datum frame back to nominal frame."""
        dataarray = dataarray.copy()
        dataarray[:] += self.datum[:3]
        self.rotate_array(dataarray, self.datum[-1])
        return dataarray

    @cached_property
    def nominal(self):
        """Return nominal point dataarray in datum frame."""
        return self.to_datum(self.read_points(self.files['nominal']))

    @cached_property
    def nominal_ccl(self):
        """Return nominal ccl point dataarray in datum frame."""
        return self.to_datum(self.read_ccl(self.files['nominal_ccl'])[:])

    @cached_property
    def reference_ccl(self):
        """Return reference ccl point dataarray in datum frame."""
        return self.to_datum(self.read_ccl(self.files['reference_ccl']))

    def read_csv(self, filename: str) -> pandas.DataFrame:
        """Read SA point group csv file."""
        filename = self.path + f'{filename}.txt'
        dataframe = pandas.read_csv(filename,
                                    comment='/', header=None,
                                    delimiter=r'[ \t]*,[ \t]*',
                                    engine='python')
        dataframe.columns = self.columns
        with open(filename, 'r') as file:
            for attr, line in zip(self.header, file):
                if line[:2] != '//':
                    break
                dataframe.attrs[attr] = line[3:].strip()
        assert dataframe.attrs['format'] == ',  '.join([
            col.capitalize() for col in self.columns])
        for attr in ['collection', 'group']:
            assert len(label := dataframe[attr].unique()) == 1
            dataframe.attrs[attr] = label[0]

        return dataframe.set_index('point', drop=False)

    def read_ccl(self, filename: str):
        """Read CCL point group."""
        dataframe = self.read_csv(filename)
        dataframe[['coil', 'point']] = \
            dataframe.point.str.split('-', expand=True)
        dataframe.set_index('point', inplace=True)

        coils = dataframe.coil.unique()
        fiducial = dataframe.index.unique().values
        dataarray = xarray.DataArray(0.,
                                     dims=['coil', 'fiducial', 'cartesian'],
                                     coords=dict(coil=coils, fiducial=fiducial,
                                                 cartesian=list('xyz')))
        for i, coil in enumerate(coils):
            dataarray[i] = dataframe.loc[dataframe.coil == coil,
                                         ['x', 'y', 'z']].values
        dataarray.coords['coil'] = \
            [int(''.join(filter(str.isdigit, coil)))
             for coil in dataarray.coil.values]
        dataarray.attrs |= dataframe.attrs
        dataarray = dataarray.sel(fiducial=list('ABCDEFGH'))
        dataarray['clock'] = np.mean(np.arctan2(dataarray[..., 1],
                                                dataarray[..., 0]), axis=1)
        return dataarray.sortby(['clock'])

    def read_points(self, filename: str) -> xarray.DataArray:
        """Read point group."""
        dataframe = self.read_csv(filename)
        dataarray = xarray.DataArray(
            dataframe.loc[:, ['x', 'y', 'z']].values,
            dims=['fiducial_ex', 'cartesian'],
            coords=dict(fiducial_ex=dataframe.index.values,
                        cartesian=list('xyz')))
        dataarray.attrs |= dataframe.attrs
        return dataarray

    def write(self, *data: xarray.DataArray, collection='SCOD'):
        """Write spacial analyzer point groups including optimal fit."""
        filename = f'SM{self.sector}_{collection}.txt'
        attrs = self.nominal_ccl.attrs
        with open(self.path + filename, 'w') as file:
            for attr in self.header:
                file.write(f'// {attrs[attr]}\n')
            file.write('\n')
            for group in self.files:
                try:
                    self.to_file(file, getattr(self, group), collection, group)
                except FileNotFoundError:
                    pass
            for dataarray in data:
                self.to_file(file, dataarray.copy(), collection)

    def to_file(self, file, data: xarray.DataArray, collection: str,
                group: str = ''):
        """Write data to file in source frame."""
        data = self.from_datum(data)
        if 'coil' in data.coords:
            return self.write_ccl(file, data, collection, group)
        return self.write_dataarray(file, data, collection, group)

    def to_dataframe(self, dataarray, collection: str, group: str):
        """Return dataframe from dataarray whilst propagating xarray attrs."""
        dataframe = dataarray.to_pandas()
        dataframe['collection'] = collection
        group_label = dataarray.attrs['group']
        if group:
            group_label = f'{group} [{group_label}]'
        dataframe['group'] = group_label
        dataframe['point'] = dataframe.index
        return dataframe[self.columns]

    def write_ccl(self, file, dataarray: xarray.DataArray,
                  collection: str, group: str):
        """Write ccl dataarray to file."""
        for coil in dataarray.coil.values:
            dataframe = self.to_dataframe(dataarray.sel(coil=coil),
                                          collection, group)
            dataframe['point'] = \
                [f'TFC{coil:02d}-{fiducial}' for fiducial in dataframe.index]
            dataframe.to_csv(file, header=False, index=False)

    def write_dataarray(self, file, dataarray: xarray.DataArray,
                        collection: str, group: str):
        """Write 2D xarray dataarray to file."""
        dataframe = self.to_dataframe(dataarray, collection, group)
        dataframe.to_csv(file, header=False, index=False)


if __name__ == '__main__':

    space = SpacialAnalyzer(7)
    print(space.reference_ccl)

    '''
    from nova.assembly.transform import Rotate

    rotate = Rotate()
    print(rotate.clock(space.nominal_ccl[0])[..., 1])
    # space.write()
    print(space.nominal_ccl)
    print(rotate.clock(space.nominal_ccl[0])[..., 1])
    '''
