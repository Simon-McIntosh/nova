"""Manage inport and export of SpacialAnalyzer point groups."""
from dataclasses import dataclass, field
import os

import pandas
import xarray

from nova.definitions import root_dir


@dataclass
class SpacialAnalyzer:
    """Manage SpacialAnalyzer point group datasets."""

    sector: int = 6
    subpath: str = 'input/Assembly/Magnets/TFC'
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    def __post_init__(self):
        """Load sector data."""
        self.path = os.path.join(root_dir, self.subpath, f'SM{self.sector}/')
        self.read_nominal()

    def read_nominal(self):
        """Read nominal data from files placed in the SM* dir."""
        self.data['nominal'] = self.read_points('nominal')
        self.data['nominal_ccl'] = self.read_ccl('nominal_ccl')

    def read_csv(self, filename: str) -> pandas.DataFrame:
        """Read SA point group csv file."""
        dataframe = pandas.read_csv(self.path + f'{filename}.txt',
                                    comment='/', header=None,
                                    delimiter=r'[ \t]*,[ \t]*',
                                    engine='python')
        dataframe.columns = ['collection', 'group', 'point', 'x', 'y', 'z']
        return dataframe.set_index('point', drop=False)

    def read_ccl(self, filename: str):
        """Read CCL point group."""
        dataframe = self.read_csv(filename)
        dataframe[['coil', 'point']] = \
            dataframe.point.str.split('-', expand=True)
        dataframe.set_index('point', inplace=True)

        coils = dataframe.coil.unique()
        fiducial = dataframe.index.unique().values

        dataarray = xarray.DataArray(0., dims=['coil', 'fiducial', 'space'],
                                     coords=dict(coil=coils, fiducial=fiducial,
                                                 space=list('xyz')))
        for i, coil in enumerate(coils):
            dataarray[i] = dataframe.loc[dataframe.coil == coil,
                                         ['x', 'y', 'z']].values
        return dataarray

    def read_points(self, filename: str) -> xarray.DataArray:
        """Read point group."""
        dataframe = self.read_csv(filename)
        return xarray.DataArray(dataframe.loc[:, ['x', 'y', 'z']].values,
                                dims=['fiducial_ex', 'space'],
                                coords=dict(fiducial_ex=dataframe.index.values,
                                            space=list('xyz')))


if __name__ == '__main__':

    space = SpacialAnalyzer()
    print(space.data.nominal_ccl)
