"""Manage inport and export of SpacialAnalyzer point groups."""
from dataclasses import dataclass
import os

import pandas
import xarray

from nova.definitions import root_dir


@dataclass
class SpacialAnalyzer:
    """Manage SpacialAnalyzer point group datasets."""

    sector: int = 6
    subpath: str = 'input/Assembly/Magnets/TFC'

    def __post_init__(self):
        """Load sector data."""
        self.path = os.path.join(root_dir, self.subpath, f'SM{self.sector}/')

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
        points = dataframe.index.unique()

        dataarray = xarray.DataArray(0., dims=['coil', 'point', 'space'],
                                     coords=dict(coil=coils, point=points,
                                                 space=list('xyz')))
        for i, coil in enumerate(coils):
            dataarray[i] = dataframe.loc[dataframe.coil == coil,
                                         ['x', 'y', 'z']].values
        return dataarray
        print(dataarray)

    def read_points(self, filename: str) -> xarray.DataArray:
        """Read point group."""
        dataframe = self.read_csv(filename)
        dataarray = xarray.DataArray(0., dims=['point', 'space'],
                                     coords=dict(point=dataframe.index,
                                                 space=list('xyz')))
        print(dataarray)



        #print(data)


if __name__ == '__main__':

    space = SpacialAnalyzer()
    space.read_points('nominal')
