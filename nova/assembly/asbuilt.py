"""Manage as-build TFC coil data."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas
import xarray


@dataclass
class AsBuilt:
    """Manage as-build TFC ccl data."""

    coil: list[int]
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    half_angle: ClassVar[float] = np.pi / 18

    def __post_init__(self):
        """Build load, and transform."""
        #self.spacial_analyzer = SpacialAnalyzer()
        self.build()
        self.load_nominal()
        self.load_deltas()
        self.transform()

    def build(self):
        """Build data structure."""
        self.data = xarray.Dataset(coords=dict(coil=self.coil,
                                               fiducial=list('ABCDEFGH'),
                                               cartesian=list('xyz'),
                                               cylindrical=['r', 'rphi', 'z']))
        for attr, csys in zip(['xyz', 'rpz'], ['cartesian', 'cylindrical']):
            self.data[attr] = ('coil', 'fiducial', csys), \
                np.zeros((len(self.coil), 8, 3))

    #def load_nominal()
    def load_deltas(self):
        """Load as-built dataset."""
        for i, coil in enumerate(self.coil):
            self.data.xyz[i] = getattr(self, f'TFC{coil}')()

    def transform(self):
        """Apply coordinate transform from cartesian to cylindrical."""
        self.data.rpz[..., 0] = \
            self.data.xyz[..., 0] * np.cos(self.half_angle) - \
            self.data.xyz[..., 1] * np.sin(self.half_angle)
        self.data.rpz[..., 1] = \
            self.data.xyz[..., 1] * np.cos(self.half_angle) - \
            self.data.xyz[..., 0] * np.sin(self.half_angle)
        self.data.rpz[..., -1] = self.data.xyz[..., -1]

    @staticmethod
    def TFC12():
        """Return alignment deltas in cartesian coordinates for TFC12."""
        return np.array([[0.27, -0.7, 2.04],
                         [-1.43, -0.2, 0.34],
                         [-2.02, 2.02, 1.46],
                         [-3.33, 0.71, -1.15],
                         [-4.64, 0.73, 0.97],
                         [-0.99, 0.87, 0.5],
                         [-5.02, -0.87, -0.01],
                         [-0.22, 0.01, 0.43]])

    @staticmethod
    def TFC13():
        """Return alignment deltas in cartesian coordinates for TFC13."""
        return np.array([[0.7, 1.98, -0.35],
                         [-1.33, 1., -1.29],
                         [-4.5, 2.96, 1.57],
                         [-2.28, 2.69, -1.85],
                         [-5.54, 0.87, -1.63],
                         [ 1.44, 1.54, -0.81],
                         [-4.04, 2.5, -2.2],
                         [0.33, -0.15, -1.32]])


    def dataframe(self, frame='xyz'):
        """Return xyz dataframe."""
        frames = []
        for coil in self.coil:
            dataframe = self.data[frame].sel(coil=coil).to_pandas()
            dataframe.index = pandas.MultiIndex.from_product(
                [[coil], dataframe.index], names=['coil', 'fiducial'])
            frames.append(dataframe)
        return pandas.concat(frames)

    def formater(self, styler):
        styler.format('{:.2f}')
        styler.set_table_styles([dict(selector='th:not(.index_name)',
                                      props=[('background-color', 'white')])])
        styler.apply(self.color, axis=None)
        styler.apply(self.hide, axis=None)
        return styler

    def empty_like(self, frame):
        """Return empty_like dataframe."""
        return pandas.DataFrame('', index=frame.index, columns=frame.columns)

    def color_value(self, value):
        """Return color based on value."""
        if abs(value) < 0.9:
            return 'color:white;background-color:darkgreen'
        if abs(value) > 1.1:
            return 'color:white;background-color:darkred'
        return 'color:white;background-color:darkorange'


    def color(self, frame):
        """Color cells based on proximity to limits."""
        props = self.empty_like(frame)
        for icol, limit in enumerate(np.array([1.5, 1.5, 3]) +
                                     2*np.sqrt(0.5)):
            series = frame.iloc[:, icol] / limit
            props.iloc[:, icol] = series.map(self.color_value)
        return props

    def hide(self, frame):
        """Hide untracked fiducials."""
        props = self.empty_like(frame)
        props.loc[(self.coil, list('CDEFG')),
                  frame.columns[0]] = 'color:white;background-color:white'
        props.loc[(self.coil, list('ABGH')),
                  frame.columns[2]] = 'color:white;background-color:white'
        return props

    def html(self, frame='xyz'):
        """Return html dataframe."""
        return self.dataframe(frame).style.pipe(self.formater)



if __name__ == '__main__':

    asbuilt = AsBuilt([12, 13])

    frame = asbuilt.dataframe('rpz')
    print(frame)
