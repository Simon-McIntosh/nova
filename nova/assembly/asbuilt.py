"""Manage as-build TFC coil data."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas
import xarray

from nova.assembly.spacialanalyzer import SpacialAnalyzer
from nova.assembly.transform import Rotate


@dataclass
class AsBuilt:
    """Manage as-build TFC ccl data."""

    coil: list[int]
    target: bool = False
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    half_angle: ClassVar[float] = np.pi / 18

    def __post_init__(self):
        """Build load, and transform."""
        self.spacial_analyzer = SpacialAnalyzer()
        self.rotate = Rotate()
        self.initialize_dataset()
        self.load_nominal()
        self.load_messurment()
        self.transform()

    def initialize_dataset(self):
        """Init dataset."""
        self.data = xarray.Dataset(coords=dict(coil=self.coil,
                                               fiducial=list('ABCDEFGH'),
                                               cartesian=list('xyz'),
                                               cylindrical=['r', 'rphi', 'z']))
        for attr, csys in zip(['xyz', 'rpz'], ['cartesian', 'cylindrical']):
            self.data[attr] = ('coil', 'fiducial', csys), \
                np.zeros((len(self.coil), 8, 3))

    def load_nominal(self):
        """Load nominal coordinates in CBD frame."""
        self.data['nominal_ccl'] = self.spacial_analyzer.nominal_ccl
        self.data['nominal_ccl_cylindrical'] = self.rotate.to_cylindrical(
            self.data.nominal_ccl)

    def load_messurment(self):
        """Load as-built dataset in CBD frame."""
        for i, coil in enumerate(self.coil):
            self.data.xyz[i] = getattr(self, f'TFC{coil}')()
        self.data['xyz'] = self.spacial_analyzer.to_datum(self.data.xyz)
        self.data['delta_xyz'] = self.data.xyz - self.data.nominal_ccl

    def transform(self):
        """Apply coordinate transform from cartesian to cylindrical."""
        self.data.rpz[:] = self.rotate.to_cylindrical(self.data.xyz)
        self.data['delta_rpz'] = self.data.rpz - \
            self.data.nominal_ccl_cylindrical

    def TFC12(self):
        """Return ccl fiducials in clocked TGCS for TFC12."""
        if self.target:
            return np.array([[2671.3, -472.8, -3669.66],
                             [2669.6, -472.3, 3728.64],
                             [5249.86, -925.37, 6326.16],
                             [8839.11, -1560.09, 4464.15],
                             [9435.76, -1665.55, -3665.73],
                             [3345.61, -590.41, -5569.2],
                             [10563.37, -1866.14, 28.29],
                             [2670.81, -472.09, 28.73]])
        return np.array([[2670.36, -472.54, -3668.06],
                         [2668.96, -471.97, 3730.17],
                         [5249.19, -925.24, 6327.82],
                         [8838.53, -1560.13, 4466],
                         [9434.99, -1665.81, -3663.6],
                         [3344.69, -590.37, -5567.39],
                         [10562.91, -1866.43, 30.38],
                         [2669.87, -471.87, 30.35]])

    def TFC13(self):
        """Return ccl fiducials in clocked TGCS for TFC12."""
        if self.target:
            return np.array([[2671.8, 472.33, -3672.05],
                             [2669.77, 471.36, 3727.01],
                             [5247.52, 928.19, 6326.27],
                             [8840.4, 1560.75, 4463.45],
                             [9435.13, 1664.33, -3668.33],
                             [3348.13, 590.96, -5570.51],
                             [10564.64, 1864.76, 26.1],
                             [2671.43, 470.21, 26.98]])
        return np.array([[2671.1, 472.48, -3670.65],
                         [2669.14, 471.69, 3728.43],
                         [5246.84, 928.38, 6327.87],
                         [8839.88, 1560.71, 4465.4],
                         [9434.84, 1664.07, -3666.39],
                         [3347.66, 590.97, -5568.9],
                         [10564.49, 1864.56, 28.12],
                         [2670.49, 470.27, 28.57]])

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
        """Format pandas dataframe for HTML display."""
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
        if abs(value) < 1:
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
        return self.dataframe(f'delta_{frame}').style.pipe(self.formater)


if __name__ == '__main__':

    target = AsBuilt([12, 13], True)
    physical = AsBuilt([12, 13], False)

    physical.data['delta_rpz_allignment'] = \
        physical.data.delta_rpz - target.data.delta_rpz

    print(physical.dataframe('delta_rpz_allignment'))

    #print(frame)
