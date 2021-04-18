"""Manage shell grid."""
from dataclasses import dataclass, field, InitVar
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas
import scipy.interpolate
import shapely.geometry
from rdp import rdp

from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.polygen import PolyFrame
from nova.electromagnetic.polygeom import PolyGeom
from nova.electromagnetic.polyplot import PolyPlot
from nova.utilities import geom
from nova.utilities.pyplot import plt

# pylint:disable=no-member


@dataclass
class ShellCoords:
    """Store shell coordinates and spacing parameters."""

    x_coordinate: npt.ArrayLike
    z_coordinate: npt.ArrayLike
    length: float
    thickness: float
    delta: float = 0
    coords: list[tuple[float, float]] = field(init=False, repr=False)

    def __post_init__(self):
        """Check coordinate shape and calculate segment length."""
        self.ensure_equal_length()
        self.ensure_positive_thickness()
        self.coords = [(x, z) for x, z in
                       zip(self.x_coordinate, self.z_coordinate)]

    def ensure_equal_length(self):
        """Raise error if segment coordinates unequal."""
        if len(self.x_coordinate) != len(self.z_coordinate):
            raise IndexError('Missmatched input coordiantes '
                             f'len(x) {len(self.x_coordinate)} !='
                             f'len(z) {len(self.z_coordinate)}')

    def ensure_positive_thickness(self):
        """Raise error if thickness <= 0."""
        if self.thickness <= 0:
            raise ValueError(f'Shell thickness {self.thickness} <= 0')

    def plot_coordinates(self):
        """Plot shell coordinates."""
        plt.plot(self.x_coordinate, self.z_coordinate, '.-', label='coords')


@dataclass
class ShellInterp(ShellCoords):
    """Calculate total and unit shell lengths and generate interpolator."""

    total_length: float = field(init=False)
    unit_length: npt.ArrayLike = field(init=False, repr=False)
    interp: scipy.interpolate.interp1d = field(init=False, repr=False)

    def __post_init__(self):
        """Check coordinate shape and calculate segment length."""
        super().__post_init__()
        self.total_length, self.unit_length = self.segment_length()
        self.interp = self.generate_interpolator()

    def segment_length(self):
        """Return total segment length and unit length vector."""
        length = geom.length(self.x_coordinate, self.z_coordinate, norm=False)
        return length[-1], length/length[-1]

    def generate_interpolator(self):
        """Return segment interpolator ."""
        return scipy.interpolate.interp1d(self.unit_length, self.coords,
                                          axis=0)


@dataclass
class ShellSegment(ShellInterp):
    """Set vector spacing and Identify geometrical features within segment."""

    eps: float = 1e-3
    rdp: npt.ArrayLike = field(init=False, repr=False)
    ndiv: int = 2
    ldiv: npt.ArrayLike = field(init=False, repr=False)
    columns: list[str] = field(init=False, default_factory=lambda: [
        'x', 'z', 'dl', 'dt', 'dx', 'dz', 'rms', 'area', 'section', 'poly'])

    def __post_init__(self):
        """Construct RDP vector."""
        super().__post_init__()
        self.length = self.update_length()
        self.ndiv = self.update_ndiv()
        self.ldiv = np.linspace(0, 1, self.ndiv)
        self.rdp = self.extract_features()

    def update_length(self) -> float:
        """Return updated sub-segment spacing parameter."""
        if self.length == 0:
            return self.total_length
        if self.length < 0:  # specify segment number
            self.length = self.total_length / -self.length
        if self.length < self.thickness:
            return self.thickness
        return self.length

    def update_ndiv(self):
        """Return updated division number."""
        return int(np.max([1 + self.total_length/self.length, self.ndiv]))

    def extract_features(self):
        """
        Return unit length features identified by RDP algorithum.

        Parameters
        ----------
        eps : float, optional
            RDP epsilon = eps*segment.length[-1].

        Returns
        -------
        None.

        """
        mask = rdp(self.coords, epsilon=self.eps*self.total_length,
                   return_mask=True)
        return self.unit_length[mask]

    def plot_features(self):
        """Plot RDP features."""
        plt.plot(*self.interp(self.rdp).T, 's', label='rdp')

    @property
    def poly(self):
        """Return segment polygon."""
        coords = self.interp(self.rdp)
        poly = shapely.geometry.LineString(coords).buffer(
            self.thickness/2, cap_style=2, join_style=2)
        if isinstance(poly, shapely.geometry.MultiPolygon):
            poly = poly.geoms[0]
        return PolyFrame(poly, name='shell')

    def divide(self):
        """Return subsegment geometry including RDP features."""
        for i in range(self.ndiv-1):
            start, stop = self.ldiv[i:i+2]
            index = (self.rdp > start) & (self.rdp < stop)
            subvector = start, *self.rdp[index], stop
            coords = self.interp(subvector)
            yield ShellSegment(*coords.T, self.delta,
                               self.thickness, eps=self.eps)

    @property
    def dataframe(self):
        """Return subsegment dataframe."""
        data = [[] for __ in range(self.ndiv-1)]
        for i, segment in enumerate(self.divide()):
            geom = PolyGeom(segment.poly)
            data[i] = [*geom.centroid, self.length, self.thickness, *geom.bbox,
                       geom.rms, geom.area, geom.section, geom.poly]
        frame = pandas.DataFrame(data, columns=self.columns)
        frame['nturn'] = frame['area']
        return frame


@dataclass
class ShellGrid(ShellSegment):
    """Subdivide shell segment."""

    frame: pandas.DataFrame = field(init=False)
    subframe: list[pandas.DataFrame] = field(init=False)

    def __post_init__(self):
        """Generate shell geometory."""
        super().__post_init__()
        self.generate()

    def generate(self):
        """Generate frame and subframe(s)."""
        self.frame = self.dataframe
        self.subframe = []
        for segment in self.divide():
            self.subframe.append(segment.dataframe)

    def plot(self):
        """Plot shellgrid data."""
        self.plot_subframe()
        self.plot_geom()

    def plot_geom(self):
        """Plot shell constructive geometory."""
        self.segment.plot()
        self.plot_features()
        self.plot_coordinates()
        plt.legend()
        plt.axis('equal')
        plt.axis('off')

    def plot_frame(self):
        """Plot frame polygons."""
        PolyPlot(DataFrame(self.frame, additional=['part']))()

    def plot_subframe(self):
        """Plot subframe polygons."""
        for subframe in self.subframe:
            PolyPlot(DataFrame(subframe, additional=['part']))()


if __name__ == '__main__':

    shellgrid = ShellGrid([1, 1.5, 2, 2, 4, 4],
                          [0, 0.1, 0, 1, -1, 0], -2, 0.1,
                          delta=0.1)


    print(shellgrid.subframe)


    #shellgrid = ShellGrid([[1, 1.5, 2, 2, 4, 4],
    #                       [0, 0.1, 0, 1, -1, 0]], -6, 0.1, 1e-6,
    #                      delta=0.1)
    #shellgrid.plot()
