"""Solve maximum field on coil perimiter."""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d
import shapely.geometry

from nova.biot.biotframe import BiotFrame
from nova.biot.biotoperate import BiotOperate
from nova.biot.biotsolve import BiotSolve
from nova.frame.baseplot import Plot
from nova.geometry.polyframe import PolyFrame


@dataclass
class Sample(Plot):
    """Sample coil boundary."""

    boundary: np.ndarray
    delta: int | float = 0
    interp: dict[str, interp1d] = field(init=False, repr=False,
                                        default_factory=dict)
    data: dict[str, np.ndarray] = field(init=False, repr=False,
                                        default_factory=dict)

    def __post_init__(self):
        """Store segment coordinates and build boundary interpolators."""
        super().__post_init__()
        self.check()
        self.build()
        self.concatenate()

    def __getitem__(self, attr: str):
        """Return item from data."""
        return self.data[attr]

    def __len__(self):
        """Return sample length."""
        return len(self.data.get('radius', []))

    @property
    def number(self):
        """Return segment number."""
        return len(self.boundary) - 1

    def check(self):
        """Perform input sanity checks."""
        if self.number <= 0:
            raise IndexError(f'boundary length {len(self.boundary)} '
                             'must be greater than 1')
        if not np.allclose(self.boundary[0], self.boundary[-1]):
            raise ValueError('boundary does not form closed loop')

    def build(self):
        """Build segment interpolators for delta != 0."""
        if self.delta == 0:
            return
        length = np.sqrt(np.diff(self.boundary[:, 0])**2 +
                         np.diff(self.boundary[:, 1])**2)
        self.data['length'] = np.append(0, np.cumsum(length))
        for i, attr in enumerate(['radius', 'height']):
            self.interp[attr] = interp1d(self['length'], self.boundary[:, i])
        self.data['node_number'] = self.node_number

    @property
    def node_number(self):
        """Calculate node number for each segment."""
        match self.delta:
            case 0:
                return np.ones(self.number)
            case int() if self.delta < 0:
                return -self.delta * np.ones(self.number, dtype=int)
            case int() | float() if self.delta > 0:
                return np.array([
                    np.max([np.diff(self['length'][i:i+2])[0]/self.delta, 1])
                    for i in range(self.number)], dtype=int)
            case _:
                raise TypeError(f'invalid delta {self.delta}')

    def concatenate(self):
        """Concatenate interpolated boundary segments."""
        if self.delta == 0:
            for i, attr in enumerate(['radius', 'height']):
                self.data[attr] = self.boundary[:, i][:-1]
            return
        for attr in ['radius', 'height']:
            segments = [self.interp[attr](np.linspace(
                self['length'][i], self['length'][i+1], self['node_number'][i],
                endpoint=False)) for i in range(self.number)]
            self.data[attr] = np.concatenate(segments).ravel()

    def plot(self, axes=None):
        """Plot boundary and interpolant nodes."""
        self.get_axes(axes, '2d')
        self.axes.plot(*self.boundary.T, 'C2o', ms=4)
        self.axes.plot(self['radius'], self['height'], 'C1.', ms=4)


@dataclass
class Field(Plot, BiotOperate):
    """
    Compute maximum field around coil perimeter.

    Parameters
    ----------
    dfield : int | +float, optional
        Boundary probe resoultion. The default is 0.

            - 0: boundary contour probes
            - > 0: probe segment resolution
            - int < 0: probe segment number

    """

    dfield: float = 0
    target: BiotFrame = field(init=False, repr=False)

    def __len__(self):
        """Return field probe number."""
        return len(self.data.get('x', []))

    def extract_polyframe(self, coil: str):
        """Extract polygon from frame."""
        match self.frame.loc[coil, 'poly']:
            case str():
                return PolyFrame.loads(self.frame.loc[coil, 'poly'])
            case PolyFrame():
                return self.frame.loc[coil, 'poly']

    def solve(self, dfield=None):
        """Extract boundary and solve magnetic field around coil perimeter."""
        if dfield is not None:
            self.dfield = dfield
        self.target = BiotFrame(label='Field')
        for coil in self.loc['coil', 'frame'].unique():
            polyframe = self.extract_polyframe(coil)
            if polyframe.poly.boundary.is_ring:
                sample = Sample(polyframe.boundary, delta=self.dfield)
                self.target.insert(sample['radius'], sample['height'],
                                   link=True)
        self.data = BiotSolve(self.subframe, self.target,
                              reduce=[True, False], turns=[True, False],
                              attrs=['Br', 'Bz'], name=self.name).data
        # insert grid data
        self.data.coords['x'] = self.target.x
        self.data.coords['z'] = self.target.z
        super().post_solve()

    def max_br(self):
        #print(self.target.biotreduce.indices)
        return np.maximum.reduceat(self.br,
                                   self.target.biotreduce.indices)


    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='', color='C1', ms=4) | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)
