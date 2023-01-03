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
            raise ValueError(f'boundary does not form closed loop')

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
    """Compute maximum field around coil perimeter.

    Parameters
    ----------
    dfield : float, optional
        Boundary probe resoultion. The default is 0.

    """

    dfield: float = 0

    '''
    x, z = coil.polygon[index].boundary.coords.xy
    if dField == 0:  # no interpolation
        polygon = {'x': x, 'z': z}
    else:
        if dField == -1:  # extract dField from coil
            _dL = coil.loc[index, 'dCoil']
        else:
            _dL = dField
        nPoly = len(x)
        polygon = {'_x': x, '_z': z,
                   '_L': length(x, z, norm=False)}
        for attr in ['x', 'z']:
            polygon[f'interp{attr}'] = \
                interp1d(polygon['_L'], polygon[f'_{attr}'])
            dL = [polygon[f'interp{attr}'](
                np.linspace(
                    polygon['_L'][i], polygon['_L'][i+1],
                    1+int(np.diff(polygon['_L'][i:i+2])[0]/_dL),
                    endpoint=False))
                  for i in range(nPoly-1)]
            polygon[attr] = np.concatenate(dL).ravel()
    '''

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
        target = BiotFrame(label='Field')
        for coil in self.loc['coil', 'frame'].unique():
            polyframe = self.extract_polyframe(coil)
            if polyframe.poly.boundary.is_ring:
                sample = Sample(polyframe.boundary, delta=self.dfield)
                target.insert(sample['radius'], sample['height'], link=True)
        self.data = BiotSolve(self.subframe, target,
                              reduce=[True, False], turns=[True, False],
                              attrs=['Br', 'Bz'], name=self.name).data
        # insert grid data
        self.data.coords['x'] = target.x
        self.data.coords['z'] = target.z
        super().post_solve()

    def plot(self, axes=None, **kwargs):
        """Plot points."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='', color='C1', ms=4) | kwargs
        self.axes.plot(self.data.coords['x'], self.data.coords['z'], **kwargs)


'''
class Field(Probe):
    """Field values imposed on coil boundaries - extends Probe class."""

    _biot_attributes = Probe._biot_attributes + ['_coil_index']

    def __init__(self, subcoil, **biot_attributes):
        Probe.__init__(self, subcoil, **biot_attributes)

    def add_coil(self, coil, parts, dField=0.5):
        """
        Add field probes spaced around each coil perimiter.

        Parameters
        ----------
        coil : CoilFrame
            Coil coilframe.
        parts : str or list
            Part names to include field calculation.
        dField : float, optional
            Coil boundary probe resoultion. The default is 0.5.

        Returns
        -------
        None.

        """
        if not is_list_like(parts):
            parts = [parts]
        self._coil_index = []
        target = {'x': [], 'z': [], 'coil': [], 'nC': []}
        for part in parts:
            for index in coil.index[coil.part == part]:
                self._coil_index.append(index)
                x, z = coil.polygon[index].boundary.coords.xy
                if dField == 0:  # no interpolation
                    polygon = {'x': x, 'z': z}
                else:
                    if dField == -1:  # extract dField from coil
                        _dL = coil.loc[index, 'dCoil']
                    else:
                        _dL = dField
                    nPoly = len(x)
                    polygon = {'_x': x, '_z': z,
                               '_L': length(x, z, norm=False)}
                    for attr in ['x', 'z']:
                        polygon[f'interp{attr}'] = \
                            interp1d(polygon['_L'], polygon[f'_{attr}'])
                        dL = [polygon[f'interp{attr}'](
                            np.linspace(
                                polygon['_L'][i], polygon['_L'][i+1],
                                1+int(np.diff(polygon['_L'][i:i+2])[0]/_dL),
                                endpoint=False))
                              for i in range(nPoly-1)]
                        polygon[attr] = np.concatenate(dL).ravel()
                nP = len(polygon['x'])
                target['x'].extend(polygon['x'])
                target['z'].extend(polygon['z'])
                target['coil'].extend([index for __ in range(nP)])
                target['nC'].append(nP)
        self.target.add_coil(target['x'], target['z'],
                             label='Field', delim='',
                             coil=target['coil'])
        _nC = 0
        for nC in target['nC']:
            index = [f'Field{i}' for i in np.arange(_nC, _nC+nC)]
            self.target.add_mpc(index)
            _nC += nC
        self.assemble_biotset()
'''
