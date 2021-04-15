"""Manage shell grid."""
from dataclasses import dataclass, field, InitVar
from typing import Union

import descartes
import numpy as np
import numpy.typing as npt
import pandas
import scipy.interpolate
import shapely.geometry
from rdp import rdp

from nova.electromagnetic.polygen import PolyFrame
from nova.utilities import geom
from nova.utilities.pyplot import plt


@dataclass
class ShellSegment:
    """Manage shell segment."""

    segment: InitVar[list[npt.ArrayLike]]
    frame: pandas.DataFrame = field(init=False)
    length: npt.ArrayLike = field(init=False)
    unit_length: npt.ArrayLike = field(init=False)
    columns: list[str] = field(init=False, default_factory=lambda: [
        'x', 'z', 'dt', 'rho'])

    def __post_init__(self, segment):
        """Update segment, check segment shape and calculate segment length."""
        self.frame = self.extract_segment(segment)
        self.assert_coordinates()
        self.length, self.unit_length = self.extract_length()

    def __len__(self):
        """Return frame length."""
        return len(self.frame)

    def extract_segment(self, segment) -> dict[str, npt.ArrayLike]:
        """Return updated segment, transform to dict, assert shape."""
        data = {key: value for i, (key, value)
                in enumerate(zip(self.columns, segment))}
        return pandas.DataFrame(data)

    def assert_coordinates(self):
        """Assert that segment contains x and z coordinates."""
        assert np.array([col in self.frame for col in ['x', 'z']]).all()

    def extract_length(self):
        """Return segment length and unit length vectors."""
        return (length := geom.length(self.frame['x'], self.frame['z'],
                                      norm=False), length/length[-1])

    def data(self, name):
        """Return unit spacing, segment vaiable tuple."""
        return self.unit_length, self.frame[name]

    def plot(self):
        """Plot segment centerline."""
        plt.plot(self.frame['x'], self.frame['z'], 'd-', ms=12, label='line')


@dataclass
class ShellVector:
    """Manage 1D shell spacing vector."""

    segment: list[list[float]]
    delta: Union[int, float]
    thickness: npt.ArrayLike = None
    resistivity: npt.ArrayLike = None
    ndiv: int = 2
    unit_length: npt.ArrayLike = field(init=False, repr=False)

    def __post_init__(self):
        """Update segment and init 1D grid."""
        self.segment = self.update_segment()
        self.delta = self.update_delta()
        self.ndiv = self.update_ndiv()
        self.unit_length = np.linspace(0, 1, self.ndiv)

    def update_segment(self):
        """Assemble segment attributes and promote to ShellSegment."""
        assert isinstance(self.segment, list)
        assert len(self.segment) == 2  # x, z coordinates
        self.segment += [attr for col in ['thickness', 'resistivity']
                         if (attr := getattr(self, col)) is not None]
        return ShellSegment(self.segment)

    def update_delta(self) -> float:
        """Return updated sub-segment spacing parameter."""
        segment_length = self.segment.length[-1]
        if self.delta == 0:
            return segment_length
        if self.delta < 0:  # specify segment number
            return segment_length / -self.delta
        if self.delta < (minimum_thickness := np.min(self.thickness)):
            return minimum_thickness
        return self.delta

    def update_ndiv(self):
        """Return updated division number."""
        return int(np.max([self.segment.length[-1]/self.delta, self.ndiv]))

    def interp1d(self, name, vector='unit_length'):
        """Return interpolated variable."""
        interpolator = scipy.interpolate.interp1d(*self.segment.data(name))
        if isinstance(vector, str):
            vector = getattr(self, vector)
        return interpolator(vector)

    @property
    def coordinates(self):
        """Return segment coordinates."""
        return [self.interp1d(name) for name in ['x', 'z']]

    def plot_coordinates(self):
        """Plot interpolated coordinates."""
        plt.plot(*self.coordinates, 'o', label='coords')

    '''
        dt_min = dt.min()
        if dS < dt_min:
            dS = dt_min

        nS = int(L / dS)  # segment number
        if nS < 1:  # ensure > 0
            nS = 1

        dS = L / nS  # model resolved sub-segment length
        nSS = int(L / dt_min)
        if nSS < 2:
            nSS = 2
    '''


@dataclass
class ShellGeom(ShellVector):
    """Retain shell geometry features within segment unit length vector."""

    segment: ShellSegment
    delta: Union[int, float]
    thickness: npt.ArrayLike = None
    eps: float = 1e-3
    cluster: float = 0.75
    unit_feature: npt.ArrayLike = field(init=False, repr=False)

    def __post_init__(self):
        """Update unit length vector with RDP features."""
        super().__post_init__()
        self.insert_features()

    def insert_features(self):
        """
        Insert features to unit length.

        Insert features identified by RDP algorithum into unit length vector.

        Parameters
        ----------
        eps : float, optional
            RDP epsilon = eps*segment.length[-1].
        cluster : float, optional
            Cluster adjacent points to features.
                - factor = 0: insert all.
                - factor >> 0: cluster all.

        Returns
        -------
        None.

        """
        mask = rdp(self.segment.frame.loc[:, ['x', 'z']],
                   epsilon=self.eps*self.segment.length[-1], return_mask=True)
        self.unit_feature = self.segment.unit_length[mask]

        self.unit_sublength = [vector for vector in self.subsegment()]
        print(self.unit_sublength)
        '''
        rdp_delta = np.min(np.diff(self.unit_feature))  # minimum feature delta
        unit_length = self.unit_length.reshape(-1, 1)
        unit_length = unit_length @ np.ones((1, len(self.unit_feature)))
        delta = np.abs(unit_length - self.unit_feature)
        feature_index = np.min(delta, axis=0) < self.cluster * rdp_delta
        length_index = np.argmin(delta[:, feature_index], axis=0)
        self.unit_length[length_index] = self.unit_feature[feature_index]
        '''
        #self.unit_length = np.sort(np.append(self.unit_length,
        #                                     self.unit_feature))

    def subsegment(self):
        """Return subsegment coordinates including RDP features."""
        for i in range(len(self.segment)-1):
            start, stop = self.unit_length[i:i+2]
            feature_index = self.unit_feature > start
            feature_index &= self.unit_feature < stop
            sublength = start, *self.unit_feature[feature_index], stop

            yield self.interp1d('x', sublength), self.interp1d('z', sublength)

    def plot_features(self):
        """Plot RDP features."""
        plt.plot(self.interp1d('x', 'unit_feature'),
                 self.interp1d('z', 'unit_feature'), 's', label='rdp')


@dataclass
class ShellGrid(ShellGeom):
    """Subdivide shell segment."""

    segment: list[list[float]]
    delta: float
    thickness: npt.ArrayLike = None

    def __post_init__(self):
        """Generate shell geometory."""
        super().__post_init__()

        line = self.segment.frame.loc[:, ['x', 'z']].to_numpy()
        thickness = self.segment.frame.dt.to_numpy()[0]
        poly = shapely.geometry.LineString(line).buffer(
                    thickness/2, cap_style=2, join_style=2)

        axes = plt.gca()
        axes.add_patch(descartes.PolygonPatch(poly))

        # TODO split polygon
        #from shapely.ops import linemerge, unary_union, polygonize

        #merged = linemerge([poly.boundary, line])
        ##borders = unary_union(merged)
        #polygons = polygonize(borders)

    def plot_geom(self):
        """Plot shell constructive geometory."""
        self.segment.plot()
        self.plot_features()
        self.plot_coordinates()
        plt.legend()
        plt.axis('equal')
        plt.axis('off')


        #self.thickness *= np.ones(self.segment.number)  # variable thickness

    def _mesh(segment, dt, rho, dS):
        if not pandas.api.types.is_list_like(dt):  # variable thickness
            dt *= np.ones(np.shape(segment)[1])
        if not pandas.api.types.is_list_like(rho):  # variable resistivity
            rho *= np.ones(np.shape(segment)[1])
        #dL = geom.length(*segment, norm=False)  # cumulative segment length



        _x, _z = geom.xzfun(*segment)  # xz interpolators
        _dt = scipy.interpolate.interp1d(dL/L, dt)
        _rho = scipy.interpolate.interp1d(dL/L, rho)
        Lend = np.linspace(0, 1, nS+1)  # endpoints
        polygon = [[] for __ in range(nS)]
        x, z = np.zeros(nS), np.zeros(nS)
        rho_bar, dt_bar = np.zeros(nS), np.zeros(nS)
        dl, dt, area = np.zeros(nS), np.zeros(nS), np.zeros(nS)
        sub_segment = np.zeros((nS, 2, nSS))
        sub_rho, sub_dt = np.zeros((nS, nSS)), np.zeros((nS, nSS))
        for i in range(nS):
            Lsegment = np.linspace(Lend[i], Lend[i+1], nSS)
            dLseg = Lend[i+1] - Lend[i]
            sub_segment[i, :, :] = np.array([_x(Lsegment), _z(Lsegment)])
            sub_rho[i, :] = _rho(Lsegment)
            sub_dt[i, :] = _dt(Lsegment)
            # average attributes
            dt_bar[i] = 1 / dLseg * np.trapz(sub_dt[i, :], Lsegment)
            if np.min(sub_rho[i, :]) > 0:
                rho_bar[i] = dt_bar[i] / (1 / dLseg * np.trapz(
                        sub_dt[i, :] / sub_rho[i, :], Lsegment))
            else:
                rho_bar[i] = _rho((Lend[i] + Lend[i+1]) / 2)  # take mid-point
            line = [tuple(ss) for ss in sub_segment[i].T]
            polygon[i] = PolyFrame(shapely.geometry.LineString(line).buffer(
                    dt_bar[i]/2, cap_style=2, join_style=2), name='shell')
            x[i] = polygon[i].centroid.x
            z[i] = polygon[i].centroid.y
            dl[i] = dS  # sub-segment length
            dt[i] = dt_bar[i]  # sub-segment thickness
            area[i] = polygon[i].area
        frame = {'x': x, 'z': z, 'dl': dl, 'dt': dt, 'rho': rho_bar,
                 'nturn': area,
                 'poly': polygon, 'section': 'shell'}
        subframe = {'segment': sub_segment, 'rho': sub_rho, 'dt': sub_dt}
        return frame, subframe

    #def getattrs(self, keys: list[str]) -> dict[str, Any]:
    #    """Return default attributes."""
    #    return {key: getattr(self, key) for key in keys}

if __name__ == '__main__':

    shellgrid = ShellGrid([[1, 1.5, 2, 2, 4, 4],
                           [0, 0.1, 0, 1, -1, 0]], -6, 0.1, 1e-6)

    shellgrid.plot_geom()



