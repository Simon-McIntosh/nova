
from dataclasses import dataclass, field
from typing import Union, Any

import numpy as np
import pandas
import shapely.geometry
import shapely.strtree
import scipy.interpolate

from nova.electromagnetic.frame import Frame
from nova.utilities import geom


@dataclass
class Shell:
    """Mesh poloidal shell elements."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)
    dshell: float
    delta: float

    def insert(self, x, z, dt, rho=0, **additional):
        """
        Add shell elements to frameset.

        Lines described by x, z coordinates meshed into n coils based on
        dshell. Each frame is meshed based on delta.

        Parameters
        ----------
        x : array-like, shape(n,)
            x-coordinates of poloidal line to be meshed.
        z : array-like, shape(n,)
            z-coordinates of poloidal line to be meshed.
        dt : float
            Shell thickness.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        attrs = {'label': 'Shl', 'delim': '', 'section': 'shell',
                 'turn': 'shell', 'turn_fraction': 1, 'active': False,
                 'dshell': self.dshell, 'delta': self.delta}
        attrs |= additional
        shell, subshell = self._mesh(
            (x, z), dt, rho, attrs.pop('dshell'))
        additional = shell | attrs
        required = [additional.pop(attr)
                    for attr in self.frame.metaframe.required]
        index = self.frame.insert(*required, **additional)

        attrs |= {'delim': '_'}
        subindex = [[] for __ in range(len(index))]
        for i, frame in enumerate(index):
            shell = self._mesh(
                subshell['segment'][i], subshell['dt'][i],
                subshell['rho'][i], attrs['delta'])[0]
            additional = shell | attrs
            additional |= {'frame': frame, 'label': frame, 'link': True}
            required = [additional.pop(attr)
                        for attr in self.frame.metaframe.required]
            subindex[i] = self.subframe.insert(*required, **additional)

    @staticmethod
    def _mesh(segment, dt, rho, dS):
        if not pandas.api.types.is_list_like(dt):  # variable thickness
            dt *= np.ones(np.shape(segment)[1])
        if not pandas.api.types.is_list_like(rho):  # variable resistivity
            rho *= np.ones(np.shape(segment)[1])
        dL = geom.length(*segment, norm=False)  # cumulative segment length
        L = dL[-1]  # total segment length
        if dS == 0:
            dS = L
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
        _x, _z = geom.xzfun(*segment)  # xz interpolators
        _dt = scipy.interpolate.interp1d(dL/L, dt)
        _rho = scipy.interpolate.interp1d(dL/L, rho)
        Lend = np.linspace(0, 1, nS+1)  # endpoints
        polygon = [[] for __ in range(nS)]
        x, z = np.zeros(nS), np.zeros(nS)
        rho_bar, dt_bar = np.zeros(nS), np.zeros(nS)
        dl, dt, dA = np.zeros(nS), np.zeros(nS), np.zeros(nS)
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
            polygon[i] = shapely.geometry.LineString(line).buffer(
                    dt_bar[i]/2, cap_style=2, join_style=2)
            x[i] = polygon[i].centroid.x
            z[i] = polygon[i].centroid.y
            dl[i] = dS  # sub-segment length
            dt[i] = dt_bar[i]  # sub-segment thickness
            dA[i] = polygon[i].area
        frame = {'x': x, 'z': z, 'dl': dl, 'dt': dt, 'rho': rho_bar, 'Nt': dA,
                 'poly': polygon}
        subframe = {'segment': sub_segment, 'rho': sub_rho, 'dt': sub_dt}
        return frame, subframe

    def getattrs(self, keys: list[str]) -> dict[str, Any]:
        """Return default attributes."""
        return {key: getattr(self, key) for key in keys}
