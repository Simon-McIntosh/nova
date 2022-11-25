"""Build poloidal field coil from individual turns."""
from dataclasses import dataclass, field

import numpy as np
import pandas
import shapely

from nova.frame.coilsetattrs import CoilSetAttrs


@dataclass
class TurnGeom:
    """Derive frame geometory from subframe turns."""

    frame: pandas.DataFrame
    data: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Calculate average geometrical parameters."""
        self.data['area'] = self.frame.area.sum()
        for attr in 'xyz':
            self.data[attr] = self.weighted_mean(attr)
        if self.frame.segment[0] == 'ring':  # update ring circumference
            self.data['dy'] = 2*np.pi*self.data['x']
        self.data['rms'] = self.weighted_rms()
        self.data['nturn'] = self.frame.nturn.sum()
        self.data['turn'] = self.frame.section[0]
        self.data['dl'] = self.frame['dl'][0]
        self.data['dt'] = self.frame['dt'][0]

    @property
    def area(self):
        """Return total cross-sectional area."""
        return self.data['area']

    @property
    def columns(self) -> list[str]:
        """Return data keys."""
        return list(self.data.keys())

    def weighted_mean(self, attr):
        """Return area weighted mean."""
        return np.sum(self.frame[attr] * self.frame.area) / self.area

    def weighted_rms(self):
        """Return area weighted rms."""
        return np.sqrt(np.sum(self.frame.area * self.frame.rms**2) / self.area)


@dataclass
class Turn(CoilSetAttrs):
    """Construct single coil from turns."""

    turn: str = 'skin'
    required: list[str] = field(default_factory=lambda: ['x', 'z', 'dl', 'dt'])
    default: dict = field(init=False, default_factory=lambda: {
        'label': 'Coil', 'part': 'coil', 'active': True})

    def set_conditional_attributes(self):
        """Set conditional attrs - not required for turn."""

    def insert(self, *args, required=None, iloc=None, **additional):
        """
        Add turn bundle to frameset.

        Lines described by x, z coordinates meshed into n coils based on
        dshell. Each frame is meshed based on delta.

        Parameters
        ----------
        x : array-like, shape(n,)
            x-coordinates of turn geometric centers.
        z : array-like, shape(n,)
            z-coordinates of turn geometric centers.
        dl : float
            Turn diameter.
        dt : float
            Skin fraction.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        self.attrs = additional
        name = self.attrs.get(
            'name', self.frame.build_index(1, name=self.attrs['label'])[0])
        self.attrs['name'] = name
        subattrs = {'section': self.turn, 'label': name, 'frame': name,
                    'delim': '_', 'link': True} | self.attrs
        with self.insert_required(required):
            subindex = self.subframe.insert(*args, **subattrs)
        poly = shapely.geometry.MultiPolygon(
            [polygon.poly for polygon in self.subframe.poly[subindex]])
        attrs = {attr: self.attrs[attr] for attr in self.attrs if not
                 isinstance(self.attrs[attr], (dict, list, np.ndarray))}
        index = self.frame.insert(poly, iloc=iloc, **attrs)
        geom = TurnGeom(self.subframe.loc[subindex, :])
        self.frame.loc[index, geom.columns] = geom.data.values()
        self.update_loc_indexer()
        return index
