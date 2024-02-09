"""Manage attribute access for classes implementing a plasma grid interface."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from nova.frame.framesetloc import FrameSetLoc
from nova.geometry.select import bisect, bisect_right


@dataclass
class PlasmaLoc(FrameSetLoc):
    """Plasma grid Loc indexer."""

    def __len__(self):
        """Return number of plasma filaments."""
        return np.sum(self.aloc["plasma"])

    def __str__(self):
        """Return string representation of plasma subframe."""
        return self.loc[
            "ionize", ["x", "z", "section", "area", "Ic", "It", "nturn"]
        ].__str__()

    def check_cached(self, attr):
        """Check validity of cached attribute."""
        self.check("psi")
        if self.version[attr] != self.version["psi"]:
            self._clear_cache([attr])
            self.version[attr] = self.version["psi"]

    @cached_property
    def _slice(self):
        """Return plasma filament slice."""
        start = bisect(self.aloc["plasma"], True)
        number = bisect_right(~self.aloc["plasma"][start:], False)
        if number != np.sum(self.aloc["plasma"]):
            raise IndexError("plasma filaments are non-contiguous.")
        return slice(start, start + number)

    @cached_property
    def _nturn(self):
        """Return a view of the plasma's nturn array."""
        return self.aloc["nturn"][self._slice]

    @cached_property
    def _ionize(self):
        """Return a view of the plasma's ionize array."""
        return self.aloc["ionize"][self._slice]

    @cached_property
    def _area(self):
        """Return a view of the plasma's area array."""
        return self.aloc["area"][self._slice]

    @cached_property
    def _radius(self):
        """Return a view of the plasma's radial filiment cooridnate."""
        return self.aloc["x"][self._slice]

    @cached_property
    def _height(self):
        """Return a view of the plasma's vertical filiment cooridnate."""
        return self.aloc["z"][self._slice]

    @property
    def nturn(self):
        """Manage ionized plasma turn attribute."""
        return self._nturn[self.ionize]

    @nturn.setter
    def nturn(self, nturn):
        self._nturn[self.ionize] = nturn
        self.update_aloc_hash("nturn")

    @property
    def ionize(self):
        """Manage plasma ionization property."""
        return self._ionize

    @ionize.setter
    def ionize(self, mask):
        self._nturn[:] = 0
        self._ionize[:] = mask
        self._clear_cache(["area", "radius", "height"])

    @cached_property
    def area(self):
        """Return ionized area."""
        return self._area[self._ionize]

    @cached_property
    def radius(self):
        """Return ionized radius."""
        return self._radius[self._ionize]

    @cached_property
    def height(self):
        """Return ionized radius."""
        return self._height[self._ionize]
