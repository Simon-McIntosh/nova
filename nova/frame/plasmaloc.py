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
        assert False, "called check cached"
        self.check("psi")
        if self.version[attr] != self.version["psi"]:
            self._clear_cache([attr])
            self.version[attr] = self.version["psi"]

    @cached_property
    def slice_(self):
        """Return plasma filament slice."""
        start = bisect(self.aloc["plasma"], True)
        number = bisect_right(~self.aloc["plasma"][start:], False)
        if number != np.sum(self.aloc["plasma"]):
            raise IndexError("plasma filaments are non-contiguous.")
        return slice(start, start + number)

    @cached_property
    def nturn_(self):
        """Return a view of the plasma's nturn array."""
        return self.aloc["nturn"][self.slice_]

    @cached_property
    def ionize_(self):
        """Return a view of the plasma's ionize array."""
        return self.aloc["ionize"][self.slice_]

    @cached_property
    def area_(self):
        """Return a view of the plasma's area array."""
        return self.aloc["area"][self.slice_]

    @cached_property
    def radius_(self):
        """Return a view of the plasma's radial filiment cooridnate."""
        return self.aloc["x"][self.slice_]

    @cached_property
    def height_(self):
        """Return a view of the plasma's vertical filiment cooridnate."""
        return self.aloc["z"][self.slice_]

    @property
    def nturn(self):
        """Manage ionized plasma turn attribute."""
        return self.nturn_[self.ionize]

    @nturn.setter
    def nturn(self, nturn):
        self.nturn_[self.ionize] = nturn
        self.update_aloc_hash("nturn")

    @property
    def ionize(self):
        """Manage plasma ionization property."""
        return self.ionize_

    @ionize.setter
    def ionize(self, mask):
        self.nturn_[:] = 0
        self.ionize_[:] = mask
        self._clear_cache(["area", "radius", "height"])

    @cached_property
    def area(self):
        """Return ionized area."""
        return self.area_[self.ionize_]

    @cached_property
    def radius(self):
        """Return ionized radius."""
        return self.radius_[self.ionize_]

    @cached_property
    def height(self):
        """Return ionized radius."""
        return self.height_[self.ionize_]
