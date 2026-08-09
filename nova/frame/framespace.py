"""Frame with subspace projection, selection and geometry metamethods.

``FrameSpace`` adds the independent-degree-of-freedom subspace on top of
:class:`~nova.frame.framelink.FrameLink`: reads of a subspace column inflate
from the independent rows through the ``subref`` / ``factor`` mapping, and
direct writes are rejected in favour of the subspace / sloc accessors. It also
carries the polygon and vtk geometry metamethods and the array accessor.
"""

from __future__ import annotations

from functools import cached_property
import os

import numpy as np

from nova.frame.arraylocindexer import ArrayLocIndexer
from nova.frame.columnar import Vector
from nova.frame.error import SpaceKeyError
from nova.frame.framelink import FrameLink
from nova.frame.metamethod import PolyGeo, PolyPlot, VtkGeo, VtkPlot
from nova.frame.subspace import SubSpace

# pylint: disable=too-many-ancestors


class FrameSpace(FrameLink):
    """Columnar frame with subspace projection and geometry metamethods."""

    def __init__(self, data=None, index=None, columns=None, attrs=None, **metadata):
        """Build the frame, load geometry, then build the subspace."""
        previous = self.__dict__.get("attrs", {}).get("subspace")
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(PolyGeo, PolyPlot)
        if os.environ.get("NOVA_VTK", "True") != "False":
            self.frame_attrs(VtkGeo, VtkPlot)
        if previous is not None and previous is not data:
            # the rebuilt frame gets a fresh subspace; poison the detached one
            # so a caller's held sloc view reads as invalid, not silently stale
            self._poison_store(previous._store)
        self.attrs["subspace"] = SubSpace(self)

    def _subspace_active(self, col) -> bool:
        """Return True when col reads / writes must route through the subspace."""
        return (
            self.__dict__.get("_store") is not None
            and self.hasattrs("subspace")
            and self.hascol("subspace", col)
            and self.lock("subspace") is False
        )

    def __getitem__(self, col):
        """Inflate a subspace column (or It via Ic) before returning it."""
        if self.__dict__.get("_store") is not None and self.hasattrs("subspace"):
            if self._subspace_active(col):
                self.inflate_subspace(col)
            elif col == "It" and self._subspace_active("Ic"):
                self.inflate_subspace("Ic")
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Reject direct writes to a protected subspace column."""
        if self._subspace_active(col):
            raise SpaceKeyError("loc", col)
        if col == "It" and self._energized("It") and self._subspace_active("Ic"):
            # It couples onto a subspace-protected Ic; divert the derived current
            # onto the independent-row subspace rather than the locked frame
            current = np.asarray(value, dtype=float) / np.asarray(
                self["nturn"], dtype=float
            )
            self._set_subspace_current(current)
            return
        super().__setitem__(col, value)

    def _set_subspace_current(self, current):
        """Write a frame-aligned current array onto the subspace Ic column."""
        current = np.broadcast_to(np.asarray(current, dtype=float), len(self))
        positions = self.index.get_indexer(self.subspace.index)
        with self.subspace.setlock(True, "subspace"):
            self.subspace["Ic"] = current[positions]

    def __getattr__(self, name):
        """Inflate a subspace column accessed as an attribute."""
        if not name.startswith("_") and self.__dict__.get("_store") is not None:
            if name != "It" and self._subspace_active(name):
                self.inflate_subspace(name)
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        """Reject direct attribute writes to a protected subspace column."""
        if not name.startswith("_") and self.__dict__.get("_store") is not None:
            if self._subspace_active(name):
                raise SpaceKeyError("loc", name)
        super().__setattr__(name, value)

    @property
    def subspace(self) -> SubSpace:
        """Return the independent-degree-of-freedom subspace frame."""
        return self.attrs["subspace"]

    def update_frame(self):
        """Propagate subspace columns onto the full frame before display."""
        if self.hasattrs("subspace"):
            for col in [col for col in self.subspace.columns if col in self]:
                self.inflate_subspace(col)
        super().update_frame()

    def inflate_subspace(self, col):
        """Project a subspace column onto the full frame via subref / factor."""
        with self.setlock(False, "subspace"):
            value = np.asarray(self.subspace[col])
        try:
            subref = np.asarray(self["subref"], dtype=int)
            value = value[subref]
            if col == "Ic":
                value = value * np.asarray(self["factor"], dtype=float)
        except (KeyError, IndexError, TypeError):
            pass
        with self.setlock(True, "subspace"):
            super().__setitem__(col, value.view(Vector))

    @cached_property
    def aloc(self) -> ArrayLocIndexer:
        """Return the live array-column accessor."""
        return ArrayLocIndexer("array", self)


if __name__ == "__main__":
    framespace = FrameSpace(
        base=["x", "y", "z"],
        required=["x", "z"],
        available=["It", "poly"],
        Subspace=["Ic"],
        Array=["Ic"],
    )
    framespace.insert(range(4), 1, Ic=6.5, name="PF1", part="PF", active=False)
