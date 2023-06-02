"""Benchmark framespace."""
import os
import timeit

import numpy as np

from nova.frame.framespace import FrameSpace


class Current:
    """Benchmark current read and write."""

    number = 10_000
    timer = timeit.default_timer

    @property
    def filename(self):
        """Return coilset filename."""
        return "./framecurrent_frame.nc"

    def setup_cache(self):
        """Build reference coilset."""
        framespace = FrameSpace(
            base=["x", "y", "z"],
            required=["x", "z"],
            available=["It", "poly"],
            Subspace=["Ic"],
            Array=["Ic"],
        )
        framespace.insert(range(40), 1, Ic=6.5, name="PF1", part="PF", active=False)
        framespace.subspace.Ic = np.random.rand(len(framespace.subspace))
        framespace.store(self.filename)

    def remove(self):
        """Remove coilset."""
        os.remove(self.filename)

    def setup(self):
        """Load coilset from file."""
        self.framespace = FrameSpace().load(self.filename)


class SetCurrent(Current):
    """Benchmark current update methods."""

    def setup(self):
        """Extend Current.setup to extract current vector."""
        super().setup()
        self.current = self.framespace.subspace.Ic.copy()

    def time_metaframe_data(self):
        """Time direct metaframe data update."""
        self.framespace.subspace.metaframe.data["Ic"] = self.current

    def time_subspace(self):
        """Time update to frame subspace."""
        self.framespace.subspace.Ic = self.current


class GetCurrent(Current):
    """Benchmark current access methods."""

    def time_loc(self):
        """Time current access via loc method."""
        return self.framespace.loc[:, "Ic"]

    def time_getitem(self):
        """Time current access via getitem method."""
        return self.framespace["Ic"]

    def time_getattr(self):
        """Time current access via getattr method."""
        return self.framespace.Ic

    def time_getattr_subspace(self):
        """Time current access via getattr method."""
        return self.framespace.subspace.Ic


if __name__ == "__main__":
    setcurrent = SetCurrent()
    setcurrent.setup_cache()
    setcurrent.setup()
    print(setcurrent.framespace)
    setcurrent.remove()
