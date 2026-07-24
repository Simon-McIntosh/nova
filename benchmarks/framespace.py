"""Benchmark framespace."""

import os
import tempfile
import timeit

import numpy as np

from nova.frame.framespace import FrameSpace

# Fixed absolute cache path: asv re-imports this module in a separate process
# for setup_cache and for each benchmark, so the path must resolve identically
# in every process (a randomised mkdtemp would differ per process).
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "nova_asv_framespace")
os.makedirs(_CACHE_DIR, exist_ok=True)
_CACHE_FILE = os.path.join(_CACHE_DIR, "framespace.nc")


class Current:
    """Benchmark current read and write."""

    number = 10_000
    timer = timeit.default_timer

    @property
    def filename(self):
        """Return coilset filename."""
        return _CACHE_FILE

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
