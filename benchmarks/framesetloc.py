"""Benchmark frameset loc indexer methods."""

import os
import timeit

import numpy as np

from nova.frame.coilset import CoilSet


class SubFrameLoc:
    """Cache reference frameset."""

    number = 10_000
    params = (["Ic", "nturn", "x"], ["loc", "sloc", "aloc", "saloc"])
    param_names = ["attr", "indexer"]
    timer = timeit.default_timer
    filename = "./frameset"

    def setup_cache(self):
        """Build reference coilset."""
        coilset = CoilSet(
            dcoil=-100, dplasma=-500, array=["Ic", "nturn", "fix", "free", "plasma"]
        )
        coilset.coil.insert(5.5, [-2, -1, 1], 0.5, 0.75, label="PF", free=True)
        coilset.coil.insert(4.5, [-3.5, 2.5], 0.5, 0.75, label="PF")
        coilset.coil.insert(3, range(-3, 3), 0.4, 0.9, label="CS", free=True)
        coilset.linkframe(["CS2", "CS3"])
        coilset.firstwall.insert({"ellip": [4.2, -0.4, 1.25, 3.2]}, turn="hex")
        coilset.saloc["Ic"] = range(len(coilset.sloc))
        coilset.store(self.filename)

    def setup(self, attr, indexer):
        """Load coilset and set indexer."""
        self.coilset = CoilSet().load(self.filename)
        if "a" in indexer and attr not in self.coilset.array:
            raise NotImplementedError
        if "s" in indexer and attr not in self.coilset.subspace:
            raise NotImplementedError
        self.indexer = getattr(self.coilset, indexer)
        self.data = np.arange(len(self.indexer[attr]))
        self.plasma_index = self.indexer["plasma"]
        self.plasma_data = self.data[self.plasma_index]

    def remove(self):
        """Remove coilset."""
        os.remove(self.filename + ".nc")


class GetSubFrameLoc(SubFrameLoc):
    """Benchmark subframe loc indexer."""

    def time_item(self, attr, indexer):
        """Time attribute access."""
        return self.indexer[attr]

    def time_subitem(self, attr, indexer):
        """Time subset data access."""
        if "aloc" in indexer:
            return self.indexer[attr][self.plasma_index]
        return self.indexer["plasma", attr]


class SetSubFrameLoc(SubFrameLoc):
    """Benchmark subframe loc indexer."""

    def setup(self, attr, indexer):
        """Extend setup to exclude array and subspace incompatabilities."""
        super().setup(attr, indexer)
        if attr in self.coilset.subspace and "s" not in indexer:
            raise NotImplementedError

    def time_item(self, attr, indexer):
        """Time attribute update."""
        self.indexer[attr] = self.data

    def time_subitem(self, attr, indexer):
        """Time plasma attribute update."""
        if "aloc" in indexer:
            data = self.indexer[attr]
            data[self.plasma_index] = self.plasma_data
            return
        self.indexer["plasma", attr] = self.plasma_data


if __name__ == "__main__":
    attr, indexer = "nturn", "aloc"

    frameset = SetSubFrameLoc()
    frameset.setup_cache()
    frameset.setup(attr, indexer)
    frameset.time_item(attr, indexer)
    frameset.time_subitem(attr, indexer)
    frameset.coilset.plot()
    frameset.remove()
