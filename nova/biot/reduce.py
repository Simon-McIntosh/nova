"""Calculate reduction indices."""

from dataclasses import dataclass, field

import numpy as np

import nova.frame.metamethod as metamethod
from nova.frame.dataframe import DataFrame


@dataclass
class Reduce(metamethod.Reduce):
    """Calculate reduction indices for reduceat."""

    name = "biotreduce"

    frame: DataFrame = field(repr=False)
    index: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    indices: np.ndarray = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=int)
    )
    link: dict[int, list] = field(init=False, repr=False, default_factory=dict)
    reduce: bool = field(default=False)

    def initialize(self):
        """Calculate biot reduction indices."""
        self.indices = np.asarray(self.reduction_indices(), dtype=int)
        self.index = np.asarray(self.frame.index)[self.indices]
        if len(self.link) > 0:
            self.index = np.delete(self.index, list(self.link))
        self.reduce = len(self.indices) < len(self.frame)

    def reduction_indices(self):
        """Return reduction indices, construct link if ref not monotonic."""
        if "ref" not in self.frame:
            return np.arange(len(self.frame))
        ref = np.array(self.frame.ref)
        factor = np.array(self.frame.factor)
        if np.all(ref[:-1] <= ref[1:]) and np.all(factor == 1):  # monotonic
            return np.unique(ref)
        indices = [ref[0]]  # seed list
        for i, index in enumerate(ref):
            if factor[i] == 1:
                if index == indices[-1]:
                    continue
                if index > indices[-1] and factor[i] == 1:
                    indices.append(index)
                    continue
            if i != indices[-1]:
                indices.append(i)
            if (len(indices) - 1) != int(indices.index(index)):
                self.link[len(indices) - 1] = [int(indices.index(index)), factor[i]]
        return indices
