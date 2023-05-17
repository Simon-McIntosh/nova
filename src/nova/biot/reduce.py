"""Calculate reduction indices."""
from dataclasses import dataclass, field

import numpy as np
import pandas

import nova.frame.metamethod as metamethod
from nova.frame.dataframe import DataFrame


@dataclass
class Reduce(metamethod.Reduce):
    """Calculate reduction indices for reduceat."""

    name = 'biotreduce'

    frame: DataFrame = field(repr=False)
    index: pandas.Index = field(default=pandas.Index([]), repr=False)
    indices: list[int] = field(init=False, repr=False, default_factory=list)
    link: dict[int, int] = field(init=False, repr=False, default_factory=dict)
    reduce: bool = field(default=False)

    def initialize(self):
        """Calculate biot reduction indexies."""
        self.indices = self.reduction_indices()
        self.index = self.frame.index[self.indices]
        if len(self.link) > 0:
            self.index = self.index.drop(self.index[list(self.link)])
        self.reduce = len(self.indices) < len(self.frame)

    def reduction_indices(self):
        """Return reduction indices, construct link if ref not monotonic."""
        if 'ref' not in self.frame:
            return range(len(self.frame))
        ref = np.array(self.frame.ref)
        factor = np.array(self.frame.factor)
        if np.all(ref[:-1] <= ref[1:]) and np.all(factor == 1):  # monotonic
            return np.unique(ref)
        indices = [ref[0]]  # sead list
        for i, index in enumerate(ref):
            if factor[i] == 1:
                if index == indices[-1]:
                    continue
                if index > indices[-1] and factor[i]:
                    indices.append(index)
                    continue
            indices.append(i)
            self.link[len(indices)-1] = [int(indices.index(index)), factor[i]]
        return indices
