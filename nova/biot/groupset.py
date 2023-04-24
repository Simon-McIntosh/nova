"""Manage source and target frames."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.biotframe import BiotFrame
from nova.frame.baseplot import Plot


@dataclass
class GroupSet(Plot):
    """
    Construct Biot source/target biot frames.

    Parameters
    ----------
    source: BiotFrame
        Field source

    target: BiotFrame
        Calculation target

    turns: list[bool, bool]
        Multiply columns / rows by turn number (source, target) if True.

    reduce: list[bool, bool]
        Apply linked turn reduction (source, target) if True.

    """

    source: BiotFrame = field(repr=False, default_factory=BiotFrame)
    target: BiotFrame = field(repr=False, default_factory=BiotFrame)
    turns: list[bool] = field(default_factory=lambda: [True, False])
    reduce: list[bool] = field(default_factory=lambda: [True, True])

    def __post_init__(self):
        """Format source and target biot frames."""
        if not isinstance(self.source, BiotFrame):
            self.source = BiotFrame(self.source)
        if not isinstance(self.target, BiotFrame):
            self.target = BiotFrame(self.target, available=[])
        self.set_flags()
        self.assemble()
        super().__post_init__()

    def __len__(self):
        """Return interaction length."""
        return np.prod(self.shape)

    @property
    def shape(self):
        """Return interaction matrix shape."""
        return len(self.target), len(self.source)

    def set_flags(self):
        """Set turn and reduction flags on source and target biot frames."""
        if isinstance(self.turns, bool):
            self.turns = [self.turns, self.turns]
        if isinstance(self.reduce, bool):
            self.reduce = [self.reduce, self.reduce]
        self.source.turns = self.turns[0]
        self.target.turns = self.turns[1]
        self.source.reduce = self.reduce[0]
        self.target.reduce = self.reduce[1]

    def assemble(self):
        """Assemble GroupSet."""
        self.set_shape()
        self.update_index()

    def set_shape(self):
        """Set source and target shapes."""
        self.source.set_target(len(self.target))
        self.target.set_source(len(self.source))

    def update_index(self):
        """Update index. Product of source and target biot frames."""
        self.index = range(len(self))
        # self.index = ['_'.join(label) for label
        #               in itertools.product(self.source.index,
        #                                    self.target.index)]

    def plot(self, axes=None):
        """Plot source and target markers."""
        self.axes = axes
        self.source.plot('x', 'z', 'scatter', ax=self.axes,
                         color='C1', marker='o', label='source')
        self.target.plot('x', 'z', 'scatter', ax=self.axes,
                         color='C2', marker='.', label='target')
        self.axes.axis('equal')
        self.axes.axis('off')


if __name__ == '__main__':

    source = BiotFrame({'x': [3, 3.4, 3.6], 'z': [3.1, 3, 3.3],
                        'dl': 0.3, 'dt': 0.3, 'section': 'hex'})
    groupset = GroupSet(source, source)
    groupset.plot()
