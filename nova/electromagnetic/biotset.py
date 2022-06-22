"""Manage source and target biotframes."""
from dataclasses import dataclass, field
from typing import Union

import itertools
from nova.electromagnetic.biotframe import BiotFrame
from nova.utilities.pyplot import plt


@dataclass
class BiotSet:
    """
    Construct Biot source/target frames.

    Parameters
    ----------
    source: BiotFrame
        Field source

    target: BiotFrame
        Calculation target

    turns: list[bool, bool]
        Multiply columns / rows by turn number (source / target) if True.

    reduce: list[bool, bool]
        Apply linked turn reduction (source / target) if True.

    """

    source: BiotFrame = field(repr=False, default=None)
    target: Union[dict, BiotFrame] = field(repr=False, default=None)
    turns: list[bool] = field(default_factory=lambda: [True, False])
    reduce: list[bool] = field(default_factory=lambda: [True, True])

    def __post_init__(self):
        """Format source and target frames."""
        self.source = BiotFrame(self.source)
        self.target = BiotFrame(self.target, additional=[], available=[])
        self.set_flags()
        self.assemble()

    def __len__(self):
        """Return interaction length."""
        return len(self.source) * len(self.target)

    def set_flags(self):
        """Set turn and reduction flags on source and target BiotFrames."""
        if isinstance(self.turns, bool):
            self.turns = [self.turns, self.turns]
        if isinstance(self.reduce, bool):
            self.reduce = [self.reduce, self.reduce]
        self.source.turns = self.turns[0]
        self.target.turns = self.turns[1]
        self.source.reduce = self.reduce[0]
        self.target.reduce = self.reduce[1]

    def assemble(self):
        """Assemble BiotSet."""
        self.set_shape()
        self.update_index()

    def set_shape(self):
        """Set source and target shapes."""
        self.source.set_target(len(self.target))
        self.target.set_source(len(self.source))

    def update_index(self):
        """Update index. Product of source and target BiotFrames."""
        self.index = ['_'.join(label) for label
                      in itertools.product(self.source.index,
                                           self.target.index)]

    def plot(self, axes=None):
        """Plot source and target markers."""
        if axes is None:
            axes = plt.gca()
        self.source.plot('x', 'z', 'scatter', ax=axes,
                         color='C1', marker='o', label='source')
        self.target.plot('x', 'z', 'scatter', ax=axes,
                         color='C2', marker='.', label='target')
        axes.axis('equal')
        axes.axis('off')


if __name__ == '__main__':

    source = BiotFrame({'x': [3, 3.4, 3.6], 'z': [3.1, 3, 3.3],
                        'dl': 0.3, 'dt': 0.3, 'section': 'hex'})
    biotset = BiotSet(source, source)
    biotset.plot()
