"""Manage source and target biotframes."""
from dataclasses import dataclass, field

from nova.electromagnetic.biotframe import BiotFrame
from nova.utilities.pyplot import plt


@dataclass
class BiotSet:
    """Construct Biot source/target frames."""

    source: BiotFrame = field(repr=False, default=None)
    target: BiotFrame = field(repr=False, default=None)
    source_turns: bool = True
    target_turns: bool = False
    reduce_source: bool = True
    reduce_target: bool = False

    def __post_init__(self):
        """Format source and target frames."""
        if not isinstance(self.source, BiotFrame):
            self.source = BiotFrame(self.source)
        if not isinstance(self.target, BiotFrame):
            self.target = BiotFrame(self.target)

    def __len__(self):
        """Return interaction length."""
        return len(self.source) * len(self.target)

    def set_shape(self):
        """Set source and target shapes."""
        self.source.set_target(len(self.target))
        self.target.set_source(len(self.source))

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
