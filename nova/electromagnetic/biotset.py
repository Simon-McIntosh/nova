"""Manage source and target biotframes."""
from dataclasses import dataclass, field

import itertools
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
        self.assemble()

    def __len__(self):
        """Return interaction length."""
        return len(self.source) * len(self.target)

    def assemble(self):
        """Assemble BiotSet."""
        self.set_shape()
        self.index = self.get_index()
        self.reduce = self.get_reduce()

    def set_shape(self):
        """Set source and target shapes."""
        self.source.set_target(len(self.target))
        self.target.set_source(len(self.source))

    def get_index(self):
        """Return index. Calculated as product of source and target BiotFrames."""
        return ['_'.join(label) for label in itertools.product(self.source.index,
                                                               self.target.index)]

    def _get_indices(array):



    def get_reduce(self):
        """Return reduction index."""
        frame = np.array(self.source.frame)
        if (frame == self.source.metaframe.default['frame']).all():
            return np.arange(len(self.source))

            _name = coil[0]
            _reduction_index = [0]
            for i, name in enumerate(coil):
                if name != _name:
                    _reduction_index.append(i)
                    _name = name
        self._reduction_index = np.array(_reduction_index)
        self._plasma_iloc = np.arange(len(self._reduction_index))[
            self.plasma[self._reduction_index]]
        filament_indices = np.append(self._reduction_index, self.coil_number)
        plasma_filaments = filament_indices[self._plasma_iloc+1] - \
            filament_indices[self._plasma_iloc]
        self._plasma_reduction_index = \
            np.append(0, np.cumsum(plasma_filaments)[:-1])

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
