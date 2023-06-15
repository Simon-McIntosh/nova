"""Biot plot methods."""

from nova.graphics import plot
from nova.frame.framesetloc import FrameSetLoc


class Plot1D(plot.Plot1D, FrameSetLoc):
    """Plot baseclass for framespace derived instances."""

    def bar(
        self,
        attr: str,
        index=slice(None),
        axes=None,
        scale=1,
        label=None,
        limit=None,
        **kwargs,
    ):
        """Plot coil attributes."""
        self.set_axes("1d", axes)
        if isinstance(index, str):
            index = [name in self.loc[index, :].index for name in self.coil_name]
        names = self.coil_name[index]
        self.axes.bar(names, scale * getattr(self, attr)[index], **kwargs)
        self.axes.set_xticks(range(len(names)))
        self.axes.set_xticklabels(names, rotation=90, ha="center")
        if limit is not None:
            self.axes.set_ylim(limit)
        if label is not None:
            self.axes.set_ylabel(label)
