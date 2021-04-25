"""Mesh poloidal shells."""
from dataclasses import dataclass, field

from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.gridattrs import GridAttrs
from nova.electromagnetic.shellgrid import ShellGrid
from nova.electromagnetic.polygon import Polygon


@dataclass
class Shell(GridAttrs):
    """Mesh poloidal shell elements."""

    frame: FrameSpace = field(repr=False)
    subframe: FrameSpace = field(repr=False)
    delta: float
    turn: str = 'shell'
    default: dict = field(init=False, default_factory=lambda: {
        'label': 'Shl', 'part': 'shell', 'active': False})

    def set_conditional_attributes(self):
        """Set conditional attrs - not required for shell."""

    def insert(self, *required, iloc=None, **additional):
        """
        Add shell elements to frameset.

        Lines described by x, z coordinates meshed into n coils based on
        dshell. Each frame is meshed based on delta.

        Parameters
        ----------
        x : array-like, shape(n,)
            x-coordinates of poloidal line to be meshed.
        z : array-like, shape(n,)
            z-coordinates of poloidal line to be meshed.
        dl : float
            Shell length.
        dt : float
            Shell thickness.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        if isinstance(required[1], (int, float)):
            poly = Polygon(required[0]).poly
            required = poly.boundary.xy + required[1:]

        self.attrs = additional
        shellgrid = ShellGrid(*required, delta=self.attrs['delta'])
        index = self.frame.insert(shellgrid.frame, iloc=iloc, **self.attrs)
        frame = self.frame.loc[index, :]
        subframe = []
        for i, name in enumerate(index):
            data = frame.iloc[i].to_dict()
            data |= {'label': name, 'frame': name, 'delim': '_', 'link': True}
            subframe.append(self.subframe.assemble(
                shellgrid.subframe[i], **data, **self.subattrs))
        self.subframe.concatenate(*subframe)
