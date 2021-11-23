"""Mesh poloidal shells."""
from dataclasses import dataclass, field

from nova.electromagnetic.coilsetattrs import GridAttrs
from nova.electromagnetic.shellgrid import ShellGrid
from nova.geometry.polygeom import Polygon


@dataclass
class Shell(GridAttrs):
    """Mesh poloidal shell elements."""

    turn: str = 'shell'
    required: list[str] = field(default_factory=lambda: ['x', 'z', 'dl', 'dt'])
    default: dict = field(init=False, default_factory=lambda: {
        'label': 'Shl', 'part': 'shell', 'active': False})

    def set_conditional_attributes(self):
        """Set conditional attrs - not required for shell."""

    def insert(self, *args, required=None, iloc=None, **additional):
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
        if isinstance(args[1], (int, float)):
            poly = Polygon(args[0]).poly
            args = poly.boundary.xy + args[1:]

        self.attrs = additional
        with self.insert_required(required):
            shellgrid = ShellGrid(*args, delta=self.attrs['delta'])
            index = self.frame.insert(shellgrid.frame, iloc=iloc, **self.attrs)
        frame = self.frame.loc[index, :]
        subframe = []
        for i, name in enumerate(index):
            data = frame.iloc[i].to_dict()
            data |= {'label': name, 'frame': name, 'delim': '_', 'link': True}
            subframe.append(self.subframe.assemble(
                shellgrid.subframe[i], **data, **self.subattrs))
        self.subframe.concatenate(*subframe)
