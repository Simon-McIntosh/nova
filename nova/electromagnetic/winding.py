"""Manage 3D coil windings."""
from dataclasses import dataclass, field

import numpy as np

from nova.electromagnetic.coilsetattrs import CoilSetAttrs
from nova.geometry.volume import BoxLoop


@dataclass
class Winding(CoilSetAttrs):
    """Insert 3D coil winding."""

    turn: str = 'shell'
    required: list[str] = field(
        default_factory=lambda: ['x', 'y', 'z', 'dl', 'dt', 'vtk', 'volume'])
    default: dict = field(init=False, default_factory=lambda: {
        'label': 'Shl', 'part': 'shell', 'active': True})

    def set_conditional_attributes(self):
        """Set conditional attrs - not required for shell."""

    def insert(self, *points, required=None, iloc=None, **additional):
        """
        Add 3D coils to frameset.

        Lines described by x, y, z coordinates meshed into n elements based on
        dloop.

        Parameters
        ----------
        *points : array-like, shape(n, 3)
            Loop point data x, y, z coordinates.
        required : list[str]
            Required attribute names (args). The default is None.
        iloc : int, optional
            Index before which coils are inserted. The default is None (-1).
        **additional : dict[str, Any]
            Additional input.
        dl : float
            Box section width (normal) or tube radius.
        dt : float
            Box section depth (cross).
        dl_offset: float
            Box section width centerline offset factor (0-1)

        Returns
        -------
        index : pandas.Index
            FrameSpace index.

        """
        dl = additional.pop('dl', 0.8)
        dt = additional.pop('dt', 0.8)
        dl_offset = additional.pop('dl_offset', 0.5)
        vtk = [BoxLoop(_points, dl, dt, dl_offset) for _points in points]
        x, y, z = np.c_[[_vtk.centerOfMass() for _vtk in vtk]].T
        volume = [_vtk.volume() for _vtk in vtk]

        self.attrs = additional
        with self.insert_required(required):
            index = self.frame.insert(x, y, z, dl, dt, vtk, volume,
                                      iloc=iloc, **self.attrs)
        '''
            self.subframe_insert(index)
        if self.link:
            self.frame.multipoint.link(index)
            self.subframe.multipoint.link(index, expand=True)
        '''
        return index


    '''
    def subframe_insert(self, index):
        """
        Insert subframe(s).

        - Store filaments in subframe.
        - Link turns.

        """
        frame = self.frame.loc[index, :]
        griddata = frame.loc[:, self.required_columns +
                             [attr for attr in self.additional_columns
                              if attr in self.frame]]
        subframe = []
        #subattrs = pandas.DataFrame(self.subattrs, index=index)
        try:
            turncurrent = subattrs.pop('It')
        except KeyError:
            turncurrent = None
        for i, name in enumerate(index):
            #polygrid = PolyGrid(**griddata.iloc[i].to_dict(), **self.grid)
            data = frame.iloc[i].to_dict()
            data |= {'label': name, 'frame': name, 'delim': '_', 'link': True}
            if turncurrent is not None:
                data['It'] = turncurrent.iloc[i] * \
                    polygrid.frame['nturn'] / polygrid.nturn
            subframe.append(self.subframe.assemble(
                polygrid.frame, **data, **subattrs.iloc[i]))
        self.subframe.concatenate(*subframe)


    def insert(self, *args, required=None, iloc=None, **additional):

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
    '''
