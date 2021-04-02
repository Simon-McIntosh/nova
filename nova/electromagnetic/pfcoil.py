
from dataclasses import dataclass, field

import numpy as np
import pandas
import shapely.geometry
import shapely.strtree

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.polygen import polygen


@dataclass
class PFcoil:
    """Mesh poloidal field coils (CS and PF)."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)
    delta: float

    def insert(self, *required, iloc=None, mesh=True, **additional):
        """
        Add poloidal field coil(s).

        Parameters
        ----------
        *required : Union[DataFrame, dict, list]
            Required input.
        iloc : int, optional
            Index before which coils are inserted. The default is None (-1).
        mesh : bool, optional
            Mesh coil. The default is True.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        additional = {'delta': self.delta} | additional
        index = self.frame.insert(*required, iloc=iloc, **additional)
        if mesh:
            self._mesh(index=index)

    def _mesh(self, index=None, link=True, **kwargs):
        """
        Mesh poloidal field coil(s). Store filaments in subframe. Link turns.

        Parameters
        ----------
        index : int or list or pandas.Index, optional
            Index of coils to be meshed. The default is None (all coils).
        link : bool, optional
            create mpc constraints between subcoils. The default is True.
        **kwargs : dict
            Optional arguments:

                - frame : Frame. The default is self.frame.
                - subframe : Frame. The default is self.subframe.
                - dcoil : float. Subcoil filament dimension.
                - ...

        Returns
        -------
        None.

        """
        frame = kwargs.pop('frame', self.frame)
        subframe = kwargs.pop('subframe', self.subframe)
        if index is None:  # re-mesh all coils
            index = frame.index
        _subframe = [[] for __ in range(len(index))]
        for i, name in enumerate(index):
            if 'dpol' in kwargs:
                frame.loc[name, 'delta'] = kwargs['dpol']
            for key in kwargs:
                if key in frame:
                    frame.loc[name, key] = kwargs[key]
            if 'subindex' in frame:  # drop existing subframes
                if isinstance(frame.loc[name, 'subindex'], list):
                    subframe.drop(frame.loc[name, 'subindex'], inplace=True)
            mesh = self._mesh_single(
                frame.loc[name, :], link=link,  **kwargs)  # single coil
            subframe_args, subframe_kwargs = [], {}
            for var in mesh:
                if var in subframe.metaframe.required:
                    subframe_args.append(mesh[var])
                elif var in subframe.metaframe.additional:
                    subframe_kwargs[var] = mesh[var]
            _subframe[i] = subframe.assemble(
                    *subframe_args, label=name, frame=name, **subframe_kwargs)
            # back-propagate fillament attributes to frame
            frame.loc[name, ['Nf', 'nx', 'nz', 'delta']] = \
                mesh['Nf'], mesh['nx'], mesh['nz'], mesh['delta']
            if 'subindex' in frame:
                frame.at[name, 'subindex'] = list(_subframe[i].index)
        subframe.concatenate(*_subframe)

    @staticmethod
    def _mesh_single(frame, link=True, **kwargs):
        """Mesh single poloidal field coil."""
        delta = frame.delta
        if 'poly' in frame:
            frame_polygon = frame.poly
            bounds = frame_polygon.bounds
            dx = bounds[2] - bounds[0]
            dz = bounds[3] - bounds[1]
        else:  # assume rectangular coil cross-section
            dx, dz = frame[['dl', 'dt']]  # length, thickness == dx, dz
            bounds = (frame.x-frame.dx/2, frame.z-frame.dz/2,
                      frame.x+frame.dx/2, frame.z+frame.dz/2)
            frame_polygon = shapely.geometry.box(*bounds)

        mesh = {'link': link}  # multi-point constraint (link current)
        if 'part' in frame:
            mesh['part'] = frame['part']
        mesh['section'] = kwargs.get('section', frame['turn'])
        if delta != -1:
            mesh['section'] = 'rectangle'
        if 'turn_fraction' in frame and delta == -1:
            turn_fraction = kwargs.get('turn_fraction', frame['turn_fraction'])
        else:
            turn_fraction = kwargs.get('turn_fraction', 1)
        if delta is None or delta == 0:
            Nf = 1
            delta = np.max([dx, dz])
        elif delta == -1:  # mesh per-turn
            Nf = frame['Nt']
            if 'section' not in mesh:
                mesh['section'] = 'circle'
            if frame['section'] == 'circle':
                delta = (np.pi * ((dx + dz) / 4)**2 / Nf)**0.5
            else:
                delta = (dx * dz / Nf)**0.5
        elif delta < -1:
            Nf = -delta  # set filament number
            if frame['section'] == 'circle':
                delta = (np.pi * (dx / 2)**2 / Nf)**0.5
            else:
                delta = (dx * dz / Nf)**0.5
        elif delta > 0:
            nx = np.max([1, int(np.round(dx / delta))])
            nz = np.max([1, int(np.round(dz / delta))])
            Nf = nx * nz
        section = mesh['section']
        nx = np.max([1, int(np.round(dx / delta))])
        nz = int(np.round(Nf / nx))
        if nx < 1:
            nx = 1
        if nz < 1:
            nz = 1
        dx_, dz_ = dx / nx, dz / nz  # subframe divisions
        if section in ['circle', 'square', 'skin']:
            dx_ = dz_ = np.min([dx_, dz_])  # equal aspect
        dl_ = turn_fraction * dx_
        if section == 'skin':  # update fractional thickness
            dt_ = frame['skin_fraction']
        else:
            dt_ = turn_fraction * dz_
        x_ = np.linspace(*bounds[::2], nx+1)[:-1]
        z_ = np.linspace(*bounds[1::2], nz+1)[:-1]
        shape = polygen(section)  # polygon generator
        sub_polygons = [[] for __ in range(nx*nz)]
        __x = np.zeros(nx*nz)
        __z = np.zeros(nx*nz)
        for i in range(nx):  # radial divisions
            for j in range(nz):  # vertical divisions
                sub_polygons[i*nz + j] = \
                        shape(x_[i]+dx_/2, z_[j]+dz_/2, dl_, dt_)
                __x[i*nz + j] = x_[i]+dx_/2
                __z[i*nz + j] = z_[j]+dz_/2
        xm, zm = np.meshgrid(x_+dx_/2, z_+dz_/2, indexing='ij')
        xm = xm.reshape(-1, order='F')
        zm = zm.reshape(-1, order='F')
        dA = sub_polygons[0].area  # referance area
        if frame.section in ['square', 'rectangle']:
            polygon = sub_polygons
            xm_, zm_ = xm, zm
            dA_ = dA * np.ones(nx*nz)
            cs_ = section
        else:
            tree = shapely.strtree.STRtree(sub_polygons)
            sub_polygons = [p for p in tree.query(frame_polygon)
                            if p.intersects(frame_polygon)]
            # apply buffer to frame polygon (for within boolean)
            frame_polygon_buffer = frame_polygon.buffer(1e-12*delta)
            polygon, xm_, zm_, cs_, dA_ = [], [], [], [], []
            for i, sub_polygon in enumerate(sub_polygons):
                p = frame_polygon.intersection(sub_polygon)
                if not pandas.api.types.is_list_like(p):
                    p = [p]  # single polygon
                for p_ in p:
                    if isinstance(p_, shapely.geometry.polygon.Polygon):
                        polygon.append(p_)
                        if sub_polygon.within(frame_polygon_buffer):  # link
                            xm_.append(xm[i])
                            zm_.append(zm[i])
                            dA_.append(dA)
                            cs_.append(section)
                        else:  # re-calculate
                            xm_.append(p_.centroid.x)
                            zm_.append(p_.centroid.y)
                            dA_.append(p_.area)
                            cs_.append('polygon')
        Nf = len(xm_)  # filament number
        if Nf == 0:  # no points found within polygon (skin)
            xm_, zm_, dl_, dt_ = frame.x, frame.z, frame.dl, frame.dt
            Nf = 1
        # constant current density
        Nt_ = frame['Nt']*np.array(dA_) / np.sum(dA_)
        # subframe bundle
        mesh.update({'x': xm_, 'z': zm_, 'nx': nx, 'nz': nz,
                     'dl': dl_, 'dt': dt_, 'Nt': Nt_, 'Nf': Nf,
                     'poly': polygon, 'section': cs_,
                     'delta': delta})

        # subframe moment arms
        # xo, zo = frame.loc[['x', 'z']]
        # mesh['rx'] = xm_ - xo
        # mesh['rz'] = zm_ - zo

        # propagate current update flags to subframe
        for label in ['part', 'active', 'optimize', 'plasma', 'acloss']:
            if label in frame:
                mesh[label] = frame[label]
        mesh['Ic'] = frame['Ic']
        mesh['turn_fraction'] = turn_fraction
        return mesh
