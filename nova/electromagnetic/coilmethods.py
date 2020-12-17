"""
Coil methods mixin.

Methods inserted into CoilSet class.

"""

import pandas as pd
import numpy as np
import shapely.geometry
import shapely.strtree
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN

from nova.electromagnetic.coilframe import CoilFrame
from nova.utilities import geom
from nova.utilities.geom import amd


class CoilMethods:
    """Coil methods mixin."""

    @property
    def nC(self):
        """
        Return total coil number, read-only.

        Returns
        -------
        nC : int
            The total number of coils (including plasma).
            The number of rows in coil CoilFrame.

        """
        return self.coil.nC

    @property
    def _nC(self):
        """
        Return mpc coil number, read-only.

        Returns
        -------
        _nC : int
            The number of coils without multi-point constraints.

        """
        return self.coil._nC

    @property
    def nI(self):
        """
        Return number of indexed coils.

        Size of current vector required to set Ic or It parameters based on
        setting of the current_update flag.

        Returns
        -------
        nI : int
            Required size of current vector used in Ic or It updates.

        """
        return self.coil._nI

    @property
    def Ic(self):
        """
        Manage coil line current [A].

        Parameters
        ----------
        value : float or array-like, shape(nI,)
            Set coil line current for indexed coils.

        Returns
        -------
        Ic : array-like, shape(nC,)
            Coil line current [A] for all coils.

        """
        return self.coil.Ic

    @Ic.setter
    def Ic(self, value):
        self._set_current(value, 'Ic')

    @property
    def It(self):
        """
        Manage coil turn current [A.turns].

        Parameters
        ----------
        value : float or array-like, shape(nI,)
            Set coil turn current for indexed coils.

        Returns
        -------
        It : array-like, shape(nC,)
            Coil turn current [A.turns] for all coils.

        """
        return self.coil.It

    @It.setter
    def It(self, value):
        self._set_current(value, 'It')

    @property
    def current_index(self):
        """
        Display power, optimize, plasma, and current_update status.

        Returns
        -------
        current_index : DataFrame
            Power, optimize, plasma status.
            Current_update flag and indexed coils (last column).

        """
        return self.coil.current_index

    @property
    def current_update(self):
        """
        Manage current index via current update flag.

        Update current_index via current flag for coil current update.

        Parameters
        ----------
        update_flag : str
            Current update flag.

            - 'full': update full current vector
            - 'active': update active coils (power & ~plasma)
            - 'passive': update passive coils (~power & ~plasma)
            - 'fix': update fix coils (~optimize & ~plasma)
            - 'free': update free coils (optimize & ~plasma)
            - 'plasma': update plasma (plasma)
            - 'coil': update all coils (~plasma)

        Raises
        ------
        IndexError
            update_flag not in
            [full, active, passive, free, fix, plasma, coil].

        Returns
        -------
        update_flag : str
            Current update flag:

        """
        return self.coil.current_update  # current_update

    @current_update.setter
    def current_update(self, flag):
        self._current_update = flag
        for frame in self._coilset_frames:
            coilframe = getattr(self, frame)
            if hasattr(coilframe, 'current_update'):
                coilframe.current_update = flag

    def _set_current(self, value, current_column='Ic'):
        self.relink_mpc()  # relink subcoil mpc as required
        self.coil._set_current(value, current_column)
        subvalue = self.coil._Ic[self.coil._mpc_referance]
        self.subcoil._set_current(subvalue[self.subcoil.current_index],
            'Ic')  # [self.subcoil._current_index]
        self.update_coil_current = True
        self.update_plasma_current = True
        #self.update_field()

    @property
    def power(self):
        """
        Manage status of coil power flag (active vs. passive).

        Assignments to coils under mpc constraints are overwritten by driving
        coils.

        Parameters
        ----------
        value : bool or array-like, shape(nC,)
            Coil and subcoil power status.

        Returns
        -------
        power : array-like, shape(nC,)
            Coil power status.

        """
        return self.coil.power

    @power.setter
    def power(self, value):
        self.coil.power = value
        self.subcoil.power = value

    @property
    def optimize(self):
        """
        Manage status of coil optimize flag (fix vs. free).

        Assignments to coils under mpc constraints are overwritten by driving
        coils.

        Parameters
        ----------
        value : bool or array-like, shape(nC,)
            Coil and subcoil optimize status.

        Returns
        -------
        optimize : array-like, shape(nC,)
            Coil optimize status.

        """
        return self.coil.optimize

    @optimize.setter
    def optimize(self, value):
        self.coil.optimize = value
        self.subcoil.optimize = value

    def add_coil(self, *args, iloc=None, subcoil=True, **kwargs):
        """
        Add coil(s) to coilframe.

        Parameters
        ----------
        *args : coilframe or dataframe or list, shape(len(_required_columns),)
            Input data.
        iloc : int, optional
            Index before which coils are inserted. The default is None.
        subcoil : bool, optional
            Mesh subcoil. The default is True.
        **kwargs : dict
            coilframe_metadata, **additional_columns.

        Returns
        -------
        None.

        """
        kwargs['delim'] = kwargs.get('delim', '')
        index = self.coil.add_coil(*args, iloc=iloc, **kwargs)
        if subcoil:
            self.meshcoil(index=index)

    def drop_coil(self, index=None):
        """
        Remove coil and subcoil from coilframes.

        Parameters
        ----------
        index : int or list or pd.Index, optional
            Index of coils to be removed. The default is None (all coils).

        Returns
        -------
        iloc : [int, int]
            CoilFrame index of first removed [coil, subcoil].

        """
        if index is None:  # drop all coils
            index = self.coil.index
        if not pd.api.types.is_list_like(index):
            index = [index]
        iloc = self._get_iloc(index)
        for name in index:
            if name in self.coil.index:
                self.subcoil.drop_coil(self.coil.loc[name, 'subindex'])
                self.coil.drop_coil(name)
        return iloc

    def _get_iloc(self, index):
        iloc = [None, None]
        for name in index:
            if name in self.coil.index:
                iloc[0] = self.coil.index.get_loc(index[0])
                subindex = self.coil.subindex[index[0]][0]
                iloc[1] = self.subcoil.index.get_loc(subindex)
                break
        return iloc

    def translate(self, index=None, dx=0, dz=0):
        """
        Translate coil in polidal plane.

        Parameters
        ----------
        index : int or array-like or Index, optional
            Coil index. The default is None (all coils).
        dx : float, optional
            x-coordinate translation. The default is 0.
        dz : float, optional
            z-coordinate translation. The default is 0.

        Returns
        -------
        None.

        """
        if index is None:
            index = self.coil.index
        elif not pd.api.types.is_list_like(index):
            index = [index]
        self.coil.translate(index, dx, dz)
        for name in index:
            self.subcoil.translate(self.coil.loc[name, 'subindex'], dx, dz)

    @property
    def dCoil(self):
        """
        Manage subcoil filament dimension.

        Parameters
        ----------
        dCoil : float
            Subcoil dimension.

        Returns
        -------
        dCoil : float
            Subcoil dimension.

        """
        self._check_default('dCoil')
        return self._dCoil

    @dCoil.setter
    def dCoil(self, dCoil):
        self._dCoil = dCoil
        self._default_attributes['dCoil'] = dCoil
        self.coil._default_attributes['dCoil'] = dCoil

    def meshcoil(self, index=None, mpc=True, **kwargs):
        """
        Mesh coil(s). Store filaments in subcoil. Apply mpc constraints.

        Parameters
        ----------
        index : int or list or pd.Index, optional
            Index of coils to be meshed. The default is None (all coils).
        mpc : bool, optional
            create mpc constraints between subcoils. The default is True.
        **kwargs : dict
            Optional arguments:

                - coil : CoilFrame. The default is self.coil.
                - subcoil : CoilFrame. The default is self.coil.
                - dCoil : float. Subcoil filament dimension.
                - ...

        Returns
        -------
        None.

        """
        coil = kwargs.pop('coil', self.coil)
        coil.generate_polygon()
        #coil.update_polygon()
        subcoil = kwargs.pop('subcoil', self.subcoil)
        if index is None:  # re-mesh all coils
            index = coil.index
        _subcoil = [[] for __ in range(len(index))]
        for i, name in enumerate(index):
            if 'dCoil' in kwargs:
                coil.loc[name, 'dCoil'] = kwargs['dCoil']
            for key in kwargs:
                if key in coil:
                    coil.loc[name, key] = kwargs[key]
            if 'subindex' in coil:  # drop existing subcoils
                if isinstance(coil.loc[name, 'subindex'], list):
                    subcoil.drop(coil.loc[name, 'subindex'], inplace=True)
            mesh = self._mesh_coil(coil.loc[name, :], mpc=mpc,
                                   **kwargs)  # single coil
            subcoil_args, subcoil_kwargs = [], {}
            for var in mesh:
                if var in subcoil._required_columns:
                    subcoil_args.append(mesh[var])
                elif var in subcoil._additional_columns:
                    subcoil_kwargs[var] = mesh[var]
            _subcoil[i] = subcoil.get_coil(
                    *subcoil_args, label=name, coil=name, delim='_',
                    **subcoil_kwargs)
            _subcoil[i].update_polygon()
            # back-propagate fillament attributes to coil
            coil.loc[name, ['Nf', 'nx', 'nz', 'dCoil']] = \
                mesh['Nf'], mesh['nx'], mesh['nz'], mesh['dCoil']
            if 'subindex' in coil:
                coil.at[name, 'subindex'] = list(_subcoil[i].index)
        subcoil.concatenate(*_subcoil)

    @staticmethod
    def _mesh_coil(coil, mpc=True, **kwargs):
        """Mesh single coil."""
        dCoil = coil.dCoil
        if 'polygon' in coil:
            coil_polygon = coil.polygon
            bounds = coil_polygon.bounds
            dx = bounds[2] - bounds[0]
            dz = bounds[3] - bounds[1]
        else:  # assume rectangular coil cross-section
            dx, dz = coil[['dl', 'dt']]  # length, thickness == dx, dz
            bounds = (coil.x-coil.dx/2, coil.z-coil.dz/2,
                      coil.x+coil.dx/2, coil.z+coil.dz/2)
            coil_polygon = shapely.geometry.box(bounds)
        mesh = {'mpc': mpc}  # multi-point constraint (link current)
        if 'part' in coil:
            mesh['part'] = coil['part']
        mesh['cross_section'] = kwargs.get('cross_section',
                                           coil['turn_section'])
        if dCoil != -1:
            mesh['cross_section'] = 'rectangle'
        if 'turn_fraction' in coil and dCoil == -1:
            turn_fraction = kwargs.get('turn_fraction', coil['turn_fraction'])
        else:
            turn_fraction = kwargs.get('turn_fraction', 1)
        if dCoil is None or dCoil == 0:
            dCoil = np.max([dx, dz])
        elif dCoil == -1:  # mesh per-turn
            if 'cross_section' not in mesh:
                mesh['cross_section'] = 'circle'
            Nt = coil['Nt']
            if coil['cross_section'] == 'circle':
                dCoil = (np.pi * ((dx + dz) / 4)**2 / Nt)**0.5
            else:
                dCoil = (dx * dz / Nt)**0.5
        elif dCoil < -1:
            Nf = -dCoil  # set filament number
            if coil['cross_section'] == 'circle':
                dCoil = (np.pi * (dx / 2)**2 / Nf)**0.5
            else:
                dCoil = (dx * dz / Nf)**0.5
        cross_section = mesh['cross_section']
        nx = int(np.round(dx / dCoil))
        nz = int(np.round(dz / dCoil))
        if nx < 1:
            nx = 1
        if nz < 1:
            nz = 1
        dx_, dz_ = dx / nx, dz / nz  # subcoil divisions
        if cross_section in ['circle', 'square', 'skin']:
            dx_ = dz_ = np.min([dx_, dz_])  # equal aspect
        dl_ = turn_fraction * dx_
        if cross_section == 'skin':  # update fractional thickness
            dt_ = coil['skin_fraction']
        else:
            dt_ = turn_fraction * dz_
        x_ = np.linspace(*bounds[::2], nx+1)[:-1]
        z_ = np.linspace(*bounds[1::2], nz+1)[:-1]
        polygen = CoilFrame._get_polygen(cross_section)  # polygon generator
        sub_polygons = [[] for __ in range(nx*nz)]
        __x = np.zeros(nx*nz)
        __z = np.zeros(nx*nz)
        for i in range(nx):  # radial divisions
            for j in range(nz):  # vertical divisions
                sub_polygons[i*nz + j] = \
                        polygen(x_[i]+dx_/2, z_[j]+dz_/2, dl_, dt_)
                __x[i*nz + j] = x_[i]+dx_/2
                __z[i*nz + j] = z_[j]+dz_/2
        xm, zm = np.meshgrid(x_+dx_/2, z_+dz_/2, indexing='ij')
        xm = xm.reshape(-1, order='F')
        zm = zm.reshape(-1, order='F')
        dA = sub_polygons[0].area  # referance area
        if coil.cross_section in ['square', 'rectangle']:
            polygon = sub_polygons
            xm_, zm_ = xm, zm
            dA_ = dA * np.ones(nx*nz)
            cs_ = cross_section
        else:
            tree = shapely.strtree.STRtree(sub_polygons)
            sub_polygons = [p for p in tree.query(coil_polygon)
                            if p.intersects(coil_polygon)]
            # apply buffer to coil polygon (for within boolean)
            coil_polygon_buffer = coil_polygon.buffer(1e-12*dCoil)
            polygon, xm_, zm_, cs_, dA_ = [], [], [], [], []
            for i, sub_polygon in enumerate(sub_polygons):
                p = coil_polygon.intersection(sub_polygon)
                if not pd.api.types.is_list_like(p):
                    p = [p]  # single polygon
                for p_ in p:
                    if isinstance(p_, shapely.geometry.polygon.Polygon):
                        polygon.append(p_)
                        if sub_polygon.within(coil_polygon_buffer):  # link
                            xm_.append(xm[i])
                            zm_.append(zm[i])
                            dA_.append(dA)
                            cs_.append(cross_section)
                        else:  # re-calculate
                            xm_.append(p_.centroid.x)
                            zm_.append(p_.centroid.y)
                            dA_.append(p_.area)
                            cs_.append('polygon')
        Nf = len(xm_)  # filament number
        if Nf == 0:  # no points found within polygon (skin)
            xm_, zm_, dl_, dt_ = coil.x, coil.z, coil.dl, coil.dt
            Nf = 1
        # constant current density
        Nt_ = coil['Nt']*np.array(dA_) / np.sum(dA_)
        # subcoil bundle
        mesh.update({'x': xm_, 'z': zm_, 'nx': nx, 'nz': nz,
                     'dl': dl_, 'dt': dt_, 'Nt': Nt_, 'Nf': Nf,
                     'polygon': polygon, 'cross_section': cs_,
                     'dCoil': dCoil})

        # subcoil moment arms
        # xo, zo = coil.loc[['x', 'z']]
        # mesh['rx'] = xm_ - xo
        # mesh['rz'] = zm_ - zo

        # propagate current update flags to subcoil
        for label in ['part', 'power', 'optimize', 'plasma']:
            if label in coil:
                mesh[label] = coil[label]
        mesh['Ic'] = coil['Ic']
        mesh['turn_fraction'] = turn_fraction
        return mesh

    def add_mpc(self, index, factor=1):
        """
        Add multi-point constaint linking coils listed in index.

        Parameters
        ----------
        index : list
            list of coil indices to be linked.
            First coil acts as primary.
        factor : float or array-like, optional
            Linking factor. Can be less than zero. The default is 1.

        Returns
        -------
        None.

        """
        self.coil.add_mpc(index, factor)
        self.relink_mpc()

    def relink_mpc(self):
        """Rebuild multi-point constraints."""
        if self.coil._relink_mpc:
            # force dataframe update
            self.coil._update_dataframe['Ic'] = True
            self.coil._update_dataframe['It'] = True
            for attribute in self.coil._coilcurrent_attributes:
                setattr(self.subcoil, attribute, getattr(self.coil, attribute))
            self.subcoil.current_update = self.coil.current_update
            self.coil._relink_mpc = False

    @property
    def dShell(self):
        """
        Manage shell coil dimension.

        Line elements meshed as n seperate coils based on dShell parameter.
        Subcoil dimension then set via dCoil.

        - dCoil = 0: single subcoil per coil
        - dCoil = -1: subcoil dimesion equal to shell thickness

        Parameters
        ----------
        dShell : float
            Shell thickness parameter.

        Returns
        -------
        dShell : float
            Shell thickness parameter.

        """
        self._check_default('dShell')
        return self._dShell

    @dShell.setter
    def dShell(self, dShell):
        self._dShell = dShell
        self._default_attributes['dShell'] = dShell
        self.coil._default_attributes['dShell'] = dShell

    def add_shell(self, x, z, dt, **kwargs):
        """
        Add shell elements to coilset.

        Lines described by x, z coordinates meshed into n coils based on
        dShell. Each coil is then submeshed based on dCoil.

        Parameters
        ----------
        x : array-like, shape(n,)
            x-coordinates of poloidal line to be meshed.
        z : array-like, shape(n,)
            z-coordinates of poloidal line to be meshed.
        dt : float
            Shell thickness.
        **kwargs : dict
            Optional arguements.

        Returns
        -------
        None.

        """
        label = kwargs.pop('label', kwargs.get('part', 'Shl'))
        dShell = kwargs.pop('dShell', self._default_attributes['dShell'])
        dCoil = kwargs.pop('dCoil', self._default_attributes['dCoil'])
        power = kwargs.pop('power', False)
        delim = kwargs.pop('delim', '')
        rho = kwargs.pop('rho', 0)
        x, z, dl, dt, dA, rho_bar, polygon, sub_segment, sub_rho, sub_dt = \
            self._shlspace((x, z), dt, rho, dShell)
        index = self.coil.add_coil(x, z, dl, dt, dA=dA, polygon=polygon,
                                   cross_section='shell', turn_fraction=1,
                                   turn_section='shell', dCoil=dCoil,
                                   power=power, label=label,
                                   delim=delim, Nt=dA, rho=rho_bar,
                                   **kwargs)
        self.coil.update_polygon()
        subindex = [[] for __ in range(len(index))]
        kwargs.pop('name', None)
        for i, coil in enumerate(index):
            _x, _z, _dl, _dt, _dA, _rho_bar, _polygon = \
                self._shlspace(sub_segment[i], sub_dt[i], sub_rho[i],
                               dCoil)[:-3]
            subindex[i] = self.subcoil.add_coil(
                _x, _z, _dl, _dt,
                polygon=_polygon, coil=coil, cross_section='shell',
                mpc=True, power=power, name=index[i], Nt=_dA,
                rho=_rho_bar, **kwargs)
            self.coil.at[index[i], 'subindex'] = subindex[i]
        self.subcoil.update_polygon()

    @staticmethod
    def _shlspace(segment, dt, rho, dS):
        if not pd.api.types.is_list_like(dt):  # variable thickness segments
            dt *= np.ones(np.shape(segment)[1])
        if not pd.api.types.is_list_like(rho):  # variable resistivity segments
            rho *= np.ones(np.shape(segment)[1])
        dL = geom.length(*segment, norm=False)  # cumulative segment length
        L = dL[-1]  # total segment length
        if dS == 0:
            dS = L
        dt_min = dt.min()
        if dS < dt_min:
            dS = dt_min
        nS = int(L / dS)  # segment number
        if nS < 1:  # ensure > 0
            nS = 1
        dS = L / nS  # model resolved sub-segment length
        nSS = int(L / dt_min)
        if nSS < 2:
            nSS = 2
        _x, _z = geom.xzfun(*segment)  # xz interpolators
        _dt = interp1d(dL/L, dt)
        _rho = interp1d(dL/L, rho)
        Lend = np.linspace(0, 1, nS+1)  # endpoints
        polygon = [[] for __ in range(nS)]
        x, z = np.zeros(nS), np.zeros(nS)
        rho_bar, dt_bar = np.zeros(nS), np.zeros(nS)
        dl, dt, dA = np.zeros(nS), np.zeros(nS), np.zeros(nS)
        sub_segment = np.zeros((nS, 2, nSS))
        sub_rho, sub_dt = np.zeros((nS, nSS)), np.zeros((nS, nSS))
        for i in range(nS):
            Lsegment = np.linspace(Lend[i], Lend[i+1], nSS)
            dLseg = Lend[i+1] - Lend[i]
            sub_segment[i, :, :] = np.array([_x(Lsegment), _z(Lsegment)])
            sub_rho[i, :] = _rho(Lsegment)
            sub_dt[i, :] = _dt(Lsegment)
            # average attributes
            dt_bar[i] = 1 / dLseg * np.trapz(sub_dt[i, :], Lsegment)
            if np.min(sub_rho[i, :]) > 0:
                rho_bar[i] = dt_bar[i] / (1 / dLseg * np.trapz(
                        sub_dt[i, :] / sub_rho[i, :], Lsegment))
            else:
                rho_bar[i] = _rho((Lend[i] + Lend[i+1]) / 2)  # take mid-point
            line = [tuple(ss) for ss in sub_segment[i].T]
            polygon[i] = shapely.geometry.LineString(line).buffer(
                    dt_bar[i]/2, cap_style=2, join_style=2)
            x[i] = polygon[i].centroid.x
            z[i] = polygon[i].centroid.y
            dl[i] = dS  # sub-segment length
            dt[i] = dt_bar[i]  # sub-segment thickness
            dA[i] = polygon[i].area
        return x, z, dl, dt, dA, rho_bar, polygon, sub_segment, sub_rho, sub_dt

    def cluster(self, n, eps=0.2):
        """
        Cluster coils using DBSCAN algorithm.

        Parameters
        ----------
        n : int
            Target cluster size.
        eps : float, optional
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. The default is 0.2.

        Returns
        -------
        None.

        """
        dbscan = DBSCAN(eps=eps, min_samples=1)
        cluster_index = dbscan.fit_predict(self.coil.loc[:, ['x', 'z']])
        cluster_index = pd.Series(cluster_index, index=self.coil.index)
        merge_index = []
        for part in self.coil.part.unique():
            coil = self.subset(self.coil.index[self.coil.part == part]).coil
            cluster_subset = cluster_index.loc[self.coil.part == part]
            for cluster in cluster_subset.unique():
                index = coil.index[cluster_subset == cluster]
                if index.size > 1:
                    for i in range(index.size // n + 1):
                        if i*n != len(index):
                            merge_index.append(index[i*n:(i+1)*n])
        for index in merge_index:
            self.merge(index)

    def merge(self, coil_index, name=None):
        """
        Merge coils listed in coil_index.

        Parameters
        ----------
        coil_index : array-like or Index
            Index of coils to be merged.
        name : str, optional
            Name of merged coil.
            The default is f'{coil_index[0]}-{coil_index[-1]}.

        Returns
        -------
        None.

        """
        subset = self.subset(coil_index)
        x = geom.rms(subset.coil.x, subset.coil.Nt)
        z = amd(subset.coil.z, subset.coil.Nt)
        dx = np.max(subset.coil.x + subset.coil.dx/2) -\
            np.min(subset.coil.x - subset.coil.dx/2)
        dz = np.max(subset.coil.z + subset.coil.dz/2) -\
            np.min(subset.coil.z - subset.coil.dz/2)
        Ic = subset.coil.It.sum() / np.sum(abs(subset.coil.Nt))
        if name is None:
            name = f'{coil_index[0]}-{coil_index[-1]}'
        referance_coil = subset.coil.loc[coil_index[0], :]
        kwargs = {'name': name}
        for key in subset.coil.columns:
            if key in ['Nf', 'Nt', 'm', 'R']:
                kwargs[key] = subset.coil.loc[:, key].sum()
            elif key == 'polygon':
                polys = [p for p in subset.coil.loc[:, 'polygon'].values]
                if not pd.isnull(polys).any():
                    polygon = shapely.geometry.MultiPolygon(polys)
            elif key not in self.coil._required_columns:
                # take referance
                kwargs[key] = referance_coil[key]
        # extract current coil / subcoil locations
        coil_iloc = self.coil.index.get_loc(coil_index[0])
        subcoil_iloc = self.subcoil.index.get_loc(
                self.coil.subindex[coil_index[0]][0])
        # remove seperate coils
        self.drop_coil(coil_index)
        # add merged coil
        self.add_coil(x, z, dx, dz, subcoil=False, iloc=coil_iloc, **kwargs)
        # insert multi-polygon
        self.coil.loc[name, 'polygon'] = polygon
        # on-demand patch of top level (coil)
        if pd.isnull(subset.coil.loc[:, 'patch']).any():
            CoilMethods.patch_coil(subset.coil)  # patch on-demand
        # add subcoils
        subindex = self.subcoil.add_coil(subset.subcoil, iloc=subcoil_iloc)
        self.coil.loc[name, 'subindex'] = list(subindex)
        self.subcoil.loc[subindex, 'coil'] = name
        self.subcoil.add_mpc(subindex.to_list())
        self.Ic = {name: Ic}

    def categorize_coilset(self, rename=False):
        """
        Order / categorize coils in self.coil dataframe as CS or PF.

        Categorization split based on coils minimum radius
        CS coils ordered by x then z
        PF coils ordered by theta taken about coilset centroid

        Parameters
        ----------
        rename : bool, optional
            rename dataframe index. The default is False.

        Returns
        -------
        None.

        """
        # set part labels
        label = {part: 'part' in self.coil.part.values
                 for part in ['CS', 'PF']}
        # select CS coils
        if label['CS']:
            CS = self.coil.loc[self.coil.part == 'CS', :]
        else:
            CSo = self.coil['x'].idxmin()
            xCS = self.coil.loc[CSo, 'x'] + self.coil.loc[CSo, 'dx']
            CS = self.coil.loc[self.coil['x'] <= xCS, :]
        # select PF coils
        if label['PF']:
            PF = self.coil.loc[self.coil.part == 'PF', :]
        elif not label['CS']:
            PF = self.coil.loc[self.coil['x'] > xCS, :]
        else:
            PF = self.coil.loc[self.coil.part != 'CS', :]
        # select other coils
        if label['CS'] and label['PF']:
            index = self.coil.part != 'CS' and self.coil.part != 'PF'
            other = self.coil.loc[index, :]
        else:
            other = pd.DataFrame()
        # sort CS coils ['z']
        CS = CS.sort_values(['z'])
        CS = CS.assign(part='CS')
        # sort PF coils ['theta']
        PF = PF.assign(theta=np.arctan2(PF['z'], PF['x']))
        PF = PF.sort_values('theta')
        PF.drop(columns='theta', inplace=True)
        PF = PF.assign(part='PF')
        if rename:
            CS.index = [f'CS{i}' for i in range(CS.nC)]
            PF.index = [f'PF{i}' for i in range(PF.nC)]
        self.coil = pd.concat([PF, CS, other])

    def rename(self, index):
        self.coil.rename(index=index, inplace=True)  # rename coil
        for name in index:  # link subcoil
            self.subcoil.loc[self.coil.loc[index[name], 'subindex'], 'coil'] = \
                index[name]
        self.subcoil.rebuild_coildata()  # rebuild coildata
