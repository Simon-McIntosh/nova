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

from nova.electromagnetic.frameset import FrameSet
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
        Display active, optimize, plasma, and current_update status.

        Returns
        -------
        current_index : DataFrame
            active, optimize, plasma status.
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
            - 'active': update active coils (active & ~plasma)
            - 'passive': update passive coils (~active & ~plasma)
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
    def active(self):
        """
        Manage status of coil active flag (active vs. passive).

        Assignments to coils under mpc constraints are overwritten by driving
        coils.

        Parameters
        ----------
        value : bool or array-like, shape(nC,)
            Coil and subcoil active status.

        Returns
        -------
        active : array-like, shape(nC,)
            Coil active status.

        """
        return self.coil.active

    @active.setter
    def active(self, value):
        self.coil.active = value
        self.subcoil.active = value

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
        active = kwargs.pop('active', False)
        delim = kwargs.pop('delim', '')
        rho = kwargs.pop('rho', 0)
        x, z, dl, dt, dA, rho_bar, polygon, sub_segment, sub_rho, sub_dt = \
            self._shlspace((x, z), dt, rho, dShell)
        index = self.coil.add_coil(x, z, dl, dt, dA=dA, polygon=polygon,
                                   cross_section='shell', turn_fraction=1,
                                   turn_section='shell', dCoil=dCoil,
                                   active=active, label=label,
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
                mpc=True, active=active, name=index[i], Nt=_dA,
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
