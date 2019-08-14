from nova.coil_object import CoilObject
import pandas as pd
import numpy as np
from amigo.geom import gmd, amd
import functools
import matplotlib
import operator
from amigo.IO import human_format
from sklearn.cluster import DBSCAN
from matplotlib.collections import PatchCollection
from amigo.pyplot import plt
from nova.inductance.geometric_mean_radius import coil_gmr
import shapely.geometry
from descartes import PolygonPatch


class CoilSet(CoilObject):
    '''
    CoilSet:
        - implements methods to manage input and
            output of data to/from the CoilObject class
    '''
    def __init__(self, *args, **kwargs):
        CoilObject.__init__(self, **kwargs)  # inherent from coil object
        if len(args) == 1:
            self.add_coilset(args[0])  # overwrite coilset
        else:
            self.append_coilset(*args)  # append list of coilset instances

    @property
    def coilset(self):
        return CoilObject(self.coil, self.subcoil, inductance=self.inductance,
                          force=self.force, subforce=self.subforce,
                          grid=self.grid)

    @coilset.setter
    def coilset(self, coilset):
        for attribute in self._attributes:
            setattr(self, attribute, getattr(coilset, attribute))

    def add_coilset(self, coilset=None):
        if coilset is not None:
            self.coilset = coilset  # overwrite coilset object

    def append_coilset(self, *args):
        for coilset in args:
            for attribute in ['coil', 'subcoil']:
                coil = getattr(coilset, attribute)
                if not coil.empty:
                    getattr(self, attribute).add_coil(coil)

    def subset(self, coil_index, invert=False):
        if not pd.api.types.is_list_like(coil_index):
            coil_index = [coil_index]
        if invert:
            coil_index = self.coil.loc[~self.coil.index.isin(coil_index)].index
        subindex = []
        for index in coil_index:
            subindex.extend(self.coil.loc[index, 'subindex'])
        return CoilObject(coil=self.coil.loc[coil_index],
                          subcoil=self.subcoil.loc[subindex])

    @property
    def Ic(self):
        '''
        Returns:
            self.coil.Ic (pd.Series): coil instance line current [A]
        '''
        return self.coil.Ic

    @Ic.setter
    def Ic(self, value):
        self.set_current(value, 'Ic')

    @property
    def It(self):
        '''
        Returns:
            self.coil.It (pd.Series): coil instance turn current [A.turns]
        '''
        return self.coil.It

    @It.setter
    def It(self, value):
        self.set_current(value, 'It')

    @property
    def Ip(self):
        # return total plasma current
        return self.coil.loc['Plasma', 'Ic']

    @Ip.setter
    def Ip(self, Ip):
        self.coil.Ic = pd.Series({'Plasma': Ip})

    def set_current(self, value, current_column):
        '''
        update current in coil and subcoil instances (Ic and It)
        index built as union of value.index and coil.index
        mpc constraints applied
        Args:
            value (pd.Series, dict or itterable): current update
            current_column (str):
                'Ic' == line current [A]
                'It' == turn current [A.turns]
        '''
        if isinstance(value, dict):  # dict to pd.Series
            value = pd.Series(value)
        elif not hasattr(value, 'index'):  # itterable to pd.Series
            value = pd.Series(value, index=self.Ic.index)
        if current_column not in ['Ic', 'It']:
            raise IndexError(f'current column: {current_column} '
                             'not in [Ic, It]')
        index = [n for n in value.index if n in self.coil.index]  # union index
        setattr(self.coil, current_column, value.loc[index])  # coil
        coil_current = getattr(self.coil, current_column)
        subcoil_current = pd.Series(index=self.subcoil.index)
        for name, subindex in zip(self.coil.index, self.coil.subindex):
            subcoil_current.loc[subindex] = coil_current[name]
        setattr(self.subcoil, current_column, subcoil_current)  # subcoil

    @staticmethod
    def categorize_coilset(frame, xo=None, rename=True):
        '''
        categorize coils in frame as CS or PF
        categorization split based on coils minimum radius
        CS coils ordered by x then z
        PF coils ordered by theta taken about coilset centroid
        '''
        if xo is None:
            xo = (frame['x'].mean(), frame['z'].mean())
        # sort CS coils ['x', 'z']
        CSo = frame['x'].idxmin()
        xCS = frame.loc[CSo, 'x'] + frame.loc[CSo, 'dx']
        CS = frame.loc[frame['x'] <= xCS, :]
        CS = CS.sort_values(['x', 'z'])
        CS = CS.assign(part='CS')
        # sort PF coils ['theta']
        PF = frame.loc[frame['x'] > xCS, :]
        PF = PF.assign(theta=np.arctan2(PF['z'], PF['x']))
        PF = PF.sort_values('theta')
        PF.drop(columns='theta', inplace=True)
        PF = PF.assign(part='PF')
        if rename:
            CS.index = [f'CS{i}' for i in range(CS.nC)]
            PF.index = [f'PF{i}' for i in range(PF.nC)]
        frame = pd.concat([PF, CS])
        return frame

    def add_coil(self, *args, iloc=None, subcoil=True, **kwargs):
        # add primary coil
        index = self.coil.add_coil(*args, iloc=iloc, **kwargs)
        if subcoil:
            self.add_subcoil(index=index)

    def add_subcoil(self, index=None, remesh=False):
        if index is None:  # re-mesh all coils
            remesh = True
            index = self.coil.index
        if remesh:
            self.subcoil.drop(self.subcoil.index, inplace=True)
        frame = [[] for __ in range(len(index))]
        for i, name in enumerate(index):
            frame[i] = self._mesh_coil(name, dCoil=self.coil.at[name, 'dCoil'])
            if 'part' in self.coil.columns:  # propagate part label
                frame[i].loc[:, 'part'] = self.coil.at[name, 'part']
        self.subcoil.concatenate(*frame)

    def _mesh_coil(self, name, dCoil=None, part=None):
        '''
        mesh coil instance
        '''
        if dCoil is None:
            dCoil = self.coil.at[name, 'dCoil']
        else:
            self.coil.at[name, 'dCoil'] = dCoil  # back-propagate dCoil setting
        x, z, dx, dz = self.coil.loc[name, ['x', 'z', 'dx', 'dz']]
        kwargs = {}
        if 'turn_section' in self.coil.columns:
            kwargs['cross_section'] = self.coil.at[name, 'turn_section']
        if 'turn_fraction' in self.coil.columns and dCoil == -1:
            turn_fraction = self.coil.at[name, 'turn_fraction']
        else:
            turn_fraction = 1
        if dCoil is None or dCoil == 0:
            dCoil = np.mean([dx, dz])
        if dCoil == -1:  # mesh per-turn (for detailed inductance calculations)
            kwargs['cross_section'] = 'circle'
            Nt = self.coil.at[name, 'Nt']
            dCoil = (dx * dz / Nt)**0.5
        nx = int(np.round(dx / dCoil))
        nz = int(np.round(dz / dCoil))
        if nx < 1:
            nx = 1
        if nz < 1:
            nz = 1
        dx_, dz_ = dx / nx, dz / nz  # subcoil dimensions
        x_ = x + np.linspace(dx_ / 2, dx - dx_ / 2, nx) - dx / 2
        x_ = coil_gmr(x_, dx_)  # displace coil centroid to match section gmr
        z_ = z + np.linspace(dz_ / 2, dz - dz_ / 2, nz) - dz / 2
        xm_, zm_ = np.meshgrid(x_, z_, indexing='ij')
        xm_ = np.reshape(xm_, (-1, 1))[:, 0]
        zm_ = np.reshape(zm_, (-1, 1))[:, 0]

        # place mesh fillaments within polygon exterior
        points = shapely.geometry.MultiPoint(points=list(zip(xm_, zm_)))
        polygon = self.coil.at[name, 'polygon']
        multi_point = np.asarray(polygon.intersection(points))
        if np.size(multi_point) == 2:
            multi_point = [multi_point]
        xm_ = [point[0] for point in multi_point]
        zm_ = [point[1] for point in multi_point]
        Nf = len(xm_)  # filament number
        self.coil.at[name, 'Nf'] = Nf  # back-propagate fillament number
        #if 'It' in self.coil.columns:  # update subcoil turn-current
        #    kwargs['It'] = self.coil.at[name, 'It'] / Nf
        if 'part' in self.coil.columns:
            kwargs['part'] = self.coil.at[name, 'part']
        mesh = {'x': xm_, 'z': zm_,
                'dx': turn_fraction*dx_, 'dz': turn_fraction*dz_}
        args = []
        for var in mesh:
            if var in self.subcoil._required_columns:
                args.append(mesh[var])
            elif var in self.subcoil._additional_columns:
                kwargs[var] = mesh[var]
        #print('sub args', *args, 'xm_', xm_)
        frame = self.subcoil.get_frame(*args, name=name, coil=name, **kwargs)
        self.coil.at[name, 'subindex'] = list(frame.index)
        frame.loc[:, 'Nt'] = self.coil.at[name, 'Nt'] / Nf
        return frame

    def get_iloc(self, index):
        iloc = [None, None]
        for name in index:
            if name in self.coil.index:
                iloc[0] = self.coil.index.get_loc(index[0])
                subindex = self.coil.subindex[index[0]][0]
                iloc[1] = self.subcoil.index.get_loc(subindex)
                break
        return iloc

    def drop_coil(self, index=None):
        if index is None:  # drop all coils
            index = self.coil.index
        if not pd.api.types.is_list_like(index):
            index = [index]
        iloc = self.get_iloc(index)
        for name in index:
            if name in self.coil.index:
                self.subcoil.drop_coil(self.coil.loc[name, 'subindex'])
                self.coil.drop_coil(name)
                for M in self.inductance:
                    self.inductance[M].drop(index=name, columns=name,
                                            inplace=True, errors='ignore')
                self.grid['Psi'].drop(columns=name, inplace=True,
                                      errors='ignore')
        return iloc

    def add_plasma(self, *args, **kwargs):
        label = kwargs.pop('label', 'Pl')  # filament prefix
        name = kwargs.pop('name', 'Pl_0')
        part = kwargs.pop('part', 'plasma')
        coil = kwargs.pop('coil', 'Plasma')
        cross_section = kwargs.pop('cross_section', 'ellipse')
        turn_section = kwargs.pop('turn_section', 'square')
        iloc = [None, None]
        if 'Plasma' in self.coil.index:
            iloc = self.drop_coil('Plasma')
        nlist = sum([1 for arg in args if pd.api.types.is_list_like(arg)])
        if nlist == 0:   # add single plasma coil - mesh filaments
            dCoil = kwargs.pop('dCoil', self.dPlasma)
            self.add_coil(*args, part=part, coil=coil, name='Plasma',
                          dCoil=dCoil, cross_section=cross_section,
                          turn_section=turn_section, iloc=iloc[1], **kwargs)
        else:  # add single / multiple filaments, fit coil
            # add plasma filaments to subcoil
            subindex = self.subcoil.add_coil(
                    *args, label=label, part=part, coil=coil, name=name,
                    turn_section=turn_section, iloc=iloc[1], **kwargs)
            Ip = self.subcoil.It[subindex]  # filament currents
            Ip_net = Ip.sum()  # net plasma current
            if not np.isclose(Ip.sum(), 0):
                Nt = Ip / Ip_net  # filament turn number
            else:
                Nt = np.ones(Ip.size)
            self.subcoil.loc[subindex, 'Nt'] = Nt
            xpl = self.subcoil.x[subindex]  # filament x-location
            zpl = self.subcoil.z[subindex]  # filament z-location
            dx = dz = np.sqrt(np.sum(self.subcoil.dx[subindex] *
                                     self.subcoil.dz[subindex]))
            # add plasma to coil (x_gmd, z_amd)
            Nf = Ip.size
            self.coil.add_coil(gmd(xpl, Nt), amd(zpl, Nt),
                               dz, dx, Nf=Nf, dCoil=None,
                               cross_section=cross_section,
                               name='Plasma', part=part, turn_fraction=1,
                               material='plasma', iloc=iloc[0])
            self.coil.at['Plasma', 'subindex'] = list(subindex)
            # if Nf > 1:
            #     self.inductance('Plasma', update=True)  # re-size plasma coil
            self.Ic = pd.Series({'Plasma': Ip_net})  # update net current

    def add_mpc(self, name, factor=1):
        '''
        define multi-point constraint linking a set of coils
        name: list of coil names (present in self.coil.index)
        factor: inter-coil coupling factor
        '''
        if not pd.api.types.is_list_like(name):
            raise IndexError(f'name: {name} must be list like')
        elif len(name) == 1:
            raise IndexError(f'len({name}) must be > 1')
        if not pd.api.types.is_list_like(factor):
            factor = factor * np.ones(len(name)-1)
        elif len(factor) != len(name)-1:
            raise IndexError(f'len(factor={factor}) must == 1 '
                             f'or == len(name={name})-1')
        for n, f in zip(name[1:], factor):
            self.coil.at[n, 'mpc'] = (name[0], f)

    def cluster(self, n, eps=0.2):
        '''
        cluster coils using DBSCAN algorithm
        '''
        dbscan = DBSCAN(eps=eps, min_samples=1)
        cluster_index = dbscan.fit_predict(self.coil.loc[:, ['x', 'z']])
        self.coil.loc[:, 'cluster_index'] = cluster_index
        merge_index = []
        for part in self.coil.part.unique():
            coil = self.subset(self.coil.index[self.coil.part == part]).coil
            for cluster in coil.cluster_index.unique():
                index = coil.index[coil.cluster_index == cluster]
                if index.size > 1:
                    for i in range(index.size // n + 1):
                        if i*n != len(index):
                            merge_index.append(index[i*n:(i+1)*n])
        self.coil.drop(columns='cluster_index', inplace=True)
        for index in merge_index:
            self.merge(index)

    def merge(self, coil_index, name=None):
        subframe = self.subset(coil_index)
        x = gmd(subframe.coil.x, subframe.coil.Nt)
        z = amd(subframe.coil.z, subframe.coil.Nt)
        dr = np.sqrt(np.sum(subframe.coil.dx * subframe.coil.dz)) / 2
        Ic = subframe.coil.It.sum() / np.sum(abs(subframe.coil.Nt))
        if name is None:
            name = f'{coil_index[0]}-{coil_index[-1]}'
        referance_coil = subframe.coil.loc[coil_index[0], :]
        kwargs = {'name': name}
        for key in subframe.coil.columns:
            if key in ['cross_section', 'part', 'material', 'turn_fraction']:
                # take referance
                kwargs[key] = referance_coil[key]
            elif key in ['Nf', 'Nt', 'm', 'R']:
                kwargs[key] = subframe.coil.loc[:, key].sum()
            elif key == 'polygon':
                polys = [p for p in subframe.coil.loc[:, 'polygon'].values]
                if not pd.isnull(polys).any():
                    polygon = shapely.geometry.MultiPolygon(polys)
        # extract current coil / subcoil locations
        coil_iloc = self.coil.index.get_loc(coil_index[0])
        subcoil_iloc = self.subcoil.index.get_loc(
                self.coil.subindex[coil_index[0]][0])
        # remove seperate coils
        self.drop_coil(coil_index)

        # add merged coil
        self.add_coil(x, z, 2*dr, 2*dr, subcoil=False,
                      iloc=coil_iloc, **kwargs)
        # on-demand patch of top level (coil)
        if pd.isnull(subframe.coil.loc[:, 'patch']).any():
            CoilSet.patch_coil(subframe.coil)  # patch on-demand
        self.coil.at[name, 'patch'] = list(subframe.coil.patch)
        # insert multi-polygon
        self.coil.at[name, 'polygon'] = polygon
        # add subcoils
        subindex = self.subcoil.add_coil(subframe.subcoil, iloc=subcoil_iloc)
        self.coil.at[name, 'subindex'] = list(subindex)
        # update current
        self.Ic = {name: Ic}

    @staticmethod
    def patch_coil(frame, overwrite=False, **kwargs):
        # call on-demand
        part_color = {'VS3': 'C0', 'VS3j': 'gray', 'CS': 'C0', 'PF': 'C0',
                      'trs': 'C2', 'vvin': 'C3', 'vvout': 'C4', 'plasma': 'C4'}
        color = kwargs.get('part_color', part_color)
        zorder = kwargs.get('zorder', {'VS3': 1, 'VS3j': 0, 'CS': 3, 'PF': 2})
        alpha = {'plasma': 0.75}
        patch = [[] for __ in range(frame.nC)]
        for i, (x, z, dx, dz, cross_section,
                current_patch, polygon, color_key) in enumerate(
                frame.loc[:, ['x', 'z', 'dx', 'dz', 'cross_section', 'patch',
                              'polygon', 'part']].values):
            if overwrite or np.array(pd.isnull(current_patch)).any():
                patch[i] = [PolygonPatch(polygon)]
            else:
                patch[i] = [current_patch]
            for j in range(len(patch[i])):
                patch[i][j].set_edgecolor('darkgrey')
                patch[i][j].set_linewidth(0.25)
                patch[i][j].set_antialiased(True)
                patch[i][j].set_facecolor(color.get(color_key, 'C9'))
                patch[i][j].zorder = zorder.get(color_key, 0)
                patch[i][j].set_alpha(alpha.get(color_key, 1))
        frame.loc[:, 'patch'] = patch

    def plot_coil(self, coil, alpha=1, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if not coil.empty:
            if pd.isnull(coil.loc[:, 'patch']).any() or len(kwargs) > 0:
                CoilSet.patch_coil(coil, **kwargs)  # patch on-demand
            patch = coil.loc[:, 'patch']
            # form list of lists
            patch = [p if pd.api.types.is_list_like(p) else [p] for p in patch]
            # flatten
            patch = functools.reduce(operator.concat, patch)
            # sort
            patch = np.array(patch)[np.argsort([p.zorder for p in patch])]
            pc = PatchCollection(patch, match_original=True)
            ax.add_collection(pc)

    def plot(self, subcoil=True, plasma=True, label=False, current=None,
             ax=None):
        if ax is None:
            ax = plt.gca()
        if subcoil:
            self.plot_coil(self.subcoil, ax=ax)
        else:
            self.plot_coil(self.coil, ax=ax)
        if 'Plasma' in self.coil.index and plasma and 'Ic' in self.coil:
            self.label_plasma(ax)
        if label or current:
            self.label_coil(ax, label, current)
        ax.axis('equal')
        ax.axis('off')

    def label_plasma(self, ax, fs=None):
        if fs is None:
            fs = matplotlib.rcParams['legend.fontsize']
        x = self.coil.x['Plasma']
        z = self.coil.z['Plasma']
        ax.text(x, z, f'{1e-6*self.Ip:1.1f}MA', fontsize=fs,
                ha='center', va='center', color=0.9 * np.ones(3),
                zorder=10)

    def label_coil(self, ax, label, current, coil=None, fs=None):
        if fs is None:
            fs = matplotlib.rcParams['legend.fontsize']
        if coil is None:
            coil = self.coil
        parts = np.unique(coil.part)
        parts = [p for p in parts if p not in ['plasma', 'vvin',
                                               'vvout', 'trs']]
        ylim = np.diff(ax.get_ylim())[0]
        for name, part in zip(coil.index, coil.part):
            x, z = coil.at[name, 'x'], coil.at[name, 'z']
            dx, dz = coil.at[name, 'dx'], coil.at[name, 'dz']
            if coil.part[name] == 'CS':
                drs = -2.0 / 3 * dx
                ha = 'right'
            else:
                drs = 2.0 / 3 * dx
                ha = 'left'
            if part in parts and (label and current):
                zshift = max([dz / 10, ylim / 5])
            else:
                zshift = 0
            if part in parts and label:
                ax.text(x + drs, z + zshift, name, fontsize=fs,
                        ha=ha, va='center', color=0.2 * np.ones(3))
            if part in parts and current:
                if current == 'Ic':  # line current, amps
                    unit = 'A'
                    Ilabel = coil.at[name, 'Ic']
                elif current == 'It':  # turn current, amp turns
                    unit = 'At'
                    Ilabel = coil.at[name, 'It']
                txt = f'{human_format(Ilabel, precision=1)}{unit}'
                ax.text(x + drs, z - zshift, txt,
                        fontsize=fs, ha=ha, va='center',
                        color=0.2 * np.ones(3))


if __name__ == '__main__':

    cs = CoilSet(dCoil=0.25)
    cs.update_metadata('coil', additional_columns=['R'])

    cs.add_coil(6, -3, 1.5, 1.5, name='PF6', part='PF', Nt=600, It=5e5)
    cs.add_coil(7, -0.5, 2.5, 2.5, name='PF8', part='PF', Nt=600, Ic=2e3,
                cross_section='circle')

    cs.add_coil([2, 2, 3, 3.5], [1, 0, -1, -3], 0.3, 0.3,
                name='PF', part='PF', delim='', Nt=300)
    cs.add_coil(3, 2, 0.5, 0.8, name='PF4', part='VS3', turn_fraction=0.75,
                Nt=15, dCoil=-1)
    cs.add_coil(5.6, 3.5, 0.2, 0.2, name='PF7', part='vvin', dCoil=0.01, Ic=1e6)

    # cs.add_plasma(6, [1.5, 2, 2.5], 1.75, 0.4, It=-15e6/3)
    cs.add_plasma(7, 3, 1.5, 0.5, It=-15e6/3)

    cs.plot(label=True)



