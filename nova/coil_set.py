from nova.coil_frame import CoilFrame
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


class CoilSet:

    '''
    CoilSet:
        instance wrapper for coilset data

    Attributes:
        frame (nova.CoilFrame): coil config
        subframe (nova.CoilFrame): subcoil config
        data (nova.CoilData): coil data
        subdata (nova.CoilData): subcoil data

    '''

    # main class attribures
    _attributes = ['frame', 'subframe']

    # default class parameters
    _parameters = {'dCoil': 0, 'dPlasma': 0.25, 'turn_fraction': 1}

    # frame additional _columns
    _frame = ['dCoil', 'Nf', 'Nt', 'It', 'Ic', 'subindex',
              'cross_section', 'turn_section', 'turn_fraction',
              'patch', 'polygon', 'part', 'control']
    _subframe = ['mpc', 'coil', 'Nt', 'It', 'Ic', 'cross_section',
                 'patch', 'polygon', 'part']

    def __init__(self, *args, **kwargs):
        kwargs = self.set_parameters(**kwargs)  # set instance parameters
        self.set_attributes(**kwargs)  # set attributes from kwargs
        self.initialize_frame()  # initalize frame and subframe
        self.append_coilset(*args)  # append list of coilset instances

    def set_parameters(self, **kwargs):
        for parameter in self._parameters:
            value = kwargs.pop(parameter, self._parameters[parameter])
            setattr(self, parameter, value)
        return kwargs

    def set_attributes(self, **kwargs):
        for attribute in self._attributes:
            setattr(self, attribute, kwargs.pop(attribute, None))

    def initialize_frame(self):
        self.frame = CoilFrame(
                self.frame, additional_columns=self._frame,
                default_attributes={'dCoil': self.dCoil,
                                    'turn_fraction': self.turn_fraction})
        self.subframe = CoilFrame(
                self.subframe, additional_columns=self._subframe,
                default_attributes={})

    def update_metadata(self, frame, **metadata):
        '''
        update coilset metadata ['frame', 'subframe']
        '''
        getattr(self, frame)._update_metadata(**metadata)

    @property
    def coilset(self):
        _attributes = list(self._parameters.keys()) + self._attributes
        kwargs = {attribute: getattr(self, attribute)
                  for attribute in _attributes}
        return CoilSet(**kwargs)

    @coilset.setter
    def coilset(self, coilset):
        for attribute in self._attributes:
            setattr(self, attribute, getattr(coilset, attribute))

    def append_coilset(self, *args):
        for coilset in args:
            if self.frame.empty:  # overwrite
                self.coilset = coilset
            else:  # append
                for attribute in self._attributes:
                    frame = getattr(coilset, attribute)
                    getattr(self, attribute).add_coil(frame)

    def subset(self, index, invert=False):
        if not pd.api.types.is_list_like(index):
            index = [index]
        if invert:
            index = self.frame.loc[~self.frame.index.isin(index)].index
        subindex = []
        for _index in index:
            subindex.extend(self.frame.loc[_index, 'subindex'])
        return CoilSet(frame=self.frame.loc[index],
                       subframe=self.subframe.loc[subindex])

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
        index = self.frame.add_coil(*args, iloc=iloc, **kwargs)
        if subcoil:
            self.add_subcoil(index=index)

    def add_subcoil(self, index=None, remesh=False):
        if index is None:  # re-mesh all coils
            remesh = True
            index = self.frame.index
        if remesh:
            self.subframe.drop(self.subframe.index, inplace=True)
        frame = [[] for __ in range(len(index))]
        for i, name in enumerate(index):
            frame[i] = self._mesh_coil(
                    name, dCoil=self.frame.at[name, 'dCoil'])
            if 'part' in self.frame.columns:  # propagate part label
                frame[i].loc[:, 'part'] = self.frame.at[name, 'part']
            frame[i].add_mpc(frame[i].index.to_list())  # link turns (Ic)
        self.subframe.concatenate(*frame)

    def _mesh_coil(self, name, dCoil=None, part=None):
        '''
        mesh coil instance
        '''
        if dCoil is None:
            dCoil = self.frame.at[name, 'dCoil']
        else:
            self.frame.at[name, 'dCoil'] = dCoil  # update
        x, z, dx, dz = self.frame.loc[name, ['x', 'z', 'dx', 'dz']]
        kwargs = {}
        if 'turn_section' in self.frame.columns:
            kwargs['cross_section'] = self.frame.at[name, 'turn_section']
        if 'turn_fraction' in self.frame.columns and dCoil == -1:
            turn_fraction = self.frame.at[name, 'turn_fraction']
        else:
            turn_fraction = 1
        if dCoil is None or dCoil == 0:
            dCoil = np.mean([dx, dz])
        if dCoil == -1:  # mesh per-turn (for detailed inductance calculations)
            kwargs['cross_section'] = 'circle'
            Nt = self.frame.at[name, 'Nt']
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
        polygon = self.frame.at[name, 'polygon']
        multi_point = np.asarray(polygon.intersection(points))
        if np.size(multi_point) == 2:
            multi_point = [multi_point]
        xm_ = [point[0] for point in multi_point]
        zm_ = [point[1] for point in multi_point]
        Nf = len(xm_)  # filament number
        self.frame.at[name, 'Nf'] = Nf  # back-propagate fillament number
        if 'part' in self.frame.columns:
            kwargs['part'] = self.frame.at[name, 'part']
        mesh = {'x': xm_, 'z': zm_,
                'dx': turn_fraction*dx_, 'dz': turn_fraction*dz_}
        args = []
        for var in mesh:
            if var in self.subframe._required_columns:
                args.append(mesh[var])
            elif var in self.subframe._additional_columns:
                kwargs[var] = mesh[var]
        frame = self.subframe.get_frame(*args, name=name, coil=name, **kwargs)
        self.frame.at[name, 'subindex'] = list(frame.index)
        frame.loc[:, 'Nt'] = self.frame.at[name, 'Nt'] / Nf
        return frame

    def get_iloc(self, index):
        iloc = [None, None]
        for name in index:
            if name in self.frame.index:
                iloc[0] = self.frame.index.get_loc(index[0])
                subindex = self.frame.subindex[index[0]][0]
                iloc[1] = self.subframe.index.get_loc(subindex)
                break
        return iloc

    def drop_coil(self, index=None):
        if index is None:  # drop all coils
            index = self.frame.index
        if not pd.api.types.is_list_like(index):
            index = [index]
        iloc = self.get_iloc(index)
        for name in index:
            if name in self.frame.index:
                self.subframe.drop_coil(self.frame.loc[name, 'subindex'])
                self.frame.drop_coil(name)
                for M in self.inductance:
                    self.inductance[M].drop(index=name, columns=name,
                                            inplace=True, errors='ignore')
        return iloc

    def add_plasma(self, *args, **kwargs):
        label = kwargs.pop('label', 'Pl')  # filament prefix
        name = kwargs.pop('name', 'Pl_0')
        part = kwargs.pop('part', 'Plasma')
        coil = kwargs.pop('coil', 'Plasma')
        cross_section = kwargs.pop('cross_section', 'ellipse')
        turn_section = kwargs.pop('turn_section', 'square')
        iloc = [None, None]
        if 'Plasma' in self.frame.index:
            iloc = self.drop_coil('Plasma')
        nlist = sum([1 for arg in args if pd.api.types.is_list_like(arg)])
        if nlist == 0:   # add single plasma coil - mesh filaments
            dCoil = kwargs.pop('dCoil', self.dPlasma)
            self.add_coil(*args, part=part, name='Plasma',
                          dCoil=dCoil, cross_section=cross_section,
                          turn_section=turn_section, iloc=iloc[1], **kwargs)
        else:  # add single / multiple filaments, fit coil
            # add plasma filaments to subcoil
            subindex = self.subframe.add_coil(
                    *args, label=label, part=part, coil=coil, name=name,
                    cross_section=turn_section, iloc=iloc[1],
                    mpc=True, **kwargs)
            Ip = self.subframe.It[subindex]  # filament currents
            Ip_net = Ip.sum()  # net plasma current
            if not np.isclose(Ip.sum(), 0):
                Nt = Ip / Ip_net  # filament turn number
            else:
                Nt = 1/Ip.size * np.ones(Ip.size)
            self.subframe.loc[subindex, 'Nt'] = Nt
            xpl = self.subframe.x[subindex]  # filament x-location
            zpl = self.subframe.z[subindex]  # filament z-location
            dx = dz = np.sqrt(np.sum(self.subframe.dx[subindex] *
                                     self.subframe.dz[subindex]))
            # add plasma to coil (x_gmd, z_amd)
            Nf = Ip.size
            self.frame.add_coil(gmd(xpl, Nt), amd(zpl, Nt),
                                dz, dx, Nf=Nf, dCoil=None,
                                cross_section=cross_section,
                                name='Plasma', part=part, turn_fraction=1,
                                material='plasma', iloc=iloc[0])
            self.frame.at['Plasma', 'subindex'] = list(subindex)
            # if Nf > 1:
            #     self.inductance('Plasma', update=True)  # re-size plasma coil
            #self.Ic = pd.Series({'Plasma': Ip_net})  # update net current

    def cluster(self, n, eps=0.2):
        '''
        cluster coils using DBSCAN algorithm
        '''
        dbscan = DBSCAN(eps=eps, min_samples=1)
        cluster_index = dbscan.fit_predict(self.frame.loc[:, ['x', 'z']])
        self.frame.loc[:, 'cluster_index'] = cluster_index
        merge_index = []
        for part in self.frame.part.unique():
            coil = self.subset(self.frame.index[self.frame.part == part]).coil
            for cluster in coil.cluster_index.unique():
                index = coil.index[coil.cluster_index == cluster]
                if index.size > 1:
                    for i in range(index.size // n + 1):
                        if i*n != len(index):
                            merge_index.append(index[i*n:(i+1)*n])
        self.frame.drop(columns='cluster_index', inplace=True)
        for index in merge_index:
            self.merge(index)

    def merge(self, coil_index, name=None):
        subframe = self.subset(coil_index)
        x = gmd(subframe.coil.x, subframe.coil.Nt)
        z = amd(subframe.coil.z, subframe.coil.Nt)
        dx = np.max(subframe.coil.x + subframe.coil.dx/2) -\
            np.min(subframe.coil.x - subframe.coil.dx/2)
        dz = np.max(subframe.coil.z + subframe.coil.dz/2) -\
            np.min(subframe.coil.z - subframe.coil.dz/2)
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
        coil_iloc = self.frame.index.get_loc(coil_index[0])
        subcoil_iloc = self.subframe.index.get_loc(
                self.frame.subindex[coil_index[0]][0])
        # remove seperate coils
        self.drop_coil(coil_index)
        # add merged coil
        self.add_coil(x, z, dx, dz, subcoil=False,
                      iloc=coil_iloc, **kwargs)
        # on-demand patch of top level (coil)
        if pd.isnull(subframe.coil.loc[:, 'patch']).any():
            CoilSet.patch_coil(subframe.coil)  # patch on-demand
        self.frame.at[name, 'patch'] = list(subframe.coil.patch)
        # insert multi-polygon
        self.frame.at[name, 'polygon'] = polygon
        # add subcoils
        subindex = self.subframe.add_coil(subframe.subframe, iloc=subcoil_iloc)
        self.frame.at[name, 'subindex'] = list(subindex)
        self.subframe.loc[subindex, 'coil'] = name
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

    def plot(self, subframe=True, plasma=True, label=False, current=None,
             ax=None):
        if ax is None:
            ax = plt.gca()
        if subframe:
            self.plot_coil(self.subframe, ax=ax)
        else:
            self.plot_coil(self.frame, ax=ax)
        if 'Plasma' in self.frame.index and plasma and 'Ic' in self.frame:
            self.label_plasma(ax)
        if label or current:
            self.label_coil(ax, label, current)
        ax.axis('equal')
        ax.axis('off')

    def label_plasma(self, ax, fs=None):
        if fs is None:
            fs = matplotlib.rcParams['legend.fontsize']
        x = self.frame.x['Plasma']
        z = self.frame.z['Plasma']
        '''
        ax.text(x, z, f'{1e-6*self.Ip:1.1f}MA', fontsize=fs,
                ha='center', va='center', color=0.9 * np.ones(3),
                zorder=10)
        '''

    def label_coil(self, ax, label, current, coil=None, fs=None):
        if fs is None:
            fs = matplotlib.rcParams['legend.fontsize']
        if coil is None:
            coil = self.frame
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
                zshift = max([dz / 5, ylim / 3])
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

    cs = CoilSet(dCoil=1.25)


    cs.update_metadata('frame', additional_columns=['R'])

    cs.add_coil(6, -3, 1.5, 1.5, name='PF6', part='PF', Nt=600, It=5e5,
                turn_section='skin', dCoil=0.75)
    '''
    cs.add_coil(7, -0.5, 2.5, 2.5, name='PF8', part='PF', Nt=600, Ic=2e3,
                cross_section='circle')


    #cs.add_coil([2, 2, 3, 3.5], [1, 0, -1, -3], 0.5, 0.3,
    #            name='PF', part='PF', delim='', Nt=300)
    cs.add_coil(4, 0.75, 1.75, 1.8, name='PF4', part='VS3', turn_fraction=0.75,
                Nt=35, dCoil=-1)
    cs.add_coil(5.6, 3.5, 0.52, 0.52, name='PF7', part='vvin', dCoil=0.05,
                Ic=1e6)

    cs.add_plasma(6, [1.5, 2, 2.5], 1.75, 0.4, It=-15e6/3)
    #cs.add_plasma(7, 3, 1.5, 0.5, It=-15e6/3)


    cs.plot(label=True)
    '''









