from nova.coil_frame import CoilFrame
from nova.simulation_scripts import Mutual, Grid
from pandas import Series, DataFrame, concat, isnull
from pandas.api.types import is_list_like
import numpy as np
from amigo.geom import gmd, amd
import functools
import matplotlib
import operator
from amigo.IO import human_format
from sklearn.cluster import DBSCAN
from matplotlib.collections import PatchCollection
from amigo.pyplot import plt
import shapely.geometry
from descartes import PolygonPatch
from os import path
from amigo.IO import class_dir
import nova
from amigo.IO import pythonIO
from amigo.geom import length, xzfun
from scipy.interpolate import interp1d
import matplotlib.colors as mc
import colorsys


class CoilSet(pythonIO):

    '''
    CoilSet:
        instance wrapper for coilset data

    Attributes:
        coil (nova.CoilFrame): coil config
        subcoil (nova.CoilFrame): subcoil config
    '''
    
    # exchange coilset attributes
    _coilset_attributes = ['default_attributes', 
                           'coilset_frames', 
                           'coilset_metadata']

    # main class attribures
    _coilset_frames = ['coil', 'subcoil']
    
    # exchange simulation instances
    _simulation_scripts = ['grid', 'mutual']  

    # additional_columns
    _coil_columns = ['dCoil', 'Nf', 'Nt', 'It', 'Ic', 'mpc',
                     'power', 'plasma', 
                     'subindex', 'cross_section', 'turn_section', 
                     'turn_fraction', 'patch', 'polygon', 'part']
    
    _subcoil_columns = ['dl_x', 'dl_z', 'mpc', 'power', 'plasma', 
                        'coil', 'Nt', 'It', 'Ic', 'cross_section', 'patch',
                        'polygon', 'part']
    
    _default_attributes = {'dCoil': -1, 'dPlasma': 0.25, 'dShell': 0.5, 
                           'turn_fraction': 1, 'turn_section': 'circle', 
                           'current_update': 'full'}

    def __init__(self, **coilset):
        self.initialize_coil()  # initalize coil and subcoil
        self.initialize_simulation_scripts()
        self.coilset = coilset  # exchange coilset and instance attributes

    def initialize_coil(self):
        coil_metadata = {'_additional_columns': self._coil_columns}
        subcoil_metadata = {'_additional_columns': self._subcoil_columns}
        self.coil = CoilFrame(coilframe_metadata=coil_metadata)
        self.subcoil = CoilFrame(coilframe_metadata=subcoil_metadata)
        self.coil.rebuild_coildata()
        self.subcoil.rebuild_coildata()
        
    def initialize_simulation_scripts(self):
        if 'grid' in self._simulation_scripts:
            self.grid = Grid(self.subcoil)
        if 'mutual' in self._simulation_scripts:
            self.mutual = Mutual(self.subcoil)
        
    @property
    def coilset(self):
        coilset_attributes = {attribute: getattr(self, attribute)
                              for attribute in self._coilset_attributes}
        instance_attributes = {}
        for instance in self._simulation_scripts:
            instance_attribute = '_'.join([instance, 'attributes'])
            instance_attributes[instance_attribute] = \
                getattr(getattr(self, instance), 'attributes')
        return {**coilset_attributes, **instance_attributes}

    @coilset.setter
    def coilset(self, coilset):
        for attribute in self._coilset_attributes:
            setattr(self, attribute, coilset.get(attribute, coilset))
        for instance in self._simulation_scripts:
            instance_attribute = '_'.join([instance, 'attributes'])
            setattr(getattr(self, instance), 'attributes',
                    coilset.get(instance_attribute, coilset))

    def append_coilset(self, *args):
        for coilset in args:
            self.coilset = coilset
        
    @property
    def default_attributes(self):
        return self._default_attributes
        
    @default_attributes.setter
    def default_attributes(self, default_attributes):
        for attribute in default_attributes:
            if attribute in self._default_attributes or \
                    attribute in self._coil_columns + self._subcoil_columns:
                self._default_attributes[attribute] = \
                    default_attributes[attribute]
        self.coil.coilframe_metadata = {
                '_default_attributes': self._default_attributes}

    @property
    def coilset_frames(self):
        coilset_frames = {}
        for frame in self._coilset_frames:
            getattr(self, frame).refresh_dataframe()  # flush dataframe updates
            coilset_frames[frame] = getattr(self, frame)
        return coilset_frames
    
    @coilset_frames.setter
    def coilset_frames(self, coilset_frames):
        for frame in self._coilset_frames:
            coilframe = coilset_frames.get(frame, DataFrame())
            if not coilframe.empty:
                getattr(self, frame).add_coil(coilframe)  # append coilframe
            
    @property
    def coilset_metadata(self):
        'extract coilset metadata from coilset frames (coil, subcoil)'
        coilset_metadata = {}
        for frame in self._coilset_frames:
            coilset_metadata[frame] = getattr(self, frame).coilframe_metadata
        return coilset_metadata
        
    @coilset_metadata.setter
    def coilset_metadata(self, coilset_metadata):
        'update coilframe metadata'
        for frame in self._coilset_frames:
            getattr(self, frame).coilframe_metadata = \
                coilset_metadata.get(frame, coilset_metadata)

    def update_coilframe_metadata(self, coilframe, **coilframe_metadata):
        'update coilset metadata coilframe in [coil, subcoil]'
        getattr(self, coilframe).coilframe_metadata = coilframe_metadata
            
    @staticmethod
    def filepath(filename, directory=None):
        if directory is None:
            directory = path.join(class_dir(nova), '../geom/coilsets')
        return path.join(directory, filename)    
    
    def save_coilset(self, filename, directory=None):
        filepath = self.filepath(filename, directory)
        self._coilset = self.coilset  # link coilset for pythonIO save
        self.save_pickle(filepath, ['_coilset'])
        del self._coilset  # delete temp variable
        
    def load_coilset(self, filename, directory=None, append=False):
        filepath = self.filepath(filename, directory)
        if path.isfile(filepath + '.pk'):
            self.load_pickle(filepath)
            if append:  # append coilset
                self.coilset = self._coilset
            else:  # create new instance
                CoilSet.__init__(self, **self._coilset)
            del self._coilset  # delete temp variable
        else:
            raise LookupError(f'file {filepath} not found')

    def subset(self, index, invert=False):
        if not is_list_like(index):
            index = [index]
        if invert:
            index = self.coil.loc[~self.coil.index.isin(index)].index
        subindex = []
        for _index in index:
            subindex.extend(self.coil.loc[_index, 'subindex'])
        coilset_frames = {'coil': self.coil.loc[index], 
                          'subcoil': self.subcoil.loc[subindex]}
        return CoilSet(coilset_frames=coilset_frames)
        
    @staticmethod
    def categorize_coilset(coil, xo=None, rename=True):
        '''
        categorize coils in coil as CS or PF
        categorization split based on coils minimum radius
        CS coils ordered by x then z
        PF coils ordered by theta taken about coilset centroid
        '''
        if xo is None:
            xo = (coil['x'].mean(), coil['z'].mean())
        # sort CS coils ['x', 'z']
        CSo = coil['x'].idxmin()
        xCS = coil.loc[CSo, 'x'] + coil.loc[CSo, 'dx']
        CS = coil.loc[coil['x'] <= xCS, :]
        CS = CS.sort_values(['x', 'z'])
        CS = CS.assign(part='CS')
        # sort PF coils ['theta']
        PF = coil.loc[coil['x'] > xCS, :]
        PF = PF.assign(theta=np.arctan2(PF['z'], PF['x']))
        PF = PF.sort_values('theta')
        PF.drop(columns='theta', inplace=True)
        PF = PF.assign(part='PF')
        if rename:
            CS.index = [f'CS{i}' for i in range(CS.nC)]
            PF.index = [f'PF{i}' for i in range(PF.nC)]
        coil = concat([PF, CS])
        return coil
        
    @property
    def current_update(self):
        'display current_update status'
        return self.coil.current_update

    @current_update.setter
    def current_update(self, flag):
        self._current_update = flag
        for frame in self._coilset_frames:
            coilframe = getattr(self, frame)
            if hasattr(coilframe, 'current_update'):
                coilframe.current_update = flag
            
    def _set_current(self, value, current_column='Ic'):
        self.relink_mpc()  # relink subcoil mpc as required
        setattr(self.coil, current_column, value)
        setattr(self.subcoil, current_column,
                getattr(self.coil, 
                        current_column)[self.subcoil._update_index])
        
    @property
    def dCoil(self):
        return self.coil._default_attributes['dCoil']
    
    @dCoil.setter
    def dCoil(self, dCoil):
        self.coil._default_attributes['dCoil'] = dCoil
        
    @property
    def Ic(self):
        '''
        Returns:
            self.Ic (np.array): coil instance line subindex current [A]
        '''
        return self.coil.Ic

    @Ic.setter
    def Ic(self, value):
        self._set_current(value, 'Ic')
        
    @property
    def It(self):
        '''
        Returns:
            self.coil.It (np.array): coil instance turn current [A.turns]
        '''
        return self.coil.It

    @It.setter
    def It(self, value):
        self._set_current(value, 'It')
        
    @property
    def Ip(self):
        # return total plasma current
        return self.coil.Ip_sum

    @Ip.setter
    def Ip(self, Ip):
        self.coil.Ip = Ip   
        self.subcoil.Ip = Ip  
        
    @property
    def Np(self):
        return self.subcoil.Np
    
    @Np.setter
    def Np(self, Np):  # set plasma fillament number
        self.subcoil.Np = Np
        self.coil.Np = 1

    def add_coil(self, *args, iloc=None, subcoil=True, **kwargs):
        index = self.coil.add_coil(*args, iloc=iloc, **kwargs)
        if subcoil:
            self.add_subcoil(index=index)

    def add_subcoil(self, index=None, remesh=False, mpc=True,
                    **kwargs):
        if index is None:  # re-mesh all coils
            remesh = True
            index = self.coil.index
        if remesh:
            self.subcoil.drop(self.subcoil.index, inplace=True)
        subcoil = [[] for __ in range(len(index))]
        for i, name in enumerate(index):
            _dCoil = kwargs.get('dCoil', self.coil.at[name, 'dCoil'])
            subcoil[i] = self._mesh_coil(name, dCoil=_dCoil, mpc=mpc)
            xo, zo = self.coil.loc[name, ['x', 'z']]
            for label in ['part', 'power', 'plasma']: 
                if label in self.coil:  # propagate label
                    subcoil[i].loc[:, label] = self.coil.at[name, label]
            if 'rx' in subcoil[i].columns and 'rz' in subcoil[i].columns:
                subcoil[i].loc[:, 'rx'] = subcoil[i].x - xo
                subcoil[i].loc[:, 'rz'] = subcoil[i].z - zo
            subcoil[i].loc[:, 'Ic'] = self.coil.at[name, 'Ic']
        self.subcoil.concatenate(*subcoil)
        
    def add_mpc(self, index, factor=1):
        self.coil.add_mpc(index, factor)
        self.relink_mpc()
        
    def relink_mpc(self):
        if self.coil._relink_mpc:
            self.subcoil._power = self.coil._power[self.coil._mpc_referance]
            self.subcoil.current_update = self.coil._current_update
            self.coil._relink_mpc = False

    def _mesh_coil(self, name, dCoil=None, part=None, mpc=True):
        '''
        mesh coil instance
        '''
        if dCoil is None:
            dCoil = self.coil.at[name, 'dCoil']
        else:
            self.coil.at[name, 'dCoil'] = dCoil  # update
        x, z, dx, dz = self.coil.loc[name, ['x', 'z', 'dx', 'dz']]
        kwargs = {'mpc': mpc}
        if 'turn_section' in self.coil.columns:
            kwargs['cross_section'] = self.coil.at[name, 'turn_section']
        if 'turn_fraction' in self.coil.columns and dCoil == -1:
            turn_fraction = self.coil.at[name, 'turn_fraction']
        else:
            turn_fraction = 1
        if dCoil is None or dCoil == 0:
            dCoil = np.max([dx, dz])
        if dCoil == -1:  # mesh per-turn (for detailed inductance calculations)
            if 'cross_section' not in kwargs:
                kwargs['cross_section'] = 'circle'
            Nt = self.coil.at[name, 'Nt']
            if self.coil.at[name, 'cross_section'] == 'circle':
                dCoil = (np.pi * (dx / 2)**2 / Nt)**0.5
            else:
                dCoil = (dx * dz / Nt)**0.5
        nx = int(np.round(dx / dCoil))
        nz = int(np.round(dz / dCoil))
        if nx < 1:
            nx = 1
        if nz < 1:
            nz = 1
        dx_, dz_ = dx / nx, dz / nz  # subcoil dimensions  
        x_ = x + np.linspace(dx_ / 2, dx - dx_ / 2, nx) - dx / 2
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
        if 'part' in self.coil.columns:
            kwargs['part'] = self.coil.at[name, 'part']
        mesh = {'x': xm_, 'z': zm_,
                'dx': turn_fraction*dx_, 'dz': turn_fraction*dz_}
        mesh['alpha'] = self.coil.loc[name].get('alpha', 0)
        args = []
        for var in mesh:
            if var in self.subcoil._required_columns:
                args.append(mesh[var])
            elif var in self.subcoil._additional_columns:
                kwargs[var] = mesh[var]
        subcoil = self.subcoil.get_coil(*args, name=name, coil=name, **kwargs)
        self.coil.at[name, 'subindex'] = list(subcoil.index)
        subcoil.loc[:, 'Nt'] = self.coil.at[name, 'Nt'] / Nf
        subcoil.rebuild_coildata()
        return subcoil

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
        if not is_list_like(index):
            index = [index]
        iloc = self.get_iloc(index)
        for name in index:
            if name in self.coil.index:
                self.subcoil.drop_coil(self.coil.loc[name, 'subindex'])
                self.coil.drop_coil(name)
        return iloc
    
    @staticmethod
    def _shlspace(segment, dt, rho, dS):
        if not is_list_like(dt):  # permit variable thickness segments
            dt *= np.ones(np.shape(segment)[1])    
        if not is_list_like(rho):  # permit variable resistivity segments
            rho *= np.ones(np.shape(segment)[1])   
        dt_min = dt.min()
        if dS < dt_min:
            dS = dt_min
        dL = length(*segment, norm=False)  # cumulative segment length
        L = dL[-1]  # total segment length
        nS = int(L / dS)  # segment number
        if nS < 1:  # ensure > 0
            nS = 1
        dS = L / nS  # model resolved sub-segment length
        nSS = int(L / dt_min)
        if nSS < 2:
            nSS = 2
        _x, _z = xzfun(*segment)  # xz interpolators
        _dt = interp1d(dL/L, dt)
        _rho = interp1d(dL/L, rho)
        Lend = np.linspace(0, 1, nS+1)  # endpoints
        polygon = [[] for __ in range(nS)]
        x, z = np.zeros(nS), np.zeros(nS)
        rho_bar, dt_bar = np.zeros(nS), np.zeros(nS)
        dx, dz, dA = np.zeros(nS), np.zeros(nS), np.zeros(nS)
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
            dx[i] = dS  # sub-segment length
            dz[i] = dt_bar[i]  # sub-segment thickness
            dA[i] = polygon[i].area
        return x, z, dx, dz, dA, rho_bar, polygon, sub_segment, sub_rho, sub_dt
    
    def add_shell(self, x, z, dt, **kwargs):
        name = kwargs.pop('name', kwargs.get('part', 'Shl'))
        part = kwargs.pop('part', name)
        dShell = kwargs.pop('dShell', self._default_attributes['dShell'])
        dCoil = kwargs.pop('dCoil', self._default_attributes['dCoil'])
        power = kwargs.pop('power', False)
        delim = kwargs.pop('delim', '')
        rho = kwargs.pop('rho', 0)
        x, z, dx, dz, dA, rho_bar, polygon, sub_segment, sub_rho, sub_dt = \
            self._shlspace((x, z), dt, rho, dShell)
        
        #R[i] = resistivity_ss * 2 * np.pi * x[i] / (dx * dz)
        #m[i] = density_ss * 2 * np.pi * x[i] * dx * dz
                    
        index = self.coil.add_coil(x, z, dx, dz, dA=dA, polygon=polygon, 
                                   cross_section='shell', turn_fraction=1, 
                                   turn_section='square', dCoil=dShell,
                                   power=power, name=name, part=part,
                                   delim=delim, Nt=dA, rho=rho_bar)
        subindex = [[] for __ in range(len(index))]
        for i, coil in enumerate(index):
            _x, _z, _dx, _dz, _dA, _rho_bar, _polygon = \
                self._shlspace(sub_segment[i], sub_dt[i], sub_rho[i], 
                               dCoil)[:-3]
            subindex[i] = self.subcoil.add_coil(_x, _z, _dx, _dz, 
                    polygon=_polygon, coil=coil, cross_section='square', 
                    mpc=True, power=power, name=index[i], part=part, Nt=_dA,
                    rho=_rho_bar)
            self.coil.at[index[i], 'subindex'] = subindex[i]
            
    def add_plasma(self, *args, **kwargs):
        label = kwargs.pop('label', 'Pl')  # filament prefix
        name = kwargs.pop('name', 'Pl_0')
        part = kwargs.pop('part', 'Plasma')
        coil = kwargs.pop('coil', 'Plasma')
        cross_section = kwargs.pop('cross_section', 'ellipse')
        turn_section = kwargs.pop('turn_section', 'square')
        iloc = [None, None]
        if 'Plasma' in self.coil.index:
            iloc = self.drop_coil('Plasma')
        nlist = sum([1 for arg in args if is_list_like(arg)])
        if nlist == 0:   # add single plasma coil - mesh filaments
            dCoil = kwargs.pop('dCoil', self.dPlasma)
            self.add_coil(*args, part=part, name='Plasma',
                          dCoil=dCoil, cross_section=cross_section,
                          turn_section=turn_section, iloc=iloc[1], 
                          plasma=True, **kwargs)
        else:  # add single / multiple filaments, fit coil
            # add plasma filaments to subcoil
            subindex = self.subcoil.add_coil(
                    *args, label=label, part=part, coil=coil, name=name,
                    cross_section=turn_section, iloc=iloc[1],
                    mpc=True, plasma=True, **kwargs)
            plasma_index = self.subcoil._plasma_index
            if not np.isclose(self.subcoil.Ip_sum, 0):  # net plasma current
                Nt = self.subcoil.Ip / self.subcoil.Ip_sum  # filament turn number
            else:
                Nt = 1 / self.subcoil.nPlasma * np.ones(self.subcoil.nPlasma)
            self.subcoil.Np = Nt  # set plasma filament turn number
            xpl = self.subcoil.x[plasma_index]  # filament x-location
            zpl = self.subcoil.z[plasma_index]  # filament z-location
            dx = dz = np.sqrt(np.sum(self.subcoil.dx[plasma_index] *
                                     self.subcoil.dz[plasma_index]))
            # add plasma to coil (x_gmd, z_amd)
            Nf = self.subcoil.nP  # number of plasma filaments
            self.coil.add_coil(gmd(xpl, Nt), amd(zpl, Nt),
                                dz, dx, Nf=Nf, dCoil=None,
                                cross_section=cross_section,
                                name='Plasma', part=part, turn_fraction=1,
                                material='plasma', iloc=iloc[0],
                                plasma=True, Ic=self.subcoil.Ip.sum())
            self.coil.at['Plasma', 'subindex'] = list(subindex)
            # if Nf > 1:
            #     self.inductance('Plasma', update=True)  # re-size plasma coil
            #self.Ic = Series({'Plasma': Ip_net})  # update net current

    def cluster(self, n, eps=0.2, merge_pairs=True):
        '''
        cluster coils using DBSCAN algorithm
        '''
        dbscan = DBSCAN(eps=eps, min_samples=1)
        cluster_index = dbscan.fit_predict(self.coil.loc[:, ['x', 'z']])
        cluster_index = Series(cluster_index, index=self.coil.index)
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
            self.merge(index, merge_pairs=merge_pairs)

    def merge(self, coil_index, name=None, merge_pairs=True):
        subset = self.subset(coil_index)
        x = gmd(subset.coil.x, subset.coil.Nt)
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
                if not isnull(polys).any():
                    if merge_pairs:
                        poly_pairs = []
                        for p0, p1 in zip(polys[::2], polys[1::2]):
                            poly_pairs.append(
                                    shapely.geometry.MultiPolygon(
                                    [p0, p1]).minimum_rotated_rectangle)
                        if len(polys) % 2 == 1:  # append last
                            poly_pairs.append(polys[-1])
                        polys = poly_pairs
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
        self.coil.at[name, 'polygon'] = polygon
        # on-demand patch of top level (coil)
        if isnull(subset.coil.loc[:, 'patch']).any():
            CoilSet.patch_coil(subset.coil)  # patch on-demand
        #self.coil.at[name, 'patch'] = list(subset.coil.patch)
        # add subcoils
        subindex = self.subcoil.add_coil(subset.subcoil, iloc=subcoil_iloc,
                                         mpc=True)
        self.coil.at[name, 'subindex'] = list(subindex)
        self.subcoil.loc[subindex, 'coil'] = name
        # update current
        self.Ic = {name: Ic}
        
    def rename(self, index):
        self.coil.rename(index=index, inplace=True)  # rename coil
        for name in index:  # link subcoil
            self.subcoil.loc[self.coil.at[index[name], 'subindex'], 'coil'] = \
                index[name]
        self.subcoil.rebuild_coildata()  # rebuild coildata

    @staticmethod
    def patch_coil(coil, overwrite=False, patchwork_factor=0.01, **kwargs):
        # call on-demand
        part_color = {'VS3': 'C0', 'VS3j': 'gray', 'CS': 'C0', 'PF': 'C0',
                      'trs': 'C2', 'vvin': 'C3', 'vvout': 'C4', 'plasma': 'C4',
                      'cryo': 'C5'}
        color = kwargs.get('part_color', part_color)
        zorder = kwargs.get('zorder', {'VS3': 1, 'VS3j': 0, 'CS': 3, 'PF': 2})
        alpha = {'plasma': 0.75}
        patch = [[] for __ in range(coil.nC)]
        for i, (x, z, dx, dz, cross_section,
                current_patch, polygon, part) in enumerate(
                coil.loc[:, ['x', 'z', 'dx', 'dz', 'cross_section', 'patch',
                              'polygon', 'part']].values):
            if overwrite or np.array(isnull(current_patch)).any():
                patch[i] = [PolygonPatch(polygon)]
            else:
                patch[i] = [current_patch]
            for j in range(len(patch[i])):
                patch[i][j].set_edgecolor('darkgrey')
                patch[i][j].set_linewidth(0.25)
                patch[i][j].set_antialiased(True)
                patch[i][j].set_facecolor(color.get(part, 'C9'))
                patch[i][j].set_zorder = zorder.get(part, 0)
                patch[i][j].set_alpha(alpha.get(part, 1))
                if patchwork_factor != 0:
                    CoilSet.patchwork(i, patch[i][j], patchwork_factor)
        coil.loc[:, 'patch'] = patch
        
    @staticmethod
    def patchwork(i, patch, factor):
        'alternate facecolor lightness by +- factor'
        factor *= 1 - 2 * (i % 2)
        c = patch.get_facecolor()
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        c = colorsys.hls_to_rgb(
                c[0], max(0, min(1, (1 + factor) * c[1])), c[2])
        patch.set_facecolor(c)
        

    def plot_coil(self, coil, alpha=1, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if not coil.empty:
            if isnull(coil.loc[:, 'patch']).any() or len(kwargs) > 0:
                CoilSet.patch_coil(coil, **kwargs)  # patch on-demand
            patch = coil.loc[:, 'patch']
            # form list of lists
            patch = [p if is_list_like(p) else [p] for p in patch]
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
        plasma_index = self.coil._plasma_index
        x = self.coil.x[plasma_index]
        z = self.coil.z[plasma_index]
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
        if label == True:
            label = parts
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
            if part in parts and part in label:
                ax.text(x + drs, z + zshift, name, fontsize=fs,
                        ha=ha, va='center', color=0.2 * np.ones(3))
            if part in parts and current:
                if current == 'Ic':  # line current, amps
                    unit = 'A'
                    Ilabel = coil.at[name, 'Ic']
                elif current == 'It':  # turn current, amp turns
                    unit = 'At'
                    Ilabel = coil.at[name, 'It']
                else:
                    raise IndexError(f'current {current} not in [Ic, It]')
                txt = f'{human_format(Ilabel, precision=1)}{unit}'
                ax.text(x + drs, z - zshift, txt,
                        fontsize=fs, ha=ha, va='center',
                        color=0.2 * np.ones(3))


if __name__ == '__main__':

    cs = CoilSet(dCoil=3, current_update='full', turn_fraction=0.5,
                 cross_section='circle', mutual=True)
    
    '''
    cs.coilset_metadata = {'_default_attributes': {'dCoil': -1}}
    cs.coil.coilframe_metadata = {'_default_attributes': {'dPlasma': 0.333}}
    cs.update_coilframe_metadata('coil', additional_columns=['R'])
    
    cs.add_coil(7, -3, 1.5, 1.5, name='PF6', part='PF', Nt=500, It=1e6,
                turn_section='circle', turn_fraction=0.7, dCoil=0.12)   
    
    cs.add_coil(7, -0.5, 1.5, 1.5, name='PF8', part='PF', Nt=500, Ic=2e3,
                cross_section='circle', turn_section='square', dCoil=0.12)
    
    #cs.add_mpc(['PF6', 'PF8'])

    cs.add_coil(6, -5, 1.5, 1.5, name='PF12', part='PF', Nt=600, It=5e5,
                turn_section='circle', turn_fraction=0.7, dCoil=0.75,
                plasma=True) 
    '''
    cs.add_coil(9, -3, 1.5, 1.5, name='PF13', part='PF', Nt=400, It=5e5,
                turn_section='circle', turn_fraction=0.7, dCoil=0.75,
                plasma=True) 
    
    
    
    '''
    cs.add_coil([2, 2, 3, 3.5], [1, 0, -1, -3], 0.5, 0.3,
                name='PF', part='PF', delim='', Nt=30)
    cs.add_coil(4, 0.75, 1.75, 1.8, name='PF4', part='VS3', turn_fraction=0.75,
                Nt=350, dCoil=-1, power=False)
    
    cs.add_coil(5.6, 3.5, 0.52, 0.52, name='PF7', part='vvin', dCoil=0.05,
                Ic=1e6, Nt=7)
    cs.add_plasma(6, [1.5, 2, 2.5], 1.75, 0.4, It=-15e6)
    cs.add_plasma(7, 3, 1.5, 0.5, It=-15e6/3)
    
    cs.plot(label=True)
    '''
    
    cs.add_shell([4, 6, 7, 9], [1, 1, 2, 1.5], 0.1, 
                 dShell=1, dCoil=-1, name='VVin')
    
    
    
    cs.current_update = 'coil'
    cs.Ic = 12
    cs.Ip = -20
        
    cs.Ic = 34
    #cs.coil.Nt = 1
    
    #print(_cs.It)

    #print(_cs.mutual)
    
    #print(_cs.coil._flux)
    '''
    cs.Ic = 222
    
    cs.grid.generate_grid()
    
    
    cs.add_coil(9.6, 3.5, 0.52, 0.52, name='PF19', dCoil=0.05,
                Ic=1e6, Nt=7)
    
    cs.Ic = 333
    '''
    

    cs.plot(label=True)
    cs.grid.generate_grid(n=4e3)
    #cs.grid.plot_grid()
    cs.grid.plot_flux()

    
    '''
    _cs = CoilSet()
    

    import pickle
    cs_p = pickle.dumps(cs.coilset)
    __cs = pickle.loads(cs_p)
    #_cs.append_coilset(__cs)
    
    _cs.coilset = __cs
    
    
    plt.figure()
    _cs.plot(label=True)
    #_cs.grid.generate_grid(n=500)
    #_cs.grid.plot_grid()
    #cs.grid.solve_interaction()
    _cs.grid.plot_flux()
    '''
    

    










