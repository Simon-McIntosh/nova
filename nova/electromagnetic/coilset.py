"""
Construct coilsets for electromagnetic calculation.

Coilsets constructed from a pair of CoilFrame objects derived from
pands.DataFrames. Fast access to and from CoilFrame objects managed via the
CoilData Class.
"""

from os import path
import functools
import operator
import colorsys
import inspect

import numpy as np
from pandas import Series, DataFrame, concat, isnull, Index
from pandas.api.types import is_list_like
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.colors as mc
from sklearn.cluster import DBSCAN
import shapely.geometry
import shapely.affinity
from shapely.strtree import STRtree
from descartes import PolygonPatch
from scipy.interpolate import interp1d

from nova.definitions import root_dir
from nova.utilities.geom import amd
from nova.utilities.IO import human_format
from nova.utilities.pyplot import plt
from nova.utilities.IO import pythonIO
from nova.utilities.geom import length, xzfun

from nova.electromagnetic.coilframe import CoilFrame
from nova.electromagnetic.coildata import CoilData
from nova.electromagnetic.biotmethods import BiotMethods
from nova.electromagnetic.plasma import PlasmaMethods


class CoilSet(pythonIO, BiotMethods, PlasmaMethods):
    """
    Instance wrapper for coilset data.

    Attributes :
        coil (nova.CoilFrame): coil config
        subcoil (nova.CoilFrame): subcoil config
    """

    # exchange coilset attributes
    _coilset_attributes = ['default_attributes',
                           'coilset_frames',
                           'coilset_metadata',
                           'coildata_attributes',
                           'biot_instances',
                           'biot_attributes']

    # main class attribures
    _coilset_frames = ['coil', 'subcoil']

    # additional_columns
    _coil_columns = ['dx', 'dz', 'dA', 'dCoil', 'nx', 'nz', 'subindex', 'part',
                     'cross_section',
                     'turn_section', 'turn_fraction', 'skin_fraction',
                     'patch', 'polygon',
                     'power', 'optimize', 'plasma', 'mpc', 'Nf', 'Nt',
                     'It', 'Ic', 'Psi', 'Bx', 'Bz', 'B']

    _subcoil_columns = ['dx', 'dz', 'dA', 'dl_x', 'dl_z', 'coil', 'part',
                        'cross_section', 'patch', 'polygon',
                        'power', 'optimize', 'plasma', 'mpc', 'Nt', 'It', 'Ic',
                        'Psi', 'Bx', 'Bz', 'B']

    _coildata_attributes = {'current_update': 'full'}

    # fast access np.array linked to CoilFrame via DataFrame
    _dataframe_attributes = ['x', 'z', 'dl', 'dt', 'rms', 'dx', 'dz',
                             'Ic', 'It', 'Nt', 'Psi', 'Bx', 'Bz', 'B',
                             'Fx', 'Fz', 'xFx', 'xFz', 'zFx', 'zFz', 'My']

    def __init__(self, **coilset):
        super().__init__()
        #BiotMethods.__init__(self)  # initialize biotmethods
        self._initialize_coilset()  # initialize coil and subcoil
        self._initialize_biot()  # setup biot instances
        self.coilset = coilset  # exchange coilset and instance attributes

    def _initialize_coilset(self):
        self._extract_coilset_properties()
        self._initialize_default_attributes()
        coil_metadata = {'_additional_columns': self._coil_columns,
                         '_dataframe_attributes': self._dataframe_attributes,
                         '_coildata_attributes': {**self._coildata_attributes,
                                                  **{'subcoil': False}}}
        subcoil_metadata = {'_additional_columns': self._subcoil_columns,
                            '_dataframe_attributes':
                                self._dataframe_attributes,
                            '_coildata_attributes':
                                {**self._coildata_attributes,
                                 **{'subcoil': True}}}
        self.coil = CoilFrame(coilframe_metadata=coil_metadata)
        self.subcoil = CoilFrame(coilframe_metadata=subcoil_metadata)

    def _extract_coilset_properties(self):
        self._coilset_properties = [p for p, __ in inspect.getmembers(
            CoilSet, lambda o: isinstance(o, property))]

    def _initialize_default_attributes(self):
        self._default_attributes = {
            'dCoil': -1, 'dPlasma': 0.25, 'dShell': 0.5, 'dField': 0.2,
            'turn_fraction': 1, 'turn_section': 'circle'}

    def _initialize_biot(self):
        """Specify default biot instances."""
        self.biot_instances = {'field': 'field',
                               'forcefield': 'mutual',
                               #'plasmagrid': 'plasmagrid',
                               #'plasmafilament': 'plasmafilament',
                               'grid': 'grid'}

    @property
    def coilset(self):
        """
        Return dict of coilset attributes listed in self._coilset_attributes.

        Coilset property used to get and set coilset attributes.

        Returns
        -------
        coilset_attributes : dict
            Coilset attributes listed in self._coilset_attributes.

        """
        coilset_attributes = {attribute: getattr(self, attribute)
                              for attribute in self._coilset_attributes}
        return coilset_attributes

    @coilset.setter
    def coilset(self, coilset_attributes):
        for attribute_name in self._coilset_attributes:
            if attribute_name in ['default_attributes', 'coilset_metadata',
                                  'coildata_attributes', 'biot_attributes']:
                default = coilset_attributes
            else:  # require attributes to be passed within attribute dict
                default = {}
            setattr(self, attribute_name,
                    coilset_attributes.get(attribute_name, default))

    @property
    def coildata_attributes(self):
        """
        Expose coildata_attributes from self.coil.

        Returns
        -------
        coildata_attributes : dict
            coildata attributes exposed from self.coil CoilFrame.

        """
        return self.coil.coildata_attributes

    @coildata_attributes.setter
    def coildata_attributes(self, coildata_attributes):
        """
        Set coildata_attriutes.

        Update only.
        Create new attributes via self.coilset_metadata[_coildata_attributes]

        Parameters
        ----------
        coildata_attributes : dict
            coildata_attributes.

        Returns
        -------
        None.

        """
        for attribute in self.coildata_attributes:
            if attribute in self._coilset_properties and \
                    attribute in coildata_attributes and \
                    not hasattr(self, f'_{attribute}'):
                setattr(self, attribute, coildata_attributes[attribute])

    def append_coilset(self, *args):
        """
        Append coilsets via coilset.setter.

        Parameters
        ----------
        *args : dict
            Coilset_attributes.

        Returns
        -------
        None.

        """
        for coilset in args:
            self.coilset = coilset

    @property
    def default_attributes(self):
        for attribute in self._default_attributes:
            if attribute in self._coilset_properties \
                    and hasattr(self, f'_{attribute}'):  # update
                self._default_attributes[attribute] = getattr(self, attribute)
        return self._default_attributes

    @default_attributes.setter
    def default_attributes(self, default_attributes):
        for attribute in default_attributes:
            if attribute in self._coil_columns + self._subcoil_columns +\
                    self._coilset_properties:
                if attribute in self._coilset_properties and not \
                        hasattr(self, f'_{attribute}'):
                    setattr(self, attribute, default_attributes[attribute])
                self._default_attributes[attribute] = \
                        default_attributes[attribute]

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
                CoilData.__init__(coilframe)  # re-initalize coildata
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
            directory = path.join(root_dir, 'data/Nova/coilsets')
        return path.join(directory, filename)

    def save_coilset(self, filename, directory=None):
        filepath = self.filepath(filename, directory)
        self._coilset = self.coilset  # link coilset for pythonIO save
        self.save_pickle(filepath, ['_coilset'])
        del self._coilset  # delete temp variable

    def load_coilset(self, filename, directory=None):
        filepath = self.filepath(filename, directory)
        if path.isfile(filepath + '.pk'):
            self.load_pickle(filepath)
            self.coilset = self._coilset
            #del self._coilset  # delete temp variable
        else:
            raise LookupError(f'file {filepath} not found')
        return self.coilset

    def subset(self, index, invert=False):
        if not isinstance(index, Index):
            index = self.coil.index[index]
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

    def categorize_coilset(self, rename=False):
        '''
        categorize coils in coil as CS or PF
        categorization split based on coils minimum radius
        CS coils ordered by x then z
        PF coils ordered by theta taken about coilset centroid
        '''
        # set part labels
        label = {part: 'part' in self.coil.part.values for part in ['CS','PF']}
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
            other = DataFrame()
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
        self.coil = concat([PF, CS, other])

    @property
    def current_index(self):
        'display power, plasma and current_update status'
        return self.coil.current_index

    @property
    def current_update(self):
        'display current_update status'
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
        self.subcoil._set_current(
            self.coil.Ic[self.subcoil._current_index], 'Ic')
        self.update_field()

    def update_field(self):
        self.coil.refresh_dataframe()  # flush updates
        if self.field.nT > 0:  # maximum of coil boundary values
            frame = self.field.frame
            self.coil.loc[frame.index, frame.columns] = frame

    def update_forcefield(self, subcoil=False):

        if subcoil and self.forcefield.reduce_target:
            self.forcefield.solve(reduce_target=False)

        for variable in ['Psi', 'Bx', 'Bz']:
            setattr(self.subcoil, variable,
                    getattr(self.forcefield, variable))
        self.subcoil.B = \
            np.linalg.norm([self.subcoil.Bx, self.subcoil.Bz], axis=0)
        # set coil variables to maximum of subcoil bundles

        for variable in ['Psi', 'Bx', 'Bz', 'B']:
            setattr(self.coil, variable,
                    np.maximum.reduceat(getattr(self.subcoil, variable),
                                        self.subcoil._reduction_index))

    def _check_default(self, attribute):
        if not hasattr(self, f'_{attribute}'):
            setattr(self, f'_{attribute}', self._default_attributes[attribute])

    @property
    def dCoil(self):
        self._check_default('dCoil')
        return self._dCoil

    @dCoil.setter
    def dCoil(self, dCoil):
        self._dCoil = dCoil
        self._default_attributes['dCoil'] = dCoil
        self.coil._default_attributes['dCoil'] = dCoil

    @property
    def dField(self):
        self._check_default('dField')
        return self._dField

    @dField.setter
    def dField(self, dField):
        self._dField = dField

    @property
    def dPlasma(self):
        self._check_default('dPlasma')
        return self._dPlasma

    @dPlasma.setter
    def dPlasma(self, dPlasma):
        self._dPlasma = dPlasma
        self._default_attributes['dPlasma'] = dPlasma

    @property
    def dShell(self):
        self._check_default('dShell')
        return self._dShell

    @dShell.setter
    def dShell(self, dShell):
        self._dShell = dShell
        self._default_attributes['dShell'] = dShell
        self.coil._default_attributes['dShell'] = dShell

    @property
    def nC(self):
        'number of active coils'
        return self.coil._nC

    @property
    def Ic(self):
        '''
        Returns:
            self.Ic (np.array): coil instance line subindex current [A]
        '''
        return self.coil._Ic

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
        '''
        Returns:
           self.coil.Ip_sum (float): total plasma current
        '''
        return self.coil.Ip_sum

    @Ip.setter
    def Ip(self, Ip):
        self.coil.Ip = Ip
        self.subcoil.Ip = Ip

    @property
    def Np(self):
        return self.subcoil.Np

    @Np.setter
    def Np(self, Np):
        'set plasma fillament number'
        self.subcoil.Np = Np
        self.coil.Np = self.subcoil.Np.sum()

    @property
    def power(self):
        return self.coil.power

    @power.setter
    def power(self, value):
        self.coil.power = value
        self.subcoil.power = value

    @property
    def optimize(self):
        return self.coil.optimize

    @optimize.setter
    def optimize(self, value):
        self.coil.optimize = value
        self.subcoil.optimize = value

    def add_coil(self, *args, iloc=None, subcoil=True, **kwargs):
        index = self.coil.add_coil(*args, iloc=iloc, **kwargs)
        if subcoil:
            self.meshcoil(index=index)

    def add_mpc(self, index, factor=1):
        self.coil.add_mpc(index, factor)
        self.relink_mpc()

    def relink_mpc(self):
        if self.coil._relink_mpc:
            # force dataframe update
            self.coil._update_dataframe['Ic'] = True
            self.coil._update_dataframe['It'] = True
            for attribute in self.coil._coilcurrent_attributes:
                setattr(self.subcoil, attribute, getattr(self.coil, attribute))
            self.subcoil.current_update = self.coil.current_update
            self.coil._relink_mpc = False

    def translate(self, index=None, dx=0, dz=0):
        if index is None:
            index = self.coil.index
        elif not is_list_like(index):
            index = [index]
        self.coil.translate(index, dx, dz)
        for name in index:
            self.subcoil.translate(self.coil.loc[name, 'subindex'], dx, dz)

    def meshcoil(self, index=None, mpc=True, **kwargs):
        coil = kwargs.pop('coil', self.coil)
        coil.generate_polygon()
        coil.update_polygon()
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
                    *subcoil_args, name=name, coil=name, **subcoil_kwargs)
            _subcoil[i].update_polygon()
            # back-propagate fillament attributes to coil
            coil.loc[name, ['Nf', 'nx', 'nz', 'dCoil']] = \
                mesh['Nf'], mesh['nx'], mesh['nz'], mesh['dCoil']
            if 'subindex' in coil:
                coil.at[name, 'subindex'] = list(_subcoil[i].index)
        subcoil.concatenate(*_subcoil)

    @staticmethod
    def _mesh_coil(coil, mpc=True, **kwargs):
        'mesh single coil'
        x, z, dl, dt, dCoil = coil[['x', 'z', 'dl', 'dt', 'dCoil']]
        if 'polygon' in coil:
            coil_polygon = coil.polygon
            bounds = coil_polygon.bounds
            dx = bounds[2] - bounds[0]
            dz = bounds[3] - bounds[1]
        else:  # assume rectangular coil cross-section
            dx, dz = coil[['dl', 'dt']]  # length, thickness == dx, dz
            bounds = (x-dx/2, z-dz/2, x+dx/2, z+dz/2)
            coil_polygon = shapely.geometry.box(bounds)
        coil_polygon = coil_polygon.buffer(1e-12*dCoil)  # offset coil polygon
        # coil_area = coil_polygon.area
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
        elif dCoil == -1:  # mesh per-turn (detailed inductance calculations)
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
        x_ = np.linspace(*bounds[::2], nx+1)
        z_ = np.linspace(*bounds[1::2], nz+1)
        polygen = CoilFrame._get_polygen(cross_section)  # polygon generator
        polygon, xm_, zm_, cs_, dA_ = [], [], [], [], []
        sub_polygons = [[] for __ in range(nx*nz)]
        for i in range(nx):  # radial divisions
            for j in range(nz):  # vertical divisions
                sub_polygons[i*nz + j] = \
                        polygen(x_[i]+dx_/2, z_[j]+dz_/2, dl_, dt_)
        tree = STRtree(sub_polygons)
        sub_polygons = [p for p in tree.query(coil_polygon)
                        if p.intersects(coil_polygon)]
        for sub_polygon in sub_polygons:
            p = coil_polygon.intersection(sub_polygon)
            if isinstance(p, shapely.geometry.polygon.Polygon):
                p = [p]  # single polygon
            for p_ in p:
                if isinstance(p_, shapely.geometry.polygon.Polygon):
                    polygon.append(p_)
                    xm_.append(p_.centroid.x)
                    zm_.append(p_.centroid.y)
                    dA_.append(p_.area)
                    if sub_polygon.within(coil_polygon):
                        cs_.append(cross_section)  # maintain cs referance
                    else:
                        cs_.append('polygon')
        Nf = len(xm_)  # filament number
        if Nf == 0:  # no points found within polygon (skin)
            xm_, zm_, dl_, dt_ = x, z, dl, dt
            Nf = 1
        # constant current density
        Nt_ = coil['Nt']*np.array(dA_) / np.sum(dA_)

        # subcoil bundle
        mesh.update({'x': xm_, 'z': zm_, 'nx': nx, 'nz': nz,
                     'dl': dl_, 'dt': dt_, 'Nt': Nt_, 'Nf': Nf,
                     'polygon': polygon, 'cross_section': cs_,
                     'dCoil': dCoil})

        # subcoil moment arms
        #xo, zo = coil.loc[['x', 'z']]
        #mesh['rx'] = xm_ - xo
        #mesh['rz'] = zm_ - zo

        # propagate current update flags to subcoil
        for label in ['part', 'power', 'optimize', 'plasma']:
            if label in coil:
                mesh[label] = coil[label]
        mesh['Ic'] = coil['Ic']
        mesh['turn_fraction'] = turn_fraction
        return mesh

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
        dL = length(*segment, norm=False)  # cumulative segment length
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
        _x, _z = xzfun(*segment)  # xz interpolators
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

    def add_shell(self, x, z, dt, **kwargs):
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

    def add_plasma(self, boundary, name='Plasma', dPlasma=None):
        """
        Add plasma coil to coilset and generate plasma grid.

        Plasma coil inserted into coilframe with subcoils meshed accoriding
        to dCoil and trimmed to the inital boundary curve.

        Parameters
        ----------
        boundary : array_like or Polygon
            External plasma boundary. Coerced as a positively oriented curve.
            array_like
                shape(4,) [xmin, xmax, zmin, zmax] limit bounding box
                shape(n,2) bounding loop
        name : str, optional
            Plasma coil name.
        dPlasma : float, optional
            Plasma subcoil dimension. If None defaults to self.dPlasma


        Returns
        -------
        None.

        """
        if dPlasma is not None:  # update plasma subcoil dimension
            self.dPlasma = dPlasma
        if not isinstance(boundary, shapely.geometry.Polygon):
            boundary = np.array(boundary)  # to numpy array
            if boundary.ndim == 1:   # limit bounding box
                if len(boundary) == 4:
                    polygon = shapely.geometry.box(*boundary[::2],
                                                   *boundary[1::2])
                else:
                    raise IndexError('malformed bounding box\n'
                                     f'boundary: {boundary}\n'
                                     'require [xmin, xmax, zmin, zmax]')
            elif boundary.ndim == 2 and boundary.shape[1] == 2:  # loop
                polygon = shapely.geometry.Polygon(boundary)
            else:
                raise IndexError('malformed bounding loop\n'
                                 f'shape(boundary): {boundary.shape}\n'
                                 'require (n,2)')
        else:
            polygon = boundary
        # orient polygon
        polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
        # construct plasma coil from polygon
        self.add_coil(0, 0, 0, 0, polygon=polygon, cross_section='rectangle',
                      dCoil=self.dPlasma, name=name, plasma=True,
                      part='plasma')
        # generate plasma grid
        #self.plasma.generate_grid()

        '''
        label = kwargs.pop('label', 'Pl')  # filament prefix
        name = kwargs.pop('name', 'Pl_0')
        part = kwargs.pop('part', 'Plasma')
        coil = kwargs.pop('coil', 'Plasma')
        cross_section = kwargs.pop('cross_section', 'rectangle')
        turn_section = kwargs.pop('turn_section', 'rectangle')
        self.dPlasma = kwargs.pop('dPlasma', self.dPlasma)  # update dPlasma
        iloc = [None, None]  # coil, subcoil
        print(self.coil.part, 'Plasma' in self.coil.part)
        if 'Plasma' in self.coil.part.values:
            print('plasma', self.coil.part)
            iloc = self.drop_coil(self.coil.index[self.coil.part == 'Plasma'])
        nlist = sum([1 for arg in args if is_list_like(arg)])
        if nlist == 0:   # add single plasma coil - mesh filaments
            dCoil = kwargs.pop('dCoil', self.dPlasma)
            self.add_coil(*args,
                          part=part, name='Plasma',
                          dCoil=dCoil, cross_section=cross_section,
                          turn_section=turn_section, iloc=iloc[0],
                          plasma=True, **kwargs)
        else:  # add single / multiple filaments, fit coil
            # add plasma filaments to subcoil
            print(kwargs['It'].sum())
            subindex = self.subcoil.add_coil(
                    *args, label=label, part=part, coil=coil, name=name,
                    cross_section=turn_section, iloc=iloc[1],
                    mpc=True, plasma=True, **kwargs)
            plasma_index = self.subcoil._plasma_index
            print(self.subcoil.Ip_sum)
            if not np.isclose(self.subcoil.Ip_sum, 0):  # net plasma current
                Nt = self.subcoil.Ip / self.subcoil.Ip_sum  # filament turn number
            else:
                Nt = 1 / self.subcoil.nPlasma * np.ones(self.subcoil.nPlasma)
            print(self.subcoil.Ip_sum)
            self.subcoil.Np = Nt  # set plasma filament turn number
            print(self.subcoil.Ip_sum)
            xpl = self.subcoil.x[plasma_index]  # filament x-location
            zpl = self.subcoil.z[plasma_index]  # filament z-location
            dx = dz = np.sqrt(np.sum(self.subcoil.dx[plasma_index] *
                                     self.subcoil.dz[plasma_index]))
            # add plasma to coil (x_gmd, z_amd)
            Nf = self.subcoil.nP  # number of plasma filaments
            self.subcoil.plot()
            print(self.subcoil.Ip.sum())
            self.coil.add_coil(gmd(xpl, Nt), amd(zpl, Nt),
                                dz, dx, Nf=Nf, dCoil=None,
                                cross_section=cross_section,
                                name='Plasma', part=part, turn_fraction=1,
                                material='plasma', iloc=iloc[0],
                                plasma=True, Ic=self.subcoil.Ip_sum)
            self.coil.loc['Plasma', 'subindex'] = list(subindex)
            # if Nf > 1:
            #     self.inductance('Plasma', update=True)  # re-size plasma coil
            #self.Ic = Series({'Plasma': Ip_net})  # update net current
        '''

    def cluster(self, n, eps=0.2, merge_pairs=True):
        """
        Cluster coils using DBSCAN algorithm.

        Parameters
        ----------
        n : int
            Target cluster size.
        eps : float, optional
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. The default is 0.2.
        merge_pairs : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
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
        x = amd(subset.coil.x, subset.coil.Nt)
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
        self.coil.loc[name, 'polygon'] = polygon
        # on-demand patch of top level (coil)
        if isnull(subset.coil.loc[:, 'patch']).any():
            CoilSet.patch_coil(subset.coil)  # patch on-demand
        # add subcoils
        subindex = self.subcoil.add_coil(subset.subcoil, iloc=subcoil_iloc)
        self.coil.loc[name, 'subindex'] = list(subindex)
        self.subcoil.loc[subindex, 'coil'] = name
        self.subcoil.add_mpc(subindex.to_list())
        # update current
        self.forcefield.solve()
        self.Ic = {name: Ic}

    def rename(self, index):
        self.coil.rename(index=index, inplace=True)  # rename coil
        for name in index:  # link subcoil
            self.subcoil.loc[self.coil.loc[index[name], 'subindex'], 'coil'] = \
                index[name]
        self.subcoil.rebuild_coildata()  # rebuild coildata

    @staticmethod
    def patch_coil(coil, overwrite=False, patchwork_factor=0.15, **kwargs):
        # call on-demand
        part_color = {'VS3': 'C0', 'VS3j': 'gray', 'CS': 'C0', 'PF': 'C0',
                      'trs': 'C3', 'dir': 'C3',
                      'vv': 'C3', 'vvin': 'C3', 'vvout': 'C3',
                      'bb': 'C7',
                      'plasma': 'C4', 'Plasma': 'C4',
                      'cryo': 'C5'}
        color = kwargs.get('part_color', part_color)
        zorder = kwargs.get('zorder', {'VS3': 1, 'VS3j': 0, 'CS': 3, 'PF': 2})
        alpha = {'plasma': 0.75}
        if 'coil' not in coil:
            patchwork_factor = 0
        patch = [[] for __ in range(coil.nC)]
        for i, (x, z, dx, dz, cross_section,
                current_patch, polygon, part) in enumerate(
                coil.loc[:, ['x', 'z', 'dx', 'dz', 'cross_section', 'patch',
                              'polygon', 'part']].values):
            if overwrite or np.array(isnull(current_patch)).any():
                if isinstance(polygon, shapely.geometry.Polygon):
                    patch[i] = [PolygonPatch(polygon)]
                else:
                    patch[i] = []
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
                    CoilSet.patchwork(patch[i][j], patchwork_factor)
        coil.loc[:, 'patch'] = np.asarray(patch, object)

    @staticmethod
    def patchwork(patch, factor):
        'alternate facecolor lightness by +- factor'
        factor *= 1 - 2 * np.random.rand(1)[0]
        c = patch.get_facecolor()
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        c = colorsys.hls_to_rgb(
                c[0], max(0, min(1, (1 + factor) * c[1])), c[2])
        patch.set_facecolor(c)


    def plot_coil(self, coil, alpha=1, ax=None, passive=False, **kwargs):
        if ax is None:
            ax = plt.gca()
        if not coil.empty:
            if isnull(coil.loc[:, 'patch']).any() or len(kwargs) > 0:
                CoilSet.patch_coil(coil, **kwargs)  # patch on-demand
            if passive:
                patch = coil.loc[:, 'patch']
            else:  # exclude passive filaments (Nt == 0)
                patch = coil.loc[coil.Nt != 0, 'patch']  # plot iff Nt != 0
            # form list of lists
            patch = [p if is_list_like(p) else [p] for p in patch]
            if len(patch) > 0:
                # flatten
                patch = functools.reduce(operator.concat, patch)
                # sort
                patch = np.array(patch)[np.argsort([p.zorder for p in patch])]
                pc = PatchCollection(patch, match_original=True)
                ax.add_collection(pc)

    def plot(self, subcoil=False, plasma=True, label='coil', current='A',
             field=True, passive=False, ax=None):
        if ax is None:
            ax = plt.gca()
        if subcoil:
            self.plot_coil(self.subcoil, passive=passive, ax=ax)
        else:
            self.plot_coil(self.coil, passive=passive, ax=ax)
        ax.axis('equal')
        ax.axis('off')
        plt.tight_layout()
        if  plasma and self.coil._plasma_index.sum() > 0:
            self.label_plasma(ax)
        if label or current or field:
            self.label_coil(ax, label, current, field)

    def label_plasma(self, ax, fs=None):
        if fs is None:
            fs = matplotlib.rcParams['legend.fontsize']
        plasma_index = self.coil._plasma_index
        x = self.coil.x[plasma_index]
        z = self.coil.z[plasma_index]
        ax.text(x, z, f'{1e-6*self.Ip:1.1f}MA', fontsize='medium',
                ha='center', va='center', color=0.9 * np.ones(3),
                zorder=10)

    def label_gaps(self, ax=None):
        coil_index = []
        for end in ['L', 'U']:
            position = range(1, 4) if end == 'U' else range(3, 0, -1)
            for i in position:
                coil_index.append(f'CS{i}{end}')
        gap_index = ['LDP'] + coil_index + ['LDP']
        if ax is None:
            ax = plt.gca()
        for i, name in enumerate(coil_index):
            x, z, dx, dz = self.coil.loc[name, ['x', 'z', 'dx', 'dz']]
            drs = 2/3*dx
            ax.text(x + drs, z, f'Coil {i}',
                    ha='left', va='center', color=0.2 * np.ones(3))
        xo, zo = self.coil.loc[coil_index[0], ['x', 'z']]
        z1 = self.coil.loc[coil_index[-1], 'z']
        dzo = (z1-zo) / (len(coil_index) - 1)
        z = zo - dzo/2
        for i in range(7):
            ax.text(x - drs, z, f'{gap_index[i]}-{gap_index[i+1]}',
                    ha='right', va='center', color='C3')
            ax.text(x + drs, z, f'Gap {i}',
                    ha='left', va='center', color='C3')
            z += dzo

    def label_coil(self, ax, label='coil', current='A', field=True,
                   coil=None, fs='medium', Nmax=20):
        if coil is None:
            coil = self.coil
        if label == 'all':  # all coils
            parts = coil.part.unique()
        elif label == 'status':  # based on coil.update_status
            parts = coil.part[coil._current_index[coil._mpc_referance]].unique()
        elif label == 'active':  # power == True
            parts = coil.part[coil.power & ~coil.plasma].unique()
        elif label == 'passive':  # power == False
            parts = coil.part[~coil.power & ~coil.plasma].unique()
        elif label == 'coil':  # plasma == False
            parts = coil.part[~coil.plasma].unique()
        elif label == 'plasma':  # plasma == True
            parts = coil.part[coil.plasma].unique()
        elif label == 'free':  # optimize == True
            parts = coil.part[coil.optimize & ~coil.plasma].unique()
        elif label == 'fix':  # optimize == False
            parts = coil.part[~coil.optimize & ~coil.plasma].unique()
        else:
            if not is_list_like(label):
                label = [label]
            parts = self.coil.part.unique()
            parts = [_part for _part in label if _part in parts]
        parts = list(parts)
        N = {p: sum(coil.part == p) for p in parts}
        # referance vertical length scale
        dz_ref = np.diff(ax.get_ylim())[0] / 100
        nz = np.sum(np.array([parts != False, current != None,
                              field != False]))
        if nz == 1:
            dz_ref = 0
        ztext = {name: 0 for name, value
                 in zip(['label', 'current', 'field'],
                        [label, current, field]) if value}
        for name, dz in zip(ztext, nz*dz_ref * np.linspace(1, -1, nz)):
            ztext[name] = dz
        for name, part in zip(coil.index, coil.part):
            if part in parts and N[part] < Nmax:
                x, z = coil.loc[name, 'x'], coil.loc[name, 'z']
                dx = coil.loc[name, 'dx']
                drs = 2/3*dx * np.array([-1, 1])
                if coil.part[name] == 'CS':
                    drs_index = 0
                    ha = 'right'
                else:
                    drs_index = 1
                    ha = 'left'
                # label coil
                ax.text(x + drs[drs_index], z + ztext['label'],
                        name, fontsize=fs, ha=ha, va='center',
                        color=0.2 * np.ones(3))
                if current:
                    if current == 'Ic' or current == 'A':  # line current
                        unit = 'A'
                        Ilabel = coil.loc[name, 'Ic']
                    elif current == 'It' or current == 'AT':  # turn current
                        unit = 'At'
                        Ilabel = coil.loc[name, 'It']
                    else:
                        raise IndexError(f'current {current} not in '
                                         '[Ic, A, It, AT]')
                    txt = f'{human_format(Ilabel, precision=1)}{unit}'
                    ax.text(x + drs[drs_index], z + ztext['current'], txt,
                            fontsize=fs, ha=ha, va='center',
                            color=0.2 * np.ones(3))
                if field:
                    Blabel = coil.loc[name, 'B']
                    txt = f'{human_format(Blabel, precision=2)}T'
                    ax.text(x + drs[drs_index], z + ztext['field'], txt,
                            fontsize=fs, ha=ha, va='center',
                            color=0.2 * np.ones(3))


if __name__ == '__main__':

    cs = CoilSet(dCoil=-1, current_update='coil', turn_fraction=0.5,
                 cross_section='circle')

    cs.add_coil(1.75, 0.5, 2.5, 2.5, name='PF13', part='PF', Nt=10, It=0,
                cross_section='circle', turn_fraction=1,
                dCoil=-15)

    cs.add_plasma([1, 3, 2, 3])
    cs.plot(True)
