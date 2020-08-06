import re
from warnings import warn
import string

import numpy as np
from pandas import DataFrame, Series, isna, concat
from pandas.api.types import is_list_like, is_dict_like
import shapely.geometry
import shapely.affinity
from shapely.ops import transform

from nova.electromagnetic.coildata import CoilData


class CoilSeries(Series):

    @property
    def _constructor(self):
        return CoilSeries

    @property
    def _constructor_expanddim(self):
        return CoilFrame


class CoilFrame(DataFrame, CoilData):

    '''
    CoilFrame instance inherits from Pandas DataFrame
    Inspiration taken from GeoPandas https://github.com/geopandas
    '''

    _metadata = ['_required_columns', 
                 '_additional_columns', 
                 '_default_attributes',
                 '_coildata_attributes']

    def __init__(self, *args, coilframe_metadata={}, **kwargs):
        self._initialize_coilframe_metadata()
        DataFrame.__init__(self, *args, **kwargs)  # inherit pandas DataFrame
        self.set_dtypes()
        CoilData.__init__(self)  # coil data
        self.coilframe_metadata = coilframe_metadata  # update metadata
        
    def _initialize_coilframe_metadata(self):
        self._initialize_required_columns()
        self._initialize_additional_columns()
        self._initialize_default_attributes()
        self._initialize_data_attributes()
        
    def _initialize_required_columns(self):
        'required input: self.add_coil(*args)'
        self._required_columns = ['x', 'z', 'dl', 'dt']

    def _initialize_additional_columns(self):
        'additional input: self.add_coil(**kwargs)'
        self._additional_columns = ['rms']

    def _initialize_default_attributes(self):
        'default attributes when not set via self.add_coil(**kwargs)'
        self._default_attributes = \
            {'rms': 0., 'dA': 0., 'nx': 1, 'nz': 1, 
             'Ic': 0., 'It': 0., 'm': '', 'R': 0., 'Nt': 1,
             'Nf': 1, 'material': '', 
             'turn_fraction': 1., 'skin_fraction': 1., 'patch': None,
             'cross_section': 'rectangle', 'turn_section': 'rectangle',
             'coil': '', 'part': '', 'subindex': None, 
             'dCoil': 0., 'dl_x': 0., 'dl_z': 0., 'mpc': '', 'polygon': None,
             'power': True, 'plasma': False, 'rho': 0.}
            
    def set_dtypes(self):
        if self.nC > 0:
            dtypes = {}
            for column in self:
                if column in self._default_attributes:
                    dtype = type(self._default_attributes[column])
                    if dtype in [int, float]:
                        dtypes[column] = dtype
            self = self.astype(dtypes)
            
    def _initialize_data_attributes(self):
        'convert list attributes to dict'
        for attributes in ['_coildata_attributes']:
            if not is_dict_like(getattr(self, attributes)):
                setattr(self, attributes, {
                        attribute: None 
                        for attribute in getattr(self, attributes)})
            
    def _update_coilframe_metadata(self, **coilframe_metadata):
        'extract and update coilframe_metadata'
        mode = coilframe_metadata.pop('mode', 'append')  # [overwrite, append]
        for key in coilframe_metadata:
            if mode == 'overwrite':
                null = {} if key.split('_')[-1] == 'attributes' else []
                setattr(self, key, null)
            value = coilframe_metadata.get(key, None)
            if value is not None:
                if key == '_additional_columns':
                    for v in value:  # insert additional columns
                        if v not in getattr(self, key):
                            getattr(self, key).append(v)
                elif key in ['_default_attributes', 
                             '_coildata_attributes']:
                    for k in value:  # set/overwrite dict
                        getattr(self, key)[k] = value[k]
                elif key in self._default_attributes:
                    self._default_attributes[key] = value
                elif key in self._coildata_attributes:
                    self._coildata_attributes[key] = value

    @property
    def coilframe_metadata(self):
        'extract coilframe_metadata attributes'
        self._coildata_attributes = self.coildata_attributes
        return dict((key, getattr(self, key)) for key in self._metadata)
    
    @coilframe_metadata.setter
    def coilframe_metadata(self, coilframe_metadata):
        'update coilframe_metadata attributes'
        self._update_coilframe_metadata(**coilframe_metadata)
        self.coildata_attributes = self._coildata_attributes 
        
    @property
    def _constructor(self):
        return CoilFrame

    @property
    def _constructor_sliced(self):
        return CoilSeries
    
    @property
    def nC(self):
        '''
        Returns:
            number of coils in dataframe
        '''
        return len(self.index)

    @property
    def nCol(self):
        '''
        Returns:
            number of columns
        '''
        return len(self.columns)
    
    def limit(self, index):
        'returns coil limits [xmin, xmax, zmin, zmax]'
        geom = self.loc[index, ['x', 'z', 'dx', 'dz']]
        limit = [min(geom['x'] - geom['dx'] / 2),
                 max(geom['x'] + geom['dx'] / 2),
                 min(geom['z'] - geom['dz'] / 2),
                 max(geom['z'] + geom['dz'] / 2)]
        return limit

    def get_coil(self, *args, **kwargs):
        mpc = kwargs.pop('mpc', False)
        args, kwargs = self._check_arguments(*args, **kwargs)
        delim = kwargs.pop('delim', '_')
        label = kwargs.pop('label', 'Coil')
        name = kwargs.pop('name', None)
        data = self._extract_data(*args, **kwargs)
        index = self._extract_index(data, delim, label, name)
        coil = CoilFrame(data, index=index, columns=data.keys(),
                         coilframe_metadata=self.coilframe_metadata)
        self.generate_polygon(coil)
        if mpc and coil.nC > 1:
            coil.add_mpc(coil.index.to_list())
        return coil

    def concatenate(self, *coil, iloc=None, sort=False):
        if iloc is None:  # append
            coils = [self, *coil]
        else:  # insert
            coils = [self.iloc[:iloc, :], *coil, self.iloc[iloc:, :]]
        coil = concat(coils, sort=sort)  # concatenate
        # new instance
        CoilFrame.__init__(self, coil, 
                           coilframe_metadata=self.coilframe_metadata)  
        self.rebuild_coildata()  # rebuild fast index
        
    def add_coil(self, *args, iloc=None, **kwargs):
        coil = self.get_coil(*args, **kwargs)  # additional coils
        self.concatenate(coil, iloc=iloc)
        return coil.index

    def drop_coil(self, index=None):
        if index is None:
            index = self.index
        self.drop_mpc(index)    
        self.drop(index, inplace=True)
        self.rebuild_coildata()
        
    def translate(self, index=None, dx=0, dz=0):
        if index is None:
            index = self.index
        elif not is_list_like(index):
            index = [index]
        if dx != 0:
            self.loc[index, 'x'] += dx
        if dz != 0:
            self.loc[index, 'z'] += dz
        for name in index:
            self.loc[name, 'polygon'] = \
                shapely.affinity.translate(self.loc[name, 'polygon'], 
                                           xoff=dx, yoff=dz)
            self.loc[name, 'patch'] = None  # re-generate coil patch
        
    def add_mpc(self, name, factor=1):
        '''
        define multi-point constraint linking a set of coils
        name: list of coil names (present in self.index)
        factor: inter-coil coupling factor
        '''
        if not is_list_like(name):
            raise IndexError(f'name: {name} must be list like')
        elif len(name) == 1:
            raise IndexError(f'len({name}) must be > 1')
        if not is_list_like(factor):
            factor = factor * np.ones(len(name)-1)
        elif len(factor) != len(name)-1:
            raise IndexError(f'len(factor={factor}) must == 1 '
                             f'or == len(name={name})-1')
        for n, f in zip(name[1:], factor):
            self.at[n, 'mpc'] = (name[0], f)
        self.rebuild_coildata()
        
    def drop_mpc(self, index):
        'remove multi-point constraints referancing dropped coils'
        if not is_list_like(index):
            index = [index]
        name = [mpc[0] if mpc else '' for mpc in self.mpc]
        drop = [n in index for n in name]
        self.remove_mpc(drop)
        
    def remove_mpc(self, index):
        'remove multi-point constraint on indexed coils'
        if not is_list_like(index):
            index = [index]
        self.loc[index, 'mpc'] = ''
         
    def reduce_mpc(self, matrix):
        'apply mpc constraints to coupling matrix'
        _matrix = matrix[:, self._mpc_iloc]  # extract primary coils
        if len(self._mpl_index) > 0:  # add multi-point links 
            _matrix[:, self._mpl_index[:, 0]] += \
                matrix[:, self._mpl_index[:, 1]] * \
                np.ones((len(matrix), 1)) @ self._mpl_factor.reshape(-1, 1)
        return _matrix

    def _check_arguments(self, *args, **kwargs):
        if len(args) == 1:  # data passed as CoilFrame
            coilframe = args[0]
            if isinstance(coilframe, CoilFrame):
                if not hasattr(coilframe, 'coildata_attributes'):
                    CoilData.__init__(coilframe)  # re-initialize from pickle
            args = [coilframe.loc[:, col] for col in self._required_columns]
            kwargs['name'] = coilframe.index
            for col in coilframe.columns:
                if col not in self._required_columns:
                    if col in self._additional_columns:
                        kwargs[col] = coilframe.loc[:, col]
        elif len(self._required_columns) != len(args):  # set from kwargs
            raise IndexError(f'\nincorrect argument number: {len(args)}\n'
                             f'input *args as {self._required_columns} '
                             '\nor set _default_columns=[*] in kwarg')
        for key in self._additional_columns:
            if key not in kwargs and key not in self._default_attributes:
                raise KeyError(f'default_attributes not set for {key} in '
                               f' {self._default_attributes.keys()}')
        return args, kwargs

    def _extract_data(self, *args, **kwargs):
        data = {}  # python 3.6+ assumes dict is insertion ordered
        for key, arg in zip(self._required_columns, args):
            data[key] = arg  # add required arguments
        current_label = self._extract_current_label(**kwargs)
        for key in self._additional_columns:
            if key in kwargs:
                data[key] = kwargs.pop(key)
            else:
                data[key] = self._default_attributes[key]
        for key in self._default_attributes:
            additional_columns = []
            if key in kwargs:
                additional_columns.append(key)
                data[key] = kwargs.pop(key)
        self._update_coilframe_metadata(additional_columns=additional_columns)
        self._propogate_current(current_label, data)
        if len(kwargs.keys()) > 0:
            warn(f'\n\nunset kwargs: {list(kwargs.keys())}'
                 '\nto use include within additional_columns:\n'
                 f'{self._additional_columns}'
                 '\nor within default_attributes:\n'
                 f'{self._default_attributes}\n')
        return data

    def _extract_current_label(self, **kwargs):
        current_label = None
        if 'Ic' in self._required_columns or 'Ic' in kwargs:
            current_label = 'Ic'
        elif 'It' in self._required_columns or 'It' in kwargs:
            current_label = 'It'
        return current_label

    def _propogate_current(self, current_label, data):
        if current_label == 'Ic':
            data['It'] = data['Ic'] * data['Nt']
        elif current_label == 'It':
            data['Ic'] = data['It'] / data['Nt']

    def _extract_index(self, data, delim, label, name):
        try:
            nCol = np.max([len(data[key]) for key in data
                           if is_list_like(data[key])])
        except ValueError:
            nCol = 1  # scalar input
        if is_list_like(name):
            if len(name) != nCol:
                raise IndexError(f'missmatch between name {name} and '
                                 f'column number: {nCol}')
            index = name
        else:
            if name is None:
                try:  # reverse search through coilframe index
                    offset = next(int(re.sub(r'[a-zA-Z]', '', index))
                                  for index in self.index[::-1] 
                                  if label in index) + 1
                except StopIteration:  # label not present in index
                    offset = 0
            else:
                if delim:
                    label = name.split(delim)[0]
                    try:
                        index = name.split(delim)[1]
                    except IndexError:
                        index = ''
                else:
                    label = name.rstrip(string.digits)  # trailing  number
                    index = name.rstrip(string.ascii_letters)
                try:  # build list taking starting index from name
                    offset = int(re.sub(r'[a-zA-Z]', '', index))
                except ValueError:
                    offset = 0
            if nCol > 1 or name is None:
                index = [f'{label}{delim}{i+offset:d}' for i in range(nCol)]
            else:
                index = [name]
        self._check_index(index)
        return index

    def _check_index(self, index):
        for name in index:
            if name in self.index:
                raise IndexError(f'\ncoil: {name} already defined in index\n'
                                 f'index: {self.index}')

    def generate_polygon(self, coil):
        if 'polygon' in coil.columns:
            for index in coil.index:
                cross_section = coil.at[index, 'cross_section']
                x, z, dl, dt = coil.loc[index, ['x', 'z', 'dl', 'dt']]
                if cross_section in ['circle', 'square']:
                    dl = dt = np.min([dl, dt])  # set aspect equal
                if isna(coil.at[index, 'polygon']):
                    polygen = self._get_polygen(cross_section)
                    polygon = polygen(x, z, dl, dt)
                    coil.at[index, 'polygon'] = polygon
            self.update_polygon(coil)
            
    def update_polygon(self, coil):
        for index in coil.index:
            polygon = coil.at[index, 'polygon']
            if polygon is not None:
                cross_section = coil.at[index, 'cross_section']
                dl, dt = coil.loc[index, ['dl', 'dt']]
                dA = polygon.area  # update polygon area
                x = polygon.centroid.x  # update x centroid
                z = polygon.centroid.y  # update z centroid
                coil.at[index, 'x'] = x
                coil.at[index, 'z'] = z
                if dA == 0:
                    err_txt = f'zero area polygon entered for coil {index}\n'
                    err_txt += f'cross section: {cross_section}\n'
                    err_txt += f'dl {dl}\ndt {dt}'
                    raise ValueError(err_txt)
                else:
                    coil.at[index, 'dA'] = dA
                bounds = polygon.bounds
                coil.at[index, 'dx'] = bounds[2] - bounds[0]
                coil.at[index, 'dz'] = bounds[3] - bounds[1]
                
                if cross_section == 'circle':
                    rms = np.sqrt(x**2 + dl**2 / 16)  # circle
                elif cross_section in ['square', 'rectangle']:
                    rms = np.sqrt(x**2 + dl**2 / 12)  # square
                elif cross_section == 'skin':
                    rms = np.sqrt((dl**2 * dt**2 / 24 - dl**2 * dt / 8 
                                   + dl**2 / 8 + x**2))
                else:  # calculate directly from polygon
                    p = coil.at[index, 'polygon']
                    rms = (transform(lambda x, z: 
                                     (x**2, z), p).centroid.x)**0.5
                coil.at[index, 'rms'] = rms

        
    @staticmethod
    def _get_polygen(cross_section):
        if cross_section == 'circle':
            return CoilFrame._poly_circle
        elif cross_section == 'ellipse':
            return CoilFrame._poly_ellipse
        elif cross_section == 'square':
            return CoilFrame._poly_square
        elif cross_section == 'rectangle':
            return CoilFrame._poly_rectangle
        elif cross_section == 'skin':
            return CoilFrame._poly_skin
        else:
            raise IndexError(f'cross_section: {cross_section} not implemented'
                             '\n specify as [circle, ellipse, square, '
                             'rectangle, skin]')

    @staticmethod
    def _poly_circle(x, z, dx, dz):
        radius = dx / 2
        circle = shapely.geometry.Point(x, z).buffer(radius)
        return shapely.geometry.Polygon(circle.exterior)

    @staticmethod
    def _poly_ellipse(x, z, dx, dz):
        circle = CoilFrame._poly_circle(x, z, dx, dx)
        return shapely.affinity.scale(circle, 1, dz/dx)
    
    @staticmethod
    def _poly_square(x, z, dx, dz):
        return shapely.geometry.box(x-dx/2, z-dx/2, x+dx/2, z+dx/2)    

    @staticmethod
    def _poly_rectangle(x, z, dx, dz):
        return shapely.geometry.box(x-dx/2, z-dz/2, x+dx/2, z+dz/2)

    @staticmethod
    def _poly_skin(x, z, d, dt):
        '''
        dx: fractional thickness
        dz: circle diameter
        '''
        if dt < 0 or dt > 1:
            raise ValueError(f'skin fractional thickness not 0 <= {dt} <= 1')
        circle_outer = CoilFrame._poly_circle(x, z, d, d)
        if dt == 1:
            shape = circle_outer
        else:
            if dt == 0:
                dt = 1e-3
            circle_inner = shapely.affinity.scale(circle_outer, (1-dt), (1-dt))
            shape = circle_outer.difference(circle_inner)
        return shape
    
    def __setattr__(self, key, value):
        if key in self._coilframe_attributes:
            self._update_dataframe[key] = True
            if key not in self._coildata_properties:
                # set as private variable
                if key in self._mpc_attributes:
                    nC = self._nC  # mpc variable
                else:
                    nC = self.nC  # coil number
                if not is_list_like(value):
                    value *= np.ones(nC)
                if len(value) != nC:
                    raise IndexError('Length of mpc vector does not match '
                                     'length of index')
                key = f'_{key}'
        return DataFrame.__setattr__(self, key, value)
    
    def __getattr__(self, key):
        if key in self._coilframe_attributes:
            value = getattr(self, f'_{key}')
            if key in self._mpc_attributes:  # inflate
                value = value[self._mpc_referance]
            return value
        else:
            return DataFrame.__getattr__(self, key)
        
    def __setitem__(self, key, value):
        'subclass dataframe setitem'
        self.refresh_dataframe()  # flush dataframe updates
        if key in self._coilframe_attributes:
            DataFrame.__setitem__(self, key, value)
            self.refresh_coilframe(key)
            if key in ['Nt', 'It', 'Ic']:
                self._It = self.It
            if key == 'Nt':
                self._update_dataframe['Ic'] = True
                self._update_dataframe['It'] = True
            if key in ['Ic', 'It']:
                _key = next(k for k in ['Ic', 'It'] if k != key)
                self._update_dataframe[_key] = True
        else:
            return DataFrame.__setitem__(self, key, value)
    
    def __getitem__(self, key):  
        'subclass dataframe getitem'
        if key in self._coilframe_attributes:
            self.refresh_dataframe()
        return DataFrame.__getitem__(self, key)
    
    def _get_value(self, index, col, takeable=False):
        'subclass dataframe get_value'
        if col in self._coilframe_attributes:
            self.refresh_dataframe()
        return DataFrame._get_value(self, index, col, takeable)
    
    def __repr__(self):
        self.refresh_dataframe()
        return DataFrame.__repr__(self)
        
