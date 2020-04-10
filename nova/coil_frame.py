from nova.coil_data import CoilData
from nova.coil_matrix import CoilMatrix
from pandas import DataFrame, Series, isna, concat
from pandas.api.types import is_list_like, is_dict_like
import numpy as np
from warnings import warn
import shapely.geometry
import shapely.affinity


class CoilSeries(Series):

    @property
    def _constructor(self):
        return CoilSeries

    @property
    def _constructor_expanddim(self):
        return CoilFrame


class CoilFrame(DataFrame, CoilData, CoilMatrix):

    '''
    CoilFrame instance inherits from Pandas DataFrame
    Inspiration taken from GeoPandas https://github.com/geopandas
    '''
    _internal_names = DataFrame._internal_names
    _internal_names += CoilData._coildata_flags
    _internal_names_set = set(_internal_names)

    _metadata = ['_required_columns', 
                 '_additional_columns', 
                 '_default_attributes',
                 '_coildata_attributes',
                 '_coilmatrix_attributes']

    def __init__(self, *args, coilframe_metadata={}, **kwargs):
        self._initialize_coilframe_metadata()
        DataFrame.__init__(self, *args, **kwargs)  # inherit pandas DataFrame
        CoilData.__init__(self)  # coil data
        CoilMatrix.__init__(self)  # coil matrix
        self.coilframe_metadata = coilframe_metadata  # update metadata
        
    def _initialize_coilframe_metadata(self):
        self._initialize_required_columns()
        self._initialize_additional_columns()
        self._initialize_default_attributes()
        self._initialize_data_attributes()
        
    def _initialize_required_columns(self):
        'required input: self.add_coil(*args)'
        self._required_columns = ['x', 'z', 'dx', 'dz']

    def _initialize_additional_columns(self):
        'additional input: self.add_coil(**kwargs)'
        self._additional_columns = []

    def _initialize_default_attributes(self):
        'default attributes when not set via self.add_coil(**kwargs)'
        self._default_attributes = \
            {'Ic': 0, 'It': 0, 'm': '', 'R': 0, 'Nt': 1, 'Nf': 1,
             'material': '', 'turn_fraction': 1, 'patch': None,
             'cross_section': 'square', 'turn_section': 'square',
             'coil': '', 'part': '', 'subindex': None, 'dCoil': 0,
             'dl_x': 0, 'dl_z': 0, 'mpc': '', 'polygon': None,
             'power': True, 'plasma': False}
            
    def _initialize_data_attributes(self):
        'convert list attributes to dict'
        for attributes in ['_coildata_attributes', '_coilmatrix_attributes']:
            if not is_dict_like(getattr(self, attributes)):
                setattr(self, attributes, {
                        attribute: None 
                        for attribute in getattr(self, attributes)})
            
    def _update_coilframe_metadata(self, **coilframe_metadata):
        'extract and update coilframe_metadata'
        mode = coilframe_metadata.pop('mode', 'append')  # [overwrite, append]
        for key in self._metadata:
            if mode == 'overwrite':
                null = {} if key.split('_')[-1] == 'attributes' else []
                setattr(self, key, null)
            value = coilframe_metadata.pop(key, None)
            if value is not None:
                if key == '_additional_columns':
                    for v in value:  # insert additional columns
                        if v not in getattr(self, key):
                            getattr(self, key).append(v)
                elif key in ['_default_attributes', 
                             '_coildata_attributes',
                             '_coilmatrix_attributes']:
                    for k in value:  # set/overwrite dict
                        getattr(self, key)[k] = value[k]
                else:  # overwrite
                    setattr(self, key, value)

    @property
    def coilframe_metadata(self):
        'extract coilframe_metadata attributes'
        self._coildata_attributes = self.coildata
        self._coilmatrix_attributes = self.coilmatrix
        return dict((key, getattr(self, key)) for key in self._metadata)
    
    @coilframe_metadata.setter
    def coilframe_metadata(self, coilframe_metadata):
        'update coilframe_metadata attributes'
        self._update_coilframe_metadata(**coilframe_metadata)
        self.coildata = self._coildata_attributes  # return to coildata
        self.coilmatrix = self._coilmatrix_attributes  # return to coilmatrix
        
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

    def get_coil(self, *args, **kwargs):
        mpc = kwargs.pop('mpc', False)
        args, kwargs = self._check_arguments(*args, **kwargs)
        delim = kwargs.pop('delim', '_')
        label = kwargs.pop('label', kwargs.get('name', 'Coil'))
        name = kwargs.pop('name', f'{label}{delim}{self.nC:d}')
        data = self._extract_data(*args, **kwargs)
        index = self._extract_index(data, delim, label, name)
        coil = CoilFrame(data, index=index, columns=data.keys(),
                         coilframe_metadata=self.coilframe_metadata)
        coil.update_coildata()  # rebuild fast index
        coil = self._insert_polygon(coil)
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
        self.update_coildata()
        #self.concatenate_matrix()
        
    def add_coil(self, *args, iloc=None, **kwargs):
        coil = self.get_coil(*args, **kwargs)  # additional coils
        self.concatenate(coil, iloc=iloc)
        return coil.index

    def drop_coil(self, index=None):
        if index is None:
            index = self.index
        self.drop_mpc(index)    
        self.drop(index, inplace=True)
        self.update_coildata()
        
    def add_mpc(self, name, factor=1):
        '''
        define multi-point constraint linking a set of coils
        name: list of coil names (present in self.coil.index)
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
        self.update_coildata()
        
    def drop_mpc(self, index):
        'remove multi-point constraints referancing dropped coils'
        if not is_list_like(index):
            index = [index]
        name = [mpc[0] if mpc else '' for mpc in self.mpc]
        drop = [n in index for n in name]
        self.loc[drop, 'mpc'] = ''

    def _check_arguments(self, *args, **kwargs):
        if len(args) == 1:  # data passed as pandas dataframe
            data = args[0]
            print('required', self._required_columns)
            args = [data.loc[:, col] for col in self._required_columns]
            kwargs['name'] = data.index
            for col in data.columns:
                if col not in self._required_columns:
                    if col in self._additional_columns:
                        kwargs[col] = data.loc[:, col]
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
        data = self._propogate_current(current_label, data)
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
        return data

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
            if nCol == 1:
                index = [name]
            else:
                index = [f'{label}{delim}{i}' for i in range(nCol)]
        self._check_index(index)
        return index

    def _check_index(self, index):
        for name in index:
            if name in self.index:
                raise IndexError(f'\ncoil: {name} already defined in index\n'
                                 f'index: {self.index}')

    def _insert_polygon(self, coil):
        if 'polygon' in coil.columns:
            for name in coil.index:
                if isna(coil.at[name, 'polygon']):
                    x = coil.at[name, 'x']
                    z = coil.at[name, 'z']
                    dx = coil.at[name, 'dx']
                    dz = coil.at[name, 'dz']
                    cross_section = coil.at[name, 'cross_section']
                    if (np.array([x, dx, dz]) != 0).all():
                        polygen = self._get_polygen(cross_section)
                        polygon = polygen(x, z, dx, dz)
                    else:
                        polygon = None
                    coil.at[name, 'polygon'] = polygon
        return coil

    def _get_polygen(self, cross_section):
        if cross_section == 'circle':
            return self._poly_circle
        elif cross_section == 'ellipse':
            return self._poly_ellipse
        elif cross_section == 'square' or cross_section == 'rectangle':
            return self._poly_rectangle
        elif cross_section == 'skin':
            return self._poly_skin
        else:
            raise IndexError(f'cross_section: {cross_section} not implemented'
                             '\n specify as [circle, ellipse, square, '
                             'rectangle, skin]')

    def _poly_circle(self, x, z, dx, dz):
        radius = np.min([dx, dz]) / 2
        circle = shapely.geometry.Point(x, z).buffer(radius)
        return shapely.geometry.Polygon(circle.exterior)

    def _poly_ellipse(self, x, z, dx, dz):
        circle = self._poly_circle(x, z, dx, dx)
        return shapely.affinity.scale(circle, 1, dz/dx)

    def _poly_rectangle(self, x, z, dx, dz):
        return shapely.geometry.box(x-dx/2, z-dz/2, x+dx/2, z+dz/2)

    def _poly_skin(self, x, z, dx, dz):
        circle_outer = self._poly_circle(x, z, dx, dz)
        circle_inner = shapely.affinity.scale(circle_outer, 0.8, 0.8)
        return circle_outer.difference(circle_inner)
    
    def __setattr__(self, key, value):
        if key in self._coilframe_attributes:
            self._update_coilframe[key] = True
            if key not in self._coildata_properties:
                # set as private variable
                if key in self._mpc_attributes:
                    nC = self._nC  # mpc variable
                else:
                    nC = self.nC  # coil number
                if not is_list_like(value):
                    value *= np.ones(nC)
                if len(value) != nC:
                    raise IndexError('Length of private values does not match '
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
        self.update_dataframe()  # flush dataframe updates
        if key in self._coilframe_attributes:
            DataFrame.__setitem__(self, key, value)
            self.update_data(key)
            if key in ['Nt', 'It', 'Ic']:
                self._It = self.It
            if key == 'Nt':
                self._update_coilframe['Ic'] = True
                self._update_coilframe['It'] = True
            if key in ['Ic', 'It']:
                _key = next(k for k in ['Ic', 'It'] if k != key)
                self._update_coilframe[_key] = True
        else:
            return DataFrame.__setitem__(self, key, value)
    
    def __getitem__(self, key):  
        'subclass dataframe getitem'
        if key in self._coilframe_attributes:
            self.update_dataframe()
        return DataFrame.__getitem__(self, key)
