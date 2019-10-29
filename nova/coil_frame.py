from nova.coil_data import CoilData
import pandas as pd
import numpy as np
from warnings import warn
import shapely.geometry
import shapely.affinity


class CoilSeries(pd.Series):

    @property
    def _constructor(self):
        return CoilSeries

    @property
    def _constructor_expanddim(self):
        return CoilFrame


class CoilFrame(pd.DataFrame):

    '''
    CoilFrame instance inherits from Pandas DataFrame
    Inspiration taken from GeoPandas https://github.com/geopandas
    '''

    _metadata = ['_required_columns', '_additional_columns',
                 '_default_attributes']

    def __init__(self, *args, **kwargs):
        self._initialize_instance_metadata()
        kwargs = self._update_metadata(**kwargs)
        pd.DataFrame.__init__(self, *args, **kwargs)

    def _initialize_instance_metadata(self):
        self._initialize_required_columns()
        self._initialize_additional_columns()
        self._initialize_default_attributes()

    def _initialize_required_columns(self):
        '''
        required input: self.add_coil(*args)
        '''
        self._required_columns = ['x', 'z', 'dx', 'dz']

    def _initialize_additional_columns(self):
        '''
        additional input: self.add_coil(**kwargs)
        '''
        self._additional_columns = []

    def _initialize_default_attributes(self):
        '''
        default attributes when not set via self.add_coil(**kwargs)
        '''
        self._default_attributes = \
            {'Ic': 0, 'It': 0, 'm': '', 'R': 0, 'Nt': 1, 'Nf': 1,
             'material': '', 'turn_fraction': 1, 'patch': None,
             'cross_section': 'square', 'turn_section': 'square',
             'coil': '', 'part': '', 'subindex': None, 'dCoil': 0,
             'mpc': '', 'polygon': None, 'control': True}

    def _update_metadata(self, **kwargs):
        mode = kwargs.pop('mode', 'append')  # [overwrite, append]
        for key in self._metadata:  # extract and update metadata from kwargs
            if mode == 'overwrite':
                null = {} if key == '_default_attributes' else []
                setattr(self, key, null)
            value = kwargs.pop(key[1:], None)
            if value:
                if key == '_additional_columns':
                    for v in value:  # insert additional columns
                        if v not in self._additional_columns:
                            self._additional_columns.append(v)
                elif key == '_default_attributes':
                    for k in value:  # set/overwrite default kwarg
                        self._default_attributes[k] = value[k]
                else:  # overwrite required columns
                    setattr(self, key, value)
        return kwargs

    @property
    def metadata(self):
        return dict((key[1:], getattr(self, key)) for key in self._metadata)

    @property
    def _constructor(self):
        return CoilFrame

    @property
    def _constructor_sliced(self):
        return CoilSeries

    @property
    def It(self):
        '''
        Returns:
            self['It'] (pd.Series): turn current [A.turns]
        '''
        self.update_frame()
        return self['It']

    @It.setter
    def It(self, It):
        self.data.It = It
        self.update_frame()

    @property
    def Ic(self):
        '''
        Returns:
            self['Ic'] (pd.Series): line current [A]
        '''
        self.update_frame()
        return self['Ic']

    @Ic.setter
    def Ic(self, Ic):
        self.data.Ic = Ic
        self.update_frame()

    def update_frame(self):
        self.data.update_frame()
        self['Ic'] = self.data.frame['Ic']  # line-current [A]
        self['It'] = self.data.frame['It']  # turn-current [A.turn]

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

    def get_frame(self, *args, **kwargs):
        args, kwargs = self._check_arguments(*args, **kwargs)
        delim = kwargs.pop('delim', '_')
        label = kwargs.pop('label', kwargs.get('name', 'Coil'))
        name = kwargs.pop('name', f'{label}{delim}{self.nC:d}')
        mpc = kwargs.pop('mpc', False)
        data = self._extract_data(*args, **kwargs)
        index = self._extract_index(data, delim, label, name)
        frame = CoilFrame(data, index=index, columns=data.keys(),
                          **self.metadata)
        frame = self._insert_polygon(frame)
        if mpc:
            frame.add_mpc(frame.index.to_list())
        return frame

    def add_coil(self, *args, iloc=None, **kwargs):
        frame = self.get_frame(*args, **kwargs)  # additional coils
        self.concatenate(frame, iloc=iloc)
        return frame.index

    def drop_coil(self, index=None):
        if index is None:
            index = self.index
        self = self.drop(index, inplace=False)
        self.data = CoilData(self)

    def concatenate(self, *frame, iloc=None, sort=False):
        if iloc is None:  # append
            frames = [self, *frame]
        else:  # insert
            frames = [self.iloc[:iloc, :], *frame, self.iloc[iloc:, :]]
        frame = pd.concat(frames, sort=sort)  # concatenate
        if sort:
            frame.sort_index(inplace=True)
        CoilFrame.__init__(self, frame, **self.metadata)  # relink new instance
        self.data = CoilData(self)

    def add_mpc(self, name, factor=1):
        '''
        define multi-point constraint linking a set of coils
        name: list of coil names (present in self.frame.index)
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
            self.at[n, 'mpc'] = (name[0], f)

    def _check_arguments(self, *args, **kwargs):
        if len(args) == 1:  # data passed as pandas dataframe
            data = args[0]
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
            if key in kwargs:
                data[key] = kwargs.pop(key)
                self._update_metadata(additional_columns=[key])
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
                           if pd.api.types.is_list_like(data[key])])
        except ValueError:
            nCol = 1  # scalar input
        if pd.api.types.is_list_like(name):
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

    def _insert_polygon(self, frame):
        if 'polygon' in frame.columns:
            for name in frame.index:
                if pd.isna(frame.at[name, 'polygon']):
                    x = frame.at[name, 'x']
                    z = frame.at[name, 'z']
                    dx = frame.at[name, 'dx']
                    dz = frame.at[name, 'dz']
                    cross_section = frame.at[name, 'cross_section']
                    if (np.array([x, dx, dz]) != 0).all():
                        polygen = self._get_polygen(cross_section)
                        polygon = polygen(x, z, dx, dz)
                    else:
                        polygon = None
                    frame.at[name, 'polygon'] = polygon
        return frame

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
