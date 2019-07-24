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
    Inspiration taken from GeoPandas https://github.com/geopandas
    CoilFrame instance inherits from Pandas DataFrame
    '''
    _metadata = ['_required_columns', '_additional_columns',
                 '_default_attributes']

    def __init__(self, *args, **kwargs):
        self._initalize_instance_metadata()
        kwargs = self._update_metadata(**kwargs)
        pd.DataFrame.__init__(self, *args, **kwargs)

    def _initalize_instance_metadata(self):
        self._initalize_required_columns()
        self._initalize_additional_columns()
        self._initalize_default_attributes()

    def _initalize_required_columns(self):
        '''
        required input: self.add_coil(*args)
        '''
        self._required_columns = ['x', 'z', 'dx', 'dz']

    def _initalize_additional_columns(self):
        '''
        additional input: self.add_coil(**kwargs)
        '''
        self._additional_columns = []

    def _initalize_default_attributes(self):
        '''
        default attributes when not set via self.add_coil(**kwargs)
        '''
        self._default_attributes = \
            {'Ic': 0, 'It': 0, 'm': None, 'R': 0, 'Nt': 1, 'Nf': 1,
             'material': None, 'turn_fraction': 1, 'patch': None,
             'cross_section': 'square', 'turn_section': 'square',
             'coil': '', 'part': '', 'subindex': None, 'dCoil': 0,
             'mpc': None, 'polygon': None}

    def _update_metadata(self, **kwargs):
        mode = kwargs.pop('mode', 'append')  # [empty, reset, append]
        for key in self._metadata:  # extract and update metadata from kwargs
            if mode == 'empty':
                empty = {} if key == '_default_attributes' else []
                setattr(self, key, empty)
            elif mode == 'reset':
                getattr(self, f'_initalise{mode}')()
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
        return self['It']

    @It.setter
    def It(self, It):
        idx = It.index
        self.loc[idx, 'It'] = It  # turn-current [A.turn]
        # line-current [A]
        self.loc[idx, 'Ic'] = self.loc[idx, 'It'] / self.loc[idx, 'Nt']
        self.copy_mpc()  # copy multi-point constraints

    @property
    def Ic(self):
        '''
        Returns:
            self['Ic'] (pd.Series): line current [A]
        '''
        return self['Ic']

    @Ic.setter
    def Ic(self, Ic):
        idx = Ic.index
        self.loc[idx, 'Ic'] = Ic  # line-current [A]
        # turn-current [A.turn]
        self.loc[idx, 'It'] = self.loc[idx, 'Ic'] * self.loc[idx, 'Nt']
        self.copy_mpc()  # copy multi-point constraints

    def copy_mpc(self):
        if 'mpc' in self.columns:
            for name in self.mpc.dropna().index:
                mpc = self.mpc[name]
                if name != mpc[0]:
                    self.at[name, 'Ic'] = self.at[mpc[0], 'Ic'] * mpc[1]
                    self.at[name, 'It'] = self.at[name, 'Ic'] *\
                        self.at[name, 'Nt']

    def _get_coil_number(self):
        return len(self.index)

    nC = property(fget=_get_coil_number, doc='number of coils in dataframe')

    def _get_column_number(self):
        return len(self.columns)

    nCol = property(fget=_get_column_number, doc='number of columns '
                    'in dataframe')

    def get_frame(self, *args, **kwargs):
        args, kwargs = self._check_arguments(*args, **kwargs)
        delim = kwargs.pop('delim', '_')
        label = kwargs.pop('label', kwargs.get('name', 'Coil'))
        name = kwargs.pop('name', f'{label}{delim}{self.nC:d}')
        data = self._extract_data(*args, **kwargs)
        index = self._extract_index(data, delim, label, name)
        frame = CoilFrame(data, index=index, columns=data.keys(),
                          **self.metadata)
        frame = self._insert_polygon(frame)
        return frame

    def add_coil(self, *args, iloc=None, **kwargs):
        frame = self.get_frame(*args, **kwargs)  # additional coils
        self.concatenate(frame, iloc=iloc)
        return frame.index

    def drop_coil(self, index=None):
        if index is None:
            index = self.index
        self.drop(index, inplace=True)

    def concatenate(self, *frame, iloc=None):
        if iloc is None:  # append
            frame = [self, *frame]
        else:  # insert
            frame = [self.iloc[:iloc, :], *frame, self.iloc[iloc:, :]]
        coil = pd.concat(frame, sort=False)  # concatenate
        CoilFrame.__init__(self, coil, **self.metadata)  # relink new instance

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
        for key in self._additional_columns:
            if key in kwargs:
                data[key] = kwargs.pop(key)
            else:
                data[key] = self._default_attributes[key]
        for key in self._default_attributes:
            if key in kwargs:
                data[key] = kwargs.pop(key)
                self._update_metadata(additional_columns=[key])
        if len(kwargs.keys()) > 0:
            warn(f'\n\nunset kwargs: {list(kwargs.keys())}'
                 '\nto use include within additional_columns:\n'
                 f'{self._additional_columns}'
                 '\nor within default_attributes:\n'
                 f'{self._default_attributes}\n')
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
                    x, z, dx, dz, cross_section = \
                        frame.loc[name, ['x', 'z', 'dx', 'dz',
                                         'cross_section']]
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
