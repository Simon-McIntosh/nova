import pandas as pd
import numpy as np
from matplotlib import patches
from warnings import warn


class CoilSeries(pd.Series):

    @property
    def _constructor(self):
        return CoilSeries

    @property
    def _constructor_expanddim(self):
        return CoilFrame


class CoilFrame(pd.DataFrame):
    '''
    CoilFrame instance inheritance from Pandas DataFrame
    Inspiration taken from GeoPandas https://github.com/geopandas
    '''
    _metadata = ['_current_column', '_required_columns',
                 '_additional_columns', '_default_attributes',
                 '_integer_columns']

    def __init__(self, *args, **kwargs):
        self._initalize_instance_metadata()
        kwargs = self._update_metadata(**kwargs)
        super().__init__(*args, **kwargs)

    def _initalize_instance_metadata(self):
        # current column label used in current calculation
        self._current_column = ''
        # required input *args
        self._required_columns = ['x', 'z', 'dx', 'dz']
        # additional input via **kwargs
        self._additional_columns = ['cross_section', 'patch', 'part']
        self._default_attributes = {'Ic': 0, 'It': 0, 'If': 0,
                                    'm': None, 'R': 0,
                                    'Nt': 1, 'Nf': 1, 'material': None,
                                    'cross_section': 'square',
                                    'patch': None, 'coil': '', 'part': '',
                                    'subcoil_index': None,
                                    'dCoil': 0, 'mpc': None}
        self._integer_columns = ['Nf']

    def _update_metadata(self, **kwargs):
        for key in self._metadata:  # extract and update metadata from kwargs
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
    def current(self):
        if self._current_column not in self:
            raise AttributeError(f'\ncurrent_column: {self._current_column}\n'
                                 'not present in CoilFrame.columns:\n'
                                 f'{self.columns}')
        return self[self._current_column]

    @current.setter
    def current(self, column):
        self._current_column = column

    def _get_coil_number(self):
        return len(self.index)

    nC = property(fget=_get_coil_number, doc='number of coils in dataframe')

    def _get_column_number(self):
        return len(self.columns)

    nCol = property(fget=_get_column_number, doc='number of columns '
                    'in dataframe')

    def get_frame(self, *args, **kwargs):
        args, kwargs = self._check_arguments(*args, **kwargs)
        name = kwargs.pop('name', f'Coil_{self.nC:d}')
        delim = kwargs.pop('delim', '_')
        data = self._extract_data(*args, **kwargs)
        index = self._extract_index(data, name, delim)
        frame = CoilFrame(data, index=index, columns=data.keys(),
                          **self.metadata)
        self.patch_coil(frame)
        return frame

    @staticmethod
    def patch_coil(frame, **kwargs):
        color = kwargs.get('color', {'VS3': 'C0', 'VS3j': 'gray',
                                     'CS': 'C0', 'PF': 'C0',
                                     'trs': 'C2', 'vvin': 'C3', 'vvout': 'C4',
                                     'plasma': 'C4'})
        zorder = kwargs.get('zorder', {'VS3': 1, 'VS3j': 0})
        patch = [[] for __ in range(frame.nC)]
        for i, geom in enumerate(
                frame.loc[:, ['x', 'z', 'dx', 'dz', 'cross_section',
                              'part']].values):
            x, z, dx, dz, cross_section, part = geom
            if cross_section in ['square', 'rectangle']:
                patch[i] = patches.Rectangle((x - dx/2, z - dz / 2), dx, dz)
            else:
                patch[i] = patches.Circle((x, z), (dx + dz) / 4)
            patch[i].set_edgecolor('darkgrey')
            patch[i].set_linewidth(0.25)
            patch[i].set_antialiased(True)
            patch[i].set_facecolor(color.get(part, 'C9'))
            patch[i].zorder = zorder.get(part, 0)
        frame.loc[:, 'patch'] = patch

    def add_coil(self, *args, **kwargs):
        frame = self.get_frame(*args, **kwargs)  # additional coils
        self.concatenate(frame)
        return frame.index

    def concatenate(self, *frame):
        coil = pd.concat([self, *frame], sort=False)  # concatenate
        CoilFrame.__init__(self, coil, **self.metadata)  # relink new instance

    def _check_arguments(self, *args, **kwargs):
        if len(args) == 1:  # data passed as pandas dataframe
            data = args[0]
            args = [data.loc[:, col] for col in self._required_columns]
            kwargs['name'] = data.index
            for col in data.columns:
                if col not in self._required_columns:
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
            data[key] = kwargs.pop(key, self._default_attributes[key])
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

    def _extract_index(self, data, name, delim):
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
                index = [f'{name}{delim}{i}' for i in range(nCol)]
        self._check_index(index)
        return index

    def _check_index(self, index):
        for name in index:
            if name in self.index:
                raise IndexError(f'\ncoil: {name} already defined in index\n'
                                 f'index: {self.index}')

    def set_dtype(self, frame):
        for key in self._integer_columns:
            if key in frame.columns:
                frame.loc[:, key] = frame.loc[:, key].astype(int)


class CoilSet:
    '''
    instance wrapper for coilset data

    Attributes:

        coil: a Pandas DataFrame containing all coil data
        subcoil: a Pandas DataFrame containig all subcoil data
        plasma: a Pandas DataFrame containing plasma filaments

        inductance = {coil: M_coil, coil_o: Mo_coil, subcoil: M_subcoil}

        matrix: a dictionary of force interaction matrices stored as dataframes
        matrix = {inductance: {coil: M, coil_o: M, subcoil: M}

                 {force: {coil: {Fx, Fz, xFx, xFz, zFx, zFz, My},
                          subcoil: {Fx, Fz, xFx, xFz, zFx, zFz, My}}
                 passive: {coil: {Fx, Fz},
                           subcoil: {Fx, Fz}}}
    '''
    def __init__(self, coil, subcoil, plasma, inductance=None):
        self.coil = coil
        self.subcoil = subcoil
        self.plasma = plasma
        self.inductance = inductance


if __name__ is '__main__':
    print('\nusage examples given in nova.coilclass')
