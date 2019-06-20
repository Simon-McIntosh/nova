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
    Inspiration taken from GeoPandas https://github.com/geopandas
    CoilFrame instance inherits from Pandas DataFrame
    '''
    _metadata = ['_required_columns', '_additional_columns',
                 '_default_attributes', '_integer_columns']

    def __init__(self, *args, **kwargs):
        self._initalize_instance_metadata()
        kwargs = self._update_metadata(**kwargs)
        super().__init__(*args, **kwargs)

    def _initalize_instance_metadata(self):
        # required input *args
        self._required_columns = ['x', 'z', 'dx', 'dz']
        # additional input via **kwargs
        self._additional_columns = ['cross_section', 'patch', 'part']
        self._default_attributes = {'Ic': 0, 'It': 0,
                                    'm': None, 'R': 0,
                                    'Nt': 1, 'Nf': 1, 'material': None,
                                    'cross_section': 'square',
                                    'patch': None, 'coil': '', 'part': '',
                                    'subindex': None,
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
        return frame

    @staticmethod
    def patch_coil(frame, color_label='part', **kwargs):  # call on-demand
        part_color = {'VS3': 'C0', 'VS3j': 'gray', 'CS': 'C0', 'PF': 'C0',
                      'trs': 'C2', 'vvin': 'C3', 'vvout': 'C4', 'plasma': 'C4'}
        cluster_color = dict([(i, f'C{i%10}') for i in range(frame.nC)])
        if color_label == 'part':
            color = kwargs.get('part_color', part_color)
        elif color_label == 'cluster_index':
            color = kwargs.get('cluster_color', cluster_color)
        else:
            raise IndexError(f'color_label: {color_label} '
                             'not in [part, cluster_index]')
        zorder = kwargs.get('zorder', {'VS3': 1, 'VS3j': 0})
        patch = [[] for __ in range(frame.nC)]
        for i, geom in enumerate(
                frame.loc[:, ['x', 'z', 'dx', 'dz', 'cross_section',
                              color_label]].values):
            x, z, dx, dz, cross_section, color_key = geom
            if cross_section in ['square', 'rectangle']:
                patch[i] = patches.Rectangle((x - dx/2, z - dz / 2), dx, dz)
            else:
                patch[i] = patches.Circle((x, z), (dx + dz) / 4)
            patch[i].set_edgecolor('darkgrey')
            patch[i].set_linewidth(0.25)
            patch[i].set_antialiased(True)
            patch[i].set_facecolor(color.get(color_key, 'C9'))
            patch[i].zorder = zorder.get(color_key, 0)
        frame.loc[:, 'patch'] = patch

    def add_coil(self, *args, **kwargs):
        frame = self.get_frame(*args, **kwargs)  # additional coils
        self.concatenate(frame)
        return frame.index

    def drop_coil(self, index=None):
        if index is None:
            index = self.index
        self.drop(index, inplace=True)

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

    def set_dtype(self, frame):
        for key in self._integer_columns:
            if key in frame.columns:
                frame.loc[:, key] = frame.loc[:, key].astype(int)


class CoilSet:
    '''
    instance wrapper for coilset data

    Attributes:

        coil (pd.DataFrame): coil data
        subcoil (pd.DataFrame): subcoil data

        inductance (dict): dictionary of inductance interaction matirces
            inductance['Mc'] (pd.DataFrame): line-current inductance matrix
            inductance['Mt'] (pd.DataFrame): amp-turn inductance matrix

        force (dict): coil force interaction matrices (pd.DataFrame)
            force['Fx']:  net radial force
            force['Fz']:  net vertical force
            force['xFx']: first radial moment of radial force
            force['xFz']: first radial moment of vertical force
            force['zFx']: first vertical moment of radial force
            force['zFz']: first vertical moment of vertical force
            force['My']:  net torque}
        subforce (dict): filament force interaction matrices (pd.DataFrame)
            subforce = {Fx, Fz, xFx, xFz, zFx, zFz, My}

        grid (dict): poloidal grid coordinates and interaction matrices
            grid['n'] ([2int]): grid dimensions
            grid['limit'] ([4float]): grid limits
            grid['x2d'] (np.array): x-coordinates (radial)
            grid['z2d'] (np.array): z-coordinates
            grid['Psi'] (pd.DataFrame): poloidal flux interaction matrix
            grid['psi'] (pd.DataFrame): poloidal flux
            grid['Bx'] (pd.DataFrame): radial field interaction matrix
            grid['Bz'] (pd.DataFrame): vertical field interaction matrix
            grid['bx'] (pd.DataFrame): radial field
            grid['bz'] (pd.DataFrame): vertical field
    '''

    def __init__(self, coil, subcoil, inductance=None, force=None,
                 subforce=None, grid=None):
        self.coil = coil
        self.subcoil = subcoil
        self.inductance = self.initalize_inductance(inductance)
        self.force = self.initialize_force(force)
        self.subforce = self.initialize_force(subforce)
        self.grid = self.initialize_grid(grid)

    @staticmethod
    def initalize_inductance(inductance=None):
        '''
        inductance interaction matrix, H
        '''
        if inductance is None:
            inductance = {'Mc': pd.DataFrame(),  # line-current
                          'Mt': pd.DataFrame()}  # amp-turn
        return inductance

    @staticmethod
    def initialize_force(force=None):
        '''
        force: a dictionary of force interaction matrices stored as dataframes
        '''
        if force is None:
            force = {
                    'Fx': None,  # radial force
                    'Fz': None,  # vertical force
                    'xFx': None,  # first radial moment of radial force
                    'xFz': None,  # first radial moment of vertical force
                    'zFx': None,  # first vertical moment of radial force
                    'zFz': None,  # first vertical moment of vertical force
                    'My': None}  # in-plane torque
        return force

    @staticmethod
    def initialize_grid(grid=None):
        if grid is None:
            grid = {'n': None,  # grid dimensions
                    'limit': None,  # grid limits
                    'x2d': None,  # x-coordinate
                    'z2d': None,  # z-coordinate
                    'Psi': None,  # flux interaction matrix
                    'psi': None,  # poloidal flux
                    'Bx': None,  # radial field interaction matrix
                    'Bz': None,  # radial field interaction matrix
                    'bx': None,  # radial field
                    'bz': None}  # vertical field
        return grid


if __name__ is '__main__':
    print('\nusage examples given in nova.coilclass')
