import pandas as pd
from matplotlib import patches


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
        kwargs = self.update_metadata(**kwargs)
        super().__init__(*args, **kwargs)

    def _initalize_instance_metadata(self):
        # current column label used in current calculation
        self._current_column = ''
        # required input *args
        self._required_columns = ['x', 'z', 'dx', 'dz']
        # additional input via **kwargs
        self._additional_columns = ['cross_section', 'patch', 'label']
        self._default_attributes = {'Ic': 0, 'It': 0, 'm': None, 'R': 0,
                                    'Nt': 1, 'Nf': 1, 'material': None,
                                    'cross_section': 'square',
                                    'patch': None, 'label': None,
                                    'dCoil': 0}
        self._integer_columns = ['Nf']

    def update_metadata(self, **kwargs):
        for key in self._metadata:  # extract and update metadata from kwargs
            value = kwargs.pop(key[1:], None)
            if value:
                if key == '_additional_columns':
                    for v in value:
                        if v not in self._additional_columns:
                            self._additional_columns.append(v)
                elif key == '_default_attributes':
                    for k in value:  # set/overwrite default kwarg
                        self._default_attributes[k] = value[k]
                else:  # overwrite
                    setattr(self, key, value)
        return kwargs

    @property
    def _constructor(self):
        return CoilFrame

    @property
    def _constructor_sliced(self):
        return CoilSeries

    def _get_current(self):
        if self._current_column not in self:
            raise AttributeError(f'\ncurrent_column: {self._current_column}\n'
                                 'not present in CoilFrame.columns:\n'
                                 f'{self.columns}')
        return self[self._current_column]

    def _set_current(self, col):
        self._current_column = col

    current = property(fget=_get_current, fset=_set_current,
                       doc='current data column name within CoilFrame')

    def _get_coil_number(self):
        return len(self.index)

    nC = property(fget=_get_coil_number, doc='number of coils in dataframe')

    def _get_column_number(self):
        return len(self.columns)

    nCol = property(fget=_get_column_number, doc='number of columns '
                    'in dataframe')

    def add_coil(self, *args, **kwargs):
        kwargs = self._check_input(*args, **kwargs)
        name = kwargs.pop('name', f'Coil_{self.nC:d}')
        for key in kwargs:
            self.at[name, key] = kwargs[key]
        self.set_dtype()  # insure dtype for labelled columns
        self._set_patch(name)
        return name

    def _set_patch(self, name):
        cross_section = self.at[name, 'cross_section']
        x, z, dx, dz = self.loc[name, ['x', 'z', 'dx', 'dz']]
        if cross_section in ['square', 'rectangle']:
            patch = patches.Rectangle((x - dx/2, z - dz / 2), dx, dz)
        else:
            patch = patches.Circle((x, z), (dx + dz) / 4)
        self.at[name, 'patch'] = patch

    def _check_input(self, *args, **kwargs):
        if len(self._required_columns) != len(args):
            raise IndexError(f'\nincorrect argument number: {len(args)}\n'
                             f'input *args as {self._required_columns} '
                             '\nor set _default_columns=[*] in kwarg')
        for key, arg in zip(self._required_columns, args):
            kwargs[key] = arg  # append required values from *args
        for key in self._additional_columns:
            if key not in kwargs:  # append defaults if not present in kwargs
                try:
                    kwargs[key] = self._default_attributes[key]
                except KeyError:
                    raise KeyError(f'{key} not present in default_attributes:'
                                   f' {self._default_attributes.keys()}')
        return kwargs

    def set_dtype(self):
        for key in self._integer_columns:
            if key in self.columns:
                self.loc[:, key] = self.loc[:, key].astype(int)


class CoilSet:
    '''
    Attributes:
        coil: a Pandas DataFrame containing all coil data
        subcoil: a Pandas DataFrame containig all subcoil data
        plasma: a Pandas DataFrame containing plasma filaments
    '''

    def __init__(self, dCoil=0):
        self.coil = CoilFrame(current_column='It',
                              additional_columns=['Ic', 'It', 'Nt', 'Nf',
                                                  'dCoil'],
                              default_attributes={'dCoil': dCoil})
        self.subcoil = CoilFrame(current_column='If')
        self.plasma = CoilFrame(current_column='If')


if __name__ is '__main__':
    print('\nusage examples in nova.coilclass')
