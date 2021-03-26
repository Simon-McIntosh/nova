# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:33:42 2021

#if isinstance(data, pandas.core.internals.managers.BlockManager):
#    return

@author: mcintos


"""



        '''
        patch = self.frame.loc[index, 'patch']
        # form list of lists
        patch = [_patch if pandas.api.types.is_list_like(_patch)
                 else [_patch] for _patch in patch]
        if len(patch) > 0:  # flatten and sort
            patch = functools.reduce(operator.concat, patch)
            patch = np.array(patch)[np.argsort([p.zorder for p in patch])]
            patch_collection = PatchCollection(patch, match_original=True)
            self.axes.add_collection(patch_collection, autolim=True)
            self.axes.autoscale_view()
        super().plot(axes, **kwargs)
        '''

    '''
    def patch(self):
        """Update frame patch, call on-demand."""
        patch = [[] for __ in range(len(self.frame))]
        for i, (current_patch, poly, part) in enumerate(
                self.frame.loc[:, ['patch', 'poly', 'part']].values):
            if self.overwrite or self.update_patch(current_patch):
                if isinstance(poly, dict):
                    poly = shapely.geometry.shape(poly)
                if isinstance(poly, shapely.geometry.Polygon):
                    patch[i] = [PolygonPatch(poly)]
                else:
                    patch[i] = []
            else:
                patch[i] = [current_patch]
            for j in range(len(patch[i])):
                patch[i][j].set_edgecolor(self.edgecolor)
                patch[i][j].set_linewidth(self.linewidth)
                patch[i][j].set_antialiased(True)
                patch[i][j].set_facecolor(self.get_color(part))
                patch[i][j].set_zorder = self.get_zorder(part)
                patch[i][j].set_alpha(self.get_alpha(part))
                if self.patchwork != 0:
                    self.shuffle(patch[i][j])
        self.frame.loc[:, 'patch'] = np.asarray(patch, object)
    '''

    '''
    def update_patch(self, patch=None):
        """Return True if any patches are null else False."""
        if patch is None:
            patch = self.frame.patch
        return np.array(pandas.isnull(patch)).any()
    '''

        # check unset defaults
        '''
        unset = np.array([attr not in self.default
                          for attr in self.additional])
        if unset.any():
            raise ValueError('default value not set for additional attributes '
                             f'{np.array(self.additional)[unset]}')
        '''

        #metatag = {tag: self.metaframe.default[tag]
        #           for tag in ['name', 'label', 'delim', 'offset']}
        #for tag in metatag:
        #    if tag in data.columns:
        #        metatag[tag] = data[tag]
        #        data.drop(columns=tag, inplace=True)

    #def __getattr__(self, col):
    #    """Extend DataFrame.__getattr__ to provide access to subspace."""
    #    if self.in_field(col, 'subspace'):
    #        return self.subspace.__getattr__(col)
    #    return super().__getattr__(col)

    def format_data(self, data):
        super().format_data(data)
        if isinstance(data, dict):
            for col in data:
                if self.in_array(col):
                    self._set_array(col, self._get_frame(col))


    def match_columns(self):
        """Extend DataFrame.match_columns, init data structure."""
        super().match_columns()
        if not self.empty:
            self.update_array()

    def update_array(self):
        """Set array data and backpropagate to frame if unset (default)."""
        for col in self.metaarray.array:
            self.attrs['metaarray'].data[col] = self._get_frame(col)

    def __getitem__(self, key):
        """Extend Loc getitem. Update frame prior to return if col in array."""
        print('getarray loc', key)
        col = self.obj.get_col(key)
        if self.obj.in_array(col):
            #index = self.obj.get_index(key)
            #if self.obj.metaframe.lock('array') is False:
            #    return self.obj.__getitem__(col)[index]
            #if self.obj.metaframe.lock('array') is True:
            #    print('updating frame')
            self.obj.set_frame(col)  # update frame
            with self.obj.metaframe.setlock(True, 'array'):
                return super().__getitem__(key)
        return super().__getitem__(key)


    '''

    '''

    def get_frame(self, col):
        """Return inflated subspace variable."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(True, 'subspace'):
            return super().__getitem__(col)


    def _hasattrs(self, attrs: list[str]):
        """Return True if all required attributes are available else False."""
        return np.array([attr in self.frame.columns for attr in attrs]).all()



        def _build_data(self, *args, **kwargs):
        """Extend DataFrame._build_data add line current converter."""
        attrs = self.metaframe.required + list(kwargs)  # record passed attrs
        data = super()._build_data(*args, **kwargs)
        if 'It' in attrs and 'Ic' not in attrs:  # patch line current
            data['Ic'] = \
                data['It'] / data.get('Nt', self.metaframe.default['Nt'])
        return data


    #def validate(self):
    #    """Extend MetaData.validate, set default update flags."""
    #    super().validate()
    #    self.update_flag('array', True)
    #    self.update_flag('frame', False)

    '''
    def update_flag(self, instance, default):
        """Set flag defaults for new attributes."""
        attribute = getattr(self, f'update_{instance}')
        attribute |= {attr: default for attr in self.array
                      if attr not in attribute}
        setattr(self, f'update_{instance}',
                {attr: attribute[attr] for attr in self.array})
    '''

#### subspace

    '''
    @property
    def line_current(self):
        """Manage line current."""
        return self.Ic

    @line_current.setter
    def line_current(self, line_current):
        super().__setattr__('Ic', line_current)
    '''

    '''
    if key in self._dataframe_attributes:
        self.refresh_frame(key)
        if key in ['Nt', 'It', 'Ic']:
            self._It = self.It
        if key == 'Nt':
            self.metaarray.update['Ic'] = True
            self.metaarray.update['It'] = True
        if key in ['Ic', 'It']:
            _key = next(k for k in ['Ic', 'It'] if k != key)
            self.metaarray.update[_key] = True
    '''

    '''

    def refresh_dataframe(self):
        """Transfer data from frame attributes to dataframe."""
        if self.update_dataframe:
            update = self.metaarray.update.copy()
            self.update_dataframe = False
            with self._write_dataframe():
                for attribute in update:
                    if update[attribute]:
                        if attribute in ['Ic', 'It']:
                            current = self._Ic[self._mpc_referance] * \
                                self._mpc_factor
                            if attribute == 'It':
                                current *= self._Nt
                            self.loc[:, attribute] = current
                            _attr = next(attr for attr in ['Ic', 'It']
                                         if attr != attribute)
                            self.metaarray.update[_attr] = False
                        else:
                            self.loc[:, attribute] = getattr(self, attribute)

    @contextmanager
    def _write_dataframe(self):
        """
        Apply a local attribute lock via the _update_frame flag.

        Prevent local attribute write via __setitem__ during dataframe update.

        Yields
        ------
        None
            with self._write_dataframe(self):.

        """
        self._update_frame = False
        yield
        self._update_frame = True

        if self._update_frame:  # protect against regressive update
            if key in ['Ic', 'It'] and self._mpc_iloc is not None:
                _current_update = self.current_update
                self.current_update = 'full'
                self._set_current(self.loc[self.index[self._mpc_iloc], key],
                                  current_column=key, update_dataframe=False)
                self.current_update = _current_update
            else:
                value = self.loc[:, key].to_numpy()
                if key in self._mpc_attributes:
                    value = value[self._mpc_iloc]
                setattr(self, f'_{key}', value)
            if key in self.metaarray.update:
                self.metaarray.update[key] = False

    '''

    '''
    @property
    def update(self):
        """
        Manage current_index and current_update flag.

        Set current_index based on current_flag. current_index used
        in coil current update.

        Parameters
        ----------
        update_flag : str
            Current update select flag. Full description given in
            :meth:`~coildata.CoilData.mpc_select`.

        Returns
        -------
        update_flag : str
            Current update select flag:

        """
        return self._current_update

    @update.setter
    def update(self, update_flag):
        self._current_update = update_flag
        self._current_index = self.mpc_select(update_flag)

    @property
    def index(self):
        """Return current index."""
        return self._current_index

    @property
    def current_status(self):
        """
        Display current index update status.

        - active
        - optimize
        - plasma
        - feedback
        - current update

        """
        if self.coil_number > 0:
            return DataFrame(
                    {'active': self._active,
                     'optimize': self._optimize,
                     'plasma': self._plasma,
                     'feedback': self._feedback,
                     self.current_update: self._current_index},
                    index=self._mpc_index)
        else:
            return DataFrame(columns=['active', 'optimize', 'plasma',
                                      'feedback', self.current_update])

    def get_current(self, current_column):
        """Return full mpc current vector."""
        current = self._Ic
        if current_column == 'It':  # convert to turn current
            current *= self._Nt[self._mpc_iloc]
        return current

    def _set_current(self, value, current_column='Ic', update_dataframe=True):
        """
        Update line-current in variable _Ic.

        Index built as union of value.index and coil.index.

        Parameters
        ----------
        value : dict or itterable
            Current update vector.
        current_column : str, optional
            Specify current_column. The default is 'Ic'.

            - 'Ic' == line current [A]
            - 'It' == turn current [A.turns]

        update_dataframe : bool, optional
            Update dataframe. The default is True.

        Returns
        -------
        None.

        """
        self._update_dataframe['Ic'] = update_dataframe  # update dataframe
        self._update_dataframe['It'] = update_dataframe
        nU = sum(self._current_index)  # length of update vector
        current = self.get_current(current_column)
        if is_dict_like(value):
            for i, (index, update) in enumerate(zip(self.index[self._mpc_iloc],
                                                    self._current_index)):
                if index in value and update:
                    current[i] = value[index]  # overwrite
        else:  # itterable
            if not is_list_like(value):
                value = value * np.ones(nU)
            if len(value) == nU:  # cross-check input length
                current[self._current_index] = value
            else:
                raise IndexError(
                        f'length of input {len(value)} does not match '
                        f'"{self.current_update}" coilset length'
                        f' {nU}\n'
                        'coilset.index: '
                        f'{self._mpc_index[self._current_index]}\n'
                        f'value: {value}\n\n'
                        f'{self.current_update}\n')
        if current_column == 'Ic':
            self._Ic = current
        elif current_column == 'It':
            self._Ic = current / self._Nt[self._mpc_iloc]
        else:
            raise AttributeError(f'current column {current_column} '
                                 'not in [Ic, It]')
        self._It = self._Ic * self._Nt[self._mpc_iloc]


    @property
    def Ic(self):
        """
        Coil instance line current [A].

        Returns
        -------
        Ic : np.array, shape(nI,)
            Line current array (current index).

        """
        #line_current = self._Ic[self._mpc_referance] * self._mpc_factor
        return self._Ic[self.current_index]

    @Ic.setter
    def Ic(self, value):
        self._set_current(value, 'Ic')

    @property
    def It(self):
        """
        Coil instance turn current [A.turns].

        Returns
        -------
        It : np.array, shape(nI,)
            Turn current array (current_index).

        """
        #turn_current = self._Ic[self._mpc_referance] * self._mpc_factor * \
        #    self._Nt
        return self.Ic * self.Nt[self._mpc_iloc][self.current_index]

    @It.setter
    def It(self, value):
        self._set_current(value, 'It')
    '''


#### subspace

    '''
    @property
    def metaarray(self):
        """
        Return metaarray instance, protect against pandas recursion loop.

        To understand recursion, you must understand recursion.
        """
        self.update_metaarray()
        return self.attrs['metaarray']

    def update_metaarray(self):
        """Update metaarray if not present in self.attrs."""
        if 'metaarray' not in self.attrs:
            self.attrs['metaarray'] = MetaArray()
    '''


#  data array

    '''
    def __getattr__(self, col):
        """Extend DataFrame.__getattr__. (frame.*)."""
        if col in self.metaarray.array:
            if self.metaarray.update_array[col]:
                self._update_array(col=col)
            return self.metaarray.data[col]
        return super().__getattr__(col)
    '''

        '''
    def __setattr__(self, col, value):
        """Extend DataFrame.__setattr__ (frame.* = *).."""
        if col in self.metaarray.array:
            self._update_array(col=col, value=value)
            self.metaarray.update_frame[col] = True
            return None
        return super().__setattr__(col, value)
    '''

   # def _set_value(self, index, col, value, takeable=False):
   #     """Extend DataFrame._set_value. (frame.at[i, '*'] = *)."""
   #     if col in self.metaarray.array:
   #         print('set value', col)
   #         self._update_array(index=index, col=col, value=value)
   #     return super()._set_value(index, col, value, takeable)

    #def _get_value(self, index, col, takeable=False):
    #    """Extend DataFrame._get_value. (frame.at[i, '*'])."""
    #    if col in self.metaarray.array:
    #        self._update_frame(col)
    #    return super()._get_value(index, col, takeable)



    #def _update_frame(self, col):
    #    """Copy col data to frame."""
    #    if self.metaarray.update_frame[col]:
    #        super().__setitem__(col, self[col])
    #        self.metaarray.update_frame[col] = False



# data array

    '''
    #@profile
    def _update_array(self, index=None, col=None, value=None):
        if index is None:
            index = slice(None)
        elif isinstance(index, str):
            _index = self.index.get_indexer(index)
            if _index == -1:
                raise IndexError(f'index {index} not found in {self.index}')
            index = _index
        if value is None:
            value = self._getcol(col)
        self._setarray(index, col, value)
    '''

    '''
    #@profile
    def _setarray(self, index, col, value):
        """Set value in metaarray.data."""
        if col not in self.metaarray.data or index == slice(None):
            self.metaarray.data[col] = value
        else:
            self.metaarray.data[col][index] = value
        if col not in self.columns:
            super().__setitem__(col, value)
        #self.metaarray.update_array[col] = False
    '''

# match columns
        metadata = {}
        #for attribute in ['required', 'additional']:
        #    metadata[attribute.capitalize()] = [
        #        attr for attr in getattr(self.metaframe, attribute)
        #        if attr in self.columns]

    def _validate_metadata(self):
        """Validate required and additional attributes in FrameArray."""
        # check for additional attributes in metaarray.array
        if 'metaarray' in self.attrs:
            unset = [attr not in self.metaframe.columns
                     for attr in self.metaarray.array]
            if np.array(unset).any():
                raise IndexError(
                    'attributes in metadata.array '
                    f'{np.array(self.metaarray.array)[unset]}'
                    ' not found in metaframe.required '
                    f'{self.metaframe.required}'
                    f'or metaframe.additional {self.metaframe.additional}')
        if not self.empty:
            self._format_columns()


"""Extend pandas.DataFrame to manage coil and subcoil data."""

from typing import Optional, Collection, Any, Union

import numpy as np

from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.frame import Frame
from nova.electromagnetic.subspace import SubSpace

# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object


class CoilFrame(Frame):
    """
    Extend SuperSpace.

    - Implement current properties.

    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, Union[MetaArray, MetaFrame]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, attrs, **metadata)
        self.attrs['subspace'] = SubSpace(self)


if __name__ == '__main__':

    coilframe = CoilFrame(Required=['x', 'z'], optimize=True,
                          dCoil=5, Additional=['Ic'], Nt=10)

    coilframe.add_frame(4, range(3), link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe.add_frame(4, range(4), link=True)

    def set_current():
        coilframe.Ic = np.random.rand(len(coilframe.subspace))

    set_current()

    coilframe.subspace.iloc[3, 0] = 33.4
    print(coilframe)

    #frame.x = [1, 2, 3]
    #frame.x[1] = 6

    #print(frame)

    #frame.metaarray._lock = False
    #newframe = Frame()


    '''
    @property
    def line_current(self):
        return self.loc[:, 'Ic']

    @line_current.setter
    def line_current(self, current):
        self.loc[:, 'Ic'] = current
    '''

    def _current_label(self, **kwargs):
        """Return current label, Ic or It."""
        current_label = None
        if 'Ic' in self.metaframe.required or 'Ic' in kwargs:
            current_label = 'Ic'
        elif 'It' in self.metaframe.required or 'It' in kwargs:
            current_label = 'It'
        return current_label

    @staticmethod
    def _propogate_current(current_label, data):
        """
        "Propogate current data, Ic->It or It->Ic.

        Parameters
        ----------
        current_label : str
            Current label, Ic or It.
        data : Union[pandas.DataFrame, dict]
            Current / turn data.

        Returns
        -------
        None.

        """
        if current_label == 'Ic':
            data['It'] = data['Ic'] * data['Nt']
        elif current_label == 'It':
            data['Ic'] = data['It'] / data['Nt']

    def __getattr__(self, col):
        """Extend DataFrame.__getattr__. (frame.*)."""
        print('getattr')
        if col in self.attrs:
            return self.attrs[col]
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__getattr__(col)
            if self.metaframe.lock('subspace') is False:
                self.set_frame(col)
        return super().__getattr__(col)

    def __setattr__(self, col, value):
        """Check lock. Extend DataFrame.__setattr__ (frame.* = *).."""
        value = self._format_value(col, value)
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__setattr__(col, value)
            if self.metaframe.lock('subspace') is False:
                raise SubSpaceError('setattr', col)
        if self.in_field(col, 'energize'):
            if self.metaframe.lock('energize') is False:
                print('setattr')
        return super().__setattr__(col, value)




#class _ScalarAccessIndexer(IndexerMixin,
#                           pandas.core.indexing._ScalarAccessIndexer):
#    pass


#class _LocationIndexer(IndexerMixin, pandas.core.indexing._LocationIndexer):
#    pass


##class _iLocIndexer(Indexer.Location(), pandas.core.indexing._iLocIndexer):
#    pass


#class _LocIndexer(Indexer.Location(), pandas.core.indexing._LocIndexer):
#    pass


#class _AtIndexer(Indexer.ScalarAccess(), pandas.core.indexing._AtIndexer):
#    pass


#class _iAtIndexer(Indexer.ScalarAccess(),
#                  pandas.core.indexing._iAtIndexer):
#    pass



    def _subspace(self, col: str):
        """Return True if col in metaframe.subspace else False."""
        if isinstance(col, str) and 'metaframe' in self.obj.attrs:
            if col in self.obj.metaframe.subspace:
                return True
        return False


    def _format_col(self, col: Union[int, str]) -> str:
        """Return column name."""
        if not isinstance(col, str):
            print(col)
            raise IndexError
            return self.columns[col]
        return col


    def _format_isubcol(self, col: int) -> int:
        """Return subcolumn index."""
        return self.subspace.columns.get_loc(self.columns[col])

    def _format_subindex(self, index: Union[int, str]) -> str:
        """Return subspace index label."""
        if not isinstance(index, str):
            return self.subspace.index[index]
        return index

    '''
    def _get_value(self, index, col, takeable=False):
        """Extend DataFrame._get_value. (frame.at[i, '*'])."""
        if self.insubspace(col):
            index = self._format_subindex(index)
            col = self._format_col(col)
            if self.metaframe.lock is True:
                if col in self.subspace:
                    return self.subspace._get_value(index, col, takeable)
            if self.metaframe.lock is False:
                self.set_frame(col)
        return super()._get_value(index, col, takeable)
    '''

    '''
    def _set_value(self, index, col, value, takeable=False):
        """Extend DataFrame._set_value. (frame.at[i, '*'] = *)."""
        value = self._format_value(col, value)
        if self.insubspace(col):
            index = self._format_subindex(index)
            col = self._format_col(col)
            if self.metaframe.lock is True:
                return self.subspace._set_value(
                    index, col, value, takeable=False)
            if self.metaframe.lock is False:
                raise SubSpaceIndexError(col)
        return super()._set_value(index, col, value, takeable)
    '''

    def _iset_item(self, loc: int, value):
        col = self.columns[loc]
        value = self._format_value(col, value)
        if self.insubspace(col):
            print('iset', col)
            try:
                value = value[self.subspace.index]
            except TypeError:
                pass
            if self.metaframe.lock is True:
                self.set_frame(col)
                print('subvalue', value)
                return self.subspace._set_item(col, value)
            if self.metaframe.lock is False:
                raise SubSpaceIndexError(col)
        return super()._iset_item(loc, value)


    """
    Key Attributes
    --------------
    Ic : float, array-like
        Coil line current [A]
    It : float, array-like
        Coil turn curent [A.turns]
    Nt : float, array-like
        Coil turn number.
    active : bool, array-like
        Coil current control status.
    optimize : bool, array-like
        Optimization flag.
    plasma : bool, array-like
        Plasma flag.
    feedback : bool, array-like
        Feedback stabilization flag
    """

    '''
        def add_column(self, label):
        """Add column to Frame initializing values to default."""
        if label not in self.metaframe.columns:
            self.metadata = {'additional': [label]}
            if len(self) > 0:  # initialize with default value
                print(label, self.metaframe.default[label])
                self[label] = self.metaframe.default[label]
    '''

    '''
        def update_columns(self):
        """Intersect of self.columns and self.metaframe.columns."""
        if not self.columns.empty:
            metadata = {}
            metadata['Required'] = [attr for attr in self.columns
                                    if attr in self.metaframe.required]
            metadata['Additional'] = [attr for attr in self.columns
                                      if attr not in self.metaframe.required]
            if self.metaframe.required != metadata['Required'] or \
                    self.metaframe.additional != metadata['Additional']:
                self.metaframe.metadata = metadata  # perform intersection

    '''

    '''
    def __init__(self):
        """Build fast access data."""
        #
        #for attribute in self.metaarray.array:
        #    self.data[attribute] = self[attribute].to_numpy()
        # extract properties
        #self.validate_array()
        self.metaarray.properties = [p for p, __ in inspect.getmembers(
            Array, lambda o: isinstance(o, property))]
    '''

    '''
        def reduce_multipoint(self, matrix):
        """Apply multipoint constraints to coupling matrix."""
        _matrix = matrix[:, self._mpc_iloc]  # extract primary coils
        if len(self._mpl_index) > 0:  # add multi-point links
            _matrix[:, self._mpl_index[:, 0]] += \
                matrix[:, self._mpl_index[:, 1]] * \
                np.ones((len(matrix), 1)) @ self._mpl_factor.reshape(-1, 1)
        return _matrix
    '''

name change:
    _dataframe_attributes -> metaarray.data, dict
    _coildata_attributes -> metaarray.frame
    _update_dataframe -> metaarray.update
    _coildata_properties -> metaarray.properties

    frame: dict[str, str] = field(
        repr=False, default_factory=lambda: {'current_update': 'full'})




    # current update attributes
    _coilcurrent_attributes = []

    # CoilData indices
    _coildata_indices = ['reduction_index',
                         'plasma_reduction_index',
                         'plasma_iloc',
                         'ionize_index',
                         'current_index']

    # compact mpc attributes - subset of coilframe and coildata attributes
    _mpc_attributes = ['Ic', 'active', 'plasma', 'optimize', 'feedback',
                       'current_index']

    # multi-point constraints (shared line-current)
    _mpc_constraints = ['mpc_index', 'mpc_iloc', 'mpc_referance',
                        'mpc_factor', 'mpl_index', 'mpl_factor']

    # class properties (inspect.getmembers) exclude from setattr
    _coildata_properties = []

    # update flags
    _coildata_flags = {'update_dataframe': False,
                       'update_coilframe': True,
                       'update_biotsavart': True,
                       'current_update': 'full',
                       'relink_mpc': True}

    def __init__(self):
        self._initialize_coildata_flags()
        self._initialize_coildata_attributes()
        self._initialize_dataframe_attributes()
        self._initialize_coilcurrent_attributes()
        self._unlink_coildata_attributes()



    def _initialize_coildata_flags(self):
        for flag in self._coildata_flags:  # update read/write
            setattr(self, f'_{flag}', None)  # unlink from DataFrame
            setattr(self, f'_{flag}', self._coildata_flags[flag])
        self.update_dataframe = False

    def _initialize_dataframe_attributes(self):
        self._dataframe_attributes = self._mpc_attributes.copy()

    def _initialize_coildata_attributes(self):
        self._coildata_attributes = {}

    def _initialize_coilcurrent_attributes(self):
        self._coilcurrent_attributes = [attribute for attribute in
                                        self._mpc_attributes if attribute
                                        not in ['Ic', 'current_index']]

    def _unlink_coildata_attributes(self):
        # list attributes
        for attribute in self._dataframe_attributes +\
                         self._coildata_indices + \
                         self._mpc_attributes + \
                         self._mpc_constraints:
            setattr(self, f'_{attribute}', None)
        # dict attributes
        for attribute in self._coildata_attributes:
            setattr(self, f'_{attribute}', None)
        self.coildata_attributes = self._coildata_attributes

    @property
    def dataframe_attributes(self):
        """Return list of fast access dataframe attributes."""
        return self._dataframe_attributes

    @dataframe_attributes.setter
    def dataframe_attributes(self, dataframe_attributes):
        """Append coilframe attributes (fast access)."""
        for attribute in dataframe_attributes:
            if attribute not in self._dataframe_attributes:
                setattr(self, f'_{attribute}', None)
                self._dataframe_attributes.append(attribute)
                if attribute in self.columns:
                    self.refresh_coilframe(attribute)

    @property
    def coildata_attributes(self):
        """Extract coildata attributes."""
        self._coildata_attributes = {
                attribute: getattr(self, f'_{attribute}')
                for attribute in self._coildata_attributes}
        return self._coildata_attributes

    @coildata_attributes.setter
    def coildata_attributes(self, coildata_attributes):
        'set coildata attributes'
        update = {attribute: coildata_attributes[attribute]
                  for attribute in coildata_attributes
                  if attribute not in self._coildata_attributes}
        if len(update) > 0:
            self._coildata_attributes.update(update)
            for attribute in update:
                setattr(self, f'_{attribute}', None)
        for attribute in coildata_attributes:
            setattr(self, f'_{attribute}', coildata_attributes[attribute])

    @property
    def update_dataframe(self):
        return np.fromiter(self._update_dataframe.values(), dtype=bool).any()

    @update_dataframe.setter
    def update_dataframe(self, value):
        if type(value) == bool:
            self._update_dataframe = {
                attribute: value for attribute in self._dataframe_attributes}
        elif isinstance(value, dict):
            self._update_dataframe.update(value)
        else:
            self._update_dataframe.update({
                attribute: True for attribute in value})

    def _update_flags(self, **kwargs):
        for flag in self._coildata_flags:
            if flag in kwargs:
                setattr(self, f'_{flag}', kwargs[flag])

    def rebuild_coildata(self):
        if self.coil_number > 0:
            self._extract_mpc()  # extract multi-point constraints
            self._extract_data_attributes()  # extract from DataFrame columns
            self._extract_reduction_index()
            self.current_update = self._current_update  # set flag
            self.refresh_dataframe()  # transfer from coilframe to dataframe


    def _extract_data_attributes(self):
        self.update_dataframe = False
        for attribute in self._dataframe_attributes + self._coildata_indices:
            if attribute in ['active', 'plasma', 'optimize', 'feedback']:
                dtype = bool
            else:
                dtype = float
            if attribute in self:  # read from DataFrame column
                value = self[attribute].to_numpy(dtype=dtype)
            elif attribute in self._default_attributes:  # default
                value = np.array([self._default_attributes[attribute]
                                  for __ in range(self.coil_number)],
                                 dtype=dtype)
            else:
                value = np.zeros(self.coil_number, dtype=dtype)
            if attribute in self._mpc_attributes:  # mpc compaction
                value = value[self._mpc_iloc]
            setattr(self, f'_{attribute}', value)
        #self._ionize_index = self._plasma[self._mpc_referance]


