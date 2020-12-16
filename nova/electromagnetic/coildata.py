from contextlib import contextmanager
import inspect

import numpy as np
from pandas import DataFrame, Index
from pandas.api.types import is_list_like, is_dict_like


class CoilData():
    """
    Methods enabling fast access to dynamic coil and subcoil data.

    Provided as parent to CoilFrame. Inherited alongside DataFrame.
    Fast access variables stored as np.arrays _*
    Lazy data exchange implemented with parent DataFrame.

    Key Attributes
    --------------
    Ic : float, array-like
        Coil line current [A]
    It : float, array-like
        Coil turn curent [A.turns]
    Nt : float, array-like
        Coil turn number.
    power : bool, array-like
        Coil power supply status.
    optimize : bool, array-like
        Optimization flag.
    plasma : bool, array-like
        Plasma flag.
    feedback : bool, array-like
        Feedback stabilization flag

    """

    # list of fast access np.array variables linked to CataFrame
    _dataframe_attributes = []

    # metadata attributes
    _coildata_attributes = {}

    # current update attributes
    _coilcurrent_attributes = []

    # CoilData indices
    _coildata_indices = ['reduction_index',
                         'plasma_reduction_index',
                         'plasma_iloc',
                         'ionize_index',
                         'current_index']

    # compact mpc attributes - subset of coilframe and coildata attributes
    _mpc_attributes = ['Ic', 'power', 'plasma', 'optimize', 'feedback',
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
        self._extract_coildata_properties()
        self._initialize_coildata_flags()
        self._initialize_coildata_attributes()
        self._initialize_dataframe_attributes()
        self._initialize_coilcurrent_attributes()
        self._unlink_coildata_attributes()

    def _extract_coildata_properties(self):
        self._coildata_properties = [p for p, __ in inspect.getmembers(
            CoilData, lambda o: isinstance(o, property))]

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
        if self.nC > 0:
            self._extract_mpc()  # extract multi-point constraints
            self._extract_data_attributes()  # extract from DataFrame columns
            self._extract_reduction_index()
            self.current_update = self._current_update  # set flag
            self.refresh_dataframe()  # transfer from coilframe to dataframe

    def _extract_mpc(self):
        """Extract mpc interger index and factor."""
        mpc = self.get('mpc', np.array([self._default_attributes['mpc']
                                        for __ in range(self.nC)]))
        self._mpc_iloc = [i for i, _mpc in enumerate(mpc) if not _mpc]
        self._mpc_index = self.index[self._mpc_iloc]
        self._mpc_referance = np.zeros(self.nC, dtype=int)
        self._mpc_factor = np.ones(self.nC, dtype=float)
        _mpc_list = list(self._mpc_index)
        _mpc_array = np.arange(len(_mpc_list))
        mpc_index = mpc != self._default_attributes['mpc']
        self._mpc_referance[~mpc_index] = _mpc_array
        if sum(mpc_index) > 0:
            _mpc = np.array([[name, factor]
                             for name, factor in mpc[mpc_index].values],
                            dtype=object)
            _mpc_name = [_mpc[i, 0] for i in
                         sorted(np.unique(_mpc[:, 0], return_index=True)[1])]
            _mpc_dict = {name: index for name, index in
                         zip(_mpc_name,
                             _mpc_array[np.isin(_mpc_list, _mpc_name)])}
            self._mpc_referance[mpc_index] = [_mpc_dict[name]
                                              for name in _mpc[:, 0]]

            self._mpc_factor[mpc_index] = _mpc[:, 1]
        # link subcoil to coil referance
        if 'coil' in self:
            self._mpc_index = Index(self.loc[self._mpc_index, 'coil'])
        # construct multi-point link ()
        mpl = np.array([
            [referance, couple, factor] for couple, (referance, _mpc, factor)
            in enumerate(zip(self._mpc_referance, mpc, self._mpc_factor))
            if _mpc])
        if len(mpl) > 0:
            self._mpl_index = mpl[:, :2].astype(int)  # (refernace, couple)
            self._mpl_factor = mpl[:, 2]  # coupling factor
        else:
            self._mpl_index = []
            self._mpl_factor = []
        self._relink_mpc = True

    def _extract_data_attributes(self):
        self.update_dataframe = False
        for attribute in self._dataframe_attributes + self._coildata_indices:
            if attribute in ['power', 'plasma', 'optimize', 'feedback']:
                dtype = bool
            else:
                dtype = float
            if attribute in self:  # read from DataFrame column
                value = self[attribute].to_numpy(dtype=dtype)
            elif attribute in self._default_attributes:  # default
                value = np.array([self._default_attributes[attribute]
                                  for __ in range(self.nC)], dtype=dtype)
            else:
                value = np.zeros(self.nC, dtype=dtype)
            if attribute in self._mpc_attributes:  # mpc compaction
                value = value[self._mpc_iloc]
            setattr(self, f'_{attribute}', value)
        self._ionize_index = self._plasma[self._mpc_referance]

    def _extract_reduction_index(self):
        """Extract reduction incices (reduceat)."""
        if 'coil' in self:  # subcoil
            coil = self.coil.to_numpy()
            if (coil == self._default_attributes['coil']).all():
                _reduction_index = np.arange(self._nC)
            else:
                _name = coil[0]
                _reduction_index = [0]
                for i, name in enumerate(coil):
                    if name != _name:
                        _reduction_index.append(i)
                        _name = name
            self._reduction_index = np.array(_reduction_index)
            self._plasma_iloc = np.arange(len(self._reduction_index))[
                self.plasma[self._reduction_index]]
            filament_indices = np.append(self._reduction_index, self.nC)
            plasma_filaments = filament_indices[self._plasma_iloc+1] - \
                filament_indices[self._plasma_iloc]
            self._plasma_reduction_index = \
                np.append(0, np.cumsum(plasma_filaments)[:-1])
        else:  # coil, reduction only applied to subfilaments
            self._reduction_index = None
            self._plasma_iloc = None
            self._plasma_reduction_index = None

    def mpc_index(self, mpc_flag):
        """
        Return subset of _mpc_index based on mpc_flag.

        Parameters
        ----------
        mpc_flag : str
            Selection flag. Full description given in
            :meth:`~coildata.CoilData.mpc_select`.

        Returns
        -------
        index : pandas.DataFrame.Index
            Subset of mpc_index based on mpc_flag.

        """
        return self._mpc_index[self.mpc_select(mpc_flag)]

    def mpc_select(self, mpc_flag):
        """
        Return selection boolean for _mpc_index based on mpc_flag.

        Parameters
        ----------
        mpc_flag : str
            Selection flag.

            - 'full' : update full current vector (~feedback)
            - 'active' : update active coils (power & ~plasma & ~feedback)
            - 'passive' : update passive coils (~power & ~plasma & ~feedback)
            - 'free' : update free coils (optimize & ~plasma & ~feedback)
            - 'fix' : update fix coils (~optimize & ~plasma & ~feedback)
            - 'plasma' : update plasma (plasma & ~feedback)
            - 'coil' : update all coils (~plasma & ~feedback)
            - 'feedback' : update feedback stabilization coils


        Raises
        ------
        IndexError
            mpc_flag not in
            [full, active, passive, free, fix, plasma, coil, feedback].

        Returns
        -------
        mpc_select : array-like, shape(_nC,)
            Boolean selection array.

        """
        if self.nC > 0 and self._mpc_iloc is not None:
            if mpc_flag == 'full':
                mpc_select = np.full(self._nC, True) & ~self._feedback
            elif mpc_flag == 'active':
                mpc_select = self._power & ~self._plasma & ~self._feedback
            elif mpc_flag == 'passive':
                mpc_select = ~self._power & ~self._plasma & ~self._feedback
            elif mpc_flag == 'free':
                mpc_select = self._optimize & ~self._plasma & ~self._feedback
            elif mpc_flag == 'fix':
                mpc_select = ~self._optimize & ~self._plasma & ~self._feedback
            elif mpc_flag == 'plasma':
                mpc_select = self._plasma & ~self._feedback
            elif mpc_flag == 'coil':
                mpc_select = ~self._plasma & ~self._feedback
            elif mpc_flag == 'feedback':
                mpc_select = self._feedback
            else:
                raise IndexError(f'flag {mpc_flag} not in '
                                 '[full, actitve, passive, free, fix, '
                                 'plasma, coil, feedback]')
            return mpc_select

    @property
    def current_update(self):
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

    @current_update.setter
    def current_update(self, update_flag):
        self._current_update = update_flag
        self._current_index = self.mpc_select(update_flag)

    @property
    def current_index(self):
        """Return current index."""
        return self._current_index

    @property
    def current_status(self):
        """
        Display current index update status.

        - power
        - optimize
        - plasma
        - feedback
        - current update

        """
        if self.nC > 0:
            return DataFrame(
                    {'power': self._power,
                     'optimize': self._optimize,
                     'plasma': self._plasma,
                     'feedback': self._feedback,
                     self.current_update: self._current_index},
                    index=self._mpc_index)
        else:
            return DataFrame(columns=['power', 'optimize', 'plasma',
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

    @property
    def _nC(self):
        """
        Return mpc coil number.

        Returns
        -------
        _nC : int
            Number of coils without mpc constraints.

        """
        return len(self._mpc_iloc)

    @property
    def _nI(self):
        """Return number of indexed coils."""
        return sum(self._current_index)

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

    @property
    def ionize(self):
        """
        Manage plasma ionization_index.

        Set index to True for all intra-spearatrix filaments.

        Parameters
        ----------
        index : array-like, shape(nP,)
            ionization index for plasma filament bundle, shape(nP,).

        Returns
        -------
        _ionize_index : array-like, shape(nC,)
            Ionization index.

        """
        return self._ionize_index

    @ionize.setter
    def ionize(self, index):
        active = np.full(self.nP, False)
        active[index] = True
        self._ionize_index[self.plasma] = active
        self.Np = 1  # initalize turn number

    @property
    def Np(self):
        r"""
        Plasma filament turn number.

        Parameters
        ----------
        value : float or array-like
            Set turn number of plasma filaments

            Ensure :math:`\sum |Np| = 1`.

        Returns
        -------
        Np : np.array, shape(nP,)
            Plasma filament turn number.

        """
        return self._Nt[self.plasma]

    @Np.setter
    def Np(self, value):
        self._Nt[self.plasma & ~self._ionize_index] = 0
        self._Nt[self.plasma & self._ionize_index] = value
        # normalize plasma tun number
        Nt_sum = np.sum(self._Nt[self.plasma])
        if Nt_sum > 0:
            self._Nt[self.plasma] /= Nt_sum
        self._update_dataframe['Nt'] = True

    @property
    def nP(self):
        """Return number of plasma filaments."""
        return np.sum(self.plasma)

    @property
    def nPlasma(self):
        """Return number of active plasma fillaments."""
        return len(self.Np[self.Np > 0])

    @property
    def Ip(self):
        """
        Return plasma line current [A].

        Returns
        -------
        It : float
            sum(It) (float): plasma line current [A]

        """
        return self._Ic[self._plasma]

    @Ip.setter
    def Ip(self, value):
        self._Ic[self._plasma] = value
        self._update_dataframe['Ic'] = True

    @property
    def Ip_sum(self):
        """Net plasma current."""
        return self.Ip.sum()

    @property
    def Ip_sign(self):
        """Plasma polarity."""
        return np.sign(self.Ip_sum)

    @staticmethod
    @contextmanager
    def _write_to_dataframe(self):
        """
        Apply a local attribute lock via the _update_coilframe flag.

        Prevent local attribute write via __setitem__ during dataframe update.

        Yields
        ------
        None
            with self._write_to_dataframe(self):.

        """
        self._update_coilframe = False
        yield
        self._update_coilframe = True

    def refresh_dataframe(self):
        """Transfer data from coilframe attributes to dataframe."""
        if self.update_dataframe:
            _update_dataframe = self._update_dataframe.copy()
            self.update_dataframe = False
            with self._write_to_dataframe(self):
                for attribute in _update_dataframe:
                    if _update_dataframe[attribute]:
                        if attribute in ['Ic', 'It']:
                            current = self._Ic[self._mpc_referance] * \
                                self._mpc_factor
                            if attribute == 'It':
                                current *= self._Nt
                            self.loc[:, attribute] = current
                            _attr = next(attr for attr in ['Ic', 'It']
                                         if attr != attribute)
                            self._update_dataframe[_attr] = False
                        else:
                            self.loc[:, attribute] = getattr(self, attribute)

    def refresh_coilframe(self, key):
        """
        Transfer data from dataframe to coilframe attributes.

        Parameters
        ----------
        key : str
            CoilFrame column.

        Returns
        -------
        None.

        """
        if self._update_coilframe:  # protect against regressive update
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
            if key in self._update_dataframe:
                self._update_dataframe[key] = False
