
import pandas

from nova.electromagnetic.superframe import SuperFrame


class SubSpace(SuperFrame):
    """Manage row subspace of frame (all independent coils)."""

    def __init__(self, frame):
        super().__init__(pandas.DataFrame(frame),
                         index=frame.multipoint.index,
                         columns=frame.metaframe.subspace)
        self.metaframe.clear('subspace')

    @property
    def line_current(self):
        """Manage line current."""
        return self.Ic

    @line_current.setter
    def line_current(self, line_current):
        super().__setattr__('Ic', line_current)

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
