


class Current:

    _required_attributes = ['active', 'plasma', 'optimize', 'feedback']

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
