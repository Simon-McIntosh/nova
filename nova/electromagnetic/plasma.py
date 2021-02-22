

class Plasma:

    self._ionize_index = self._plasma[self._mpc_referance]

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
        # normalize plasma turn number
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

