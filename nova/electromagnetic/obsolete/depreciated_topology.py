# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:01:13 2020

@author: mcintos
"""


    @property
    def polarity(self):
        """
        Return plasma current polarity.

        Returns
        -------
        polarity: int
            Plasma current polarity.

        """
        if self._update_polarity:
            self._polarity = self.source.coilframe.Ip_sign
            self._update_polarity = False
        return self._polarity

    def _signed_flux(self, x):
        return -1 * self.polarity * self.interpolate('Psi').ev(*x)

    def _signed_flux_gradient(self, x):
        return -1 * self.polarity * np.array(
            [self.interpolate('Psi').ev(*x, dx=1),
             self.interpolate('Psi').ev(*x, dy=1)])

    def get_Opoint(self, xo=None):
        """
        Return coordinates of plasma O-point.

        O-point defined as center of nested flux surfaces.

        Parameters
        ----------
        xo : array-like(float), shape(2,), optional
            Sead coordinates (x, z). The default is None.

            - None: xo set to grid center

        Raises
        ------
        TopologyError
            Failed to find signed flux minimum.

        Returns
        -------
        Opoint, array-like(float), shape(2,)
            Coordinates of O-point.

        """
        if xo is None:
            xo = self.bounds.mean(axis=1)
        res = scipy.optimize.minimize(
            self._signed_flux, xo,
            jac=self._signed_flux_gradient, bounds=self.bounds)
        if not res.success:
            raise TopologyError('Opoint signed flux minimization failure\n\n'
                                f'{res}.')
        return res.x

    @property
    def Opoint(self):
        """
        Return coordinates for the center(s) of nested flux surfaces.

        Returns
        -------
        Opoints : array-like, shape(n, 2)
            O-point coordinates (x, z).

        """
        if self._update_Opoint or self._Opoint is None:
            self._Opoint = self.get_Opoint(xo=self._Opoint)
            self._update_Opoint = False
        return self._Opoint

    @property
    def Opsi(self):
        """
        Return poloidal flux calculated at O-point.

        Returns
        -------
        Opsi: float
            O-point poloidal flux.

        """
        if self._update_Opsi:
            self._Opsi = float(self.interpolate('Psi').ev(*self.Opoint))
            self._update_Opsi = False
        return self._Opsi

    @property
    def Xpoint(self):
        """
        Manage Xpoint locations.

        Parameters
        ----------
        xo : array-like, shape(n, 2)
            Sead Xpoints.

        Returns
        -------
        Xpoint: ndarray, shape(2)
            Coordinates of primary Xpoint (x, z).

        """
        if self._update_Xpoint or self._Xpoint is None:
            if self._Xpoint is None:  # sead with boundary midsides
                bounds = self.bounds
                self.Xpoint = [[np.mean(bounds[0]), bounds[1][i]]
                               for i in range(2)]
            nX = len(self._Xpoint)
            _Xpoint = np.zeros((nX, 2))
            _Xpsi = np.zeros(nX)
            for i in range(nX):
                _Xpoint[i] = self.get_Xpoint(self._Xpoint[i])
                _Xpsi[i] = self.interpolate('Psi').ev(*_Xpoint[i])
            self._Xpoint = _Xpoint[np.argsort(_Xpsi)]
            if self.source.coilframe.Ip_sign > 0:
                self._Xpoint = self._Xpoint[::-1]
            self._update_Xpoint = False
        return self._Xpoint[0]

        '''
    def _get_opt(self, opt_name, minimize=True):
        """
        Return nlopt optimization instance.

        Parameters
        ----------
        opt_name : str
            Optimization name.
        minimize : bool, optional
            Minimize objective function. The default is True.

        Returns
        -------
        opt_instance : nlopt.opt
            Nlopt optimization instance.

        """
        opt_name = self._opt_name(opt_name, minimize)
        return getattr(self, opt_name)
    '''

        '''
    def set_opt(self):
        """
        Set topology optimizers.

        Returns
        -------
        None.

        """
        self._set_opt('field', minimize=True)
        self._set_opt('flux', minimize=True)
        self._set_opt('flux', minimize=False)
    '''

