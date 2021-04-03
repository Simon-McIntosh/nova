
from dataclasses import dataclass, field

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.coil import Coil

#self._ionize_index = self._plasma[self._mpc_referance]

@dataclass
class Plasma(Coil):
    """Generate plasma coils."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)
    delta: float

    def insert(self, boundary, iloc=None, mesh=True, **additional):
        """
        Extend Coil.insert.

        Add plasma to coilset and generate plasma grid.

        Plasma inserted into coilframe with subcoils meshed accoriding
        to delta and trimmed to the inital boundary curve.

        Parameters
        ----------
        boundary : array_like or Polygon
            External plasma boundary. Coerced into positively oriented curve.
        name : str, optional
            Plasma coil name.
        delta : float, optional
            Plasma subcoil dimension. If None defaults to self.dPlasma
        **kwargs : dict
            Keyword arguments passed to PlasmaGrid.generate_grid()

        Returns
        -------
        None.

        """

        #self.biot_instances = ['plasmafilament', 'plasmagrid']
        self.plasma_boundary = boundary
        # construct plasma coil from polygon
        super().insert()

        '''
        self.add_coil(0, 0, 0, 0, polygon=self.plasma_boundary,
                      cross_section='polygon', turn_section='rectangle',
                      dCoil=self.dPlasma, name=name, plasma=True, active=True,
                      part='plasma')
        '''

        '''
        self.plasmagrid.generate_grid(**kwargs)
        grid_factor = self.dPlasma/self.plasmagrid.dx
        # self._add_vertical_stabilization_coils()
        self.plasmagrid.cluster_factor = 1.5*grid_factor
        self.plasmafilament.add_plasma()
        '''


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

