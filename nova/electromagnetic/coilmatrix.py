"""CoilMatix calculation methods."""

import numpy as np

from nova.electromagnetic.biotelements import Filament


class CoilMatrix:
    r"""
    Calculation methods for Biot Savart instances.

    Subclassed by BiotSet.

    Formulae
    --------
        :math:`\mathbf{\Psi} = \mathbf{\psi} \: \mathbf{I_c}`

        :math:`\mathbf{B_x} = \mathbf{b_x} \: \mathbf{I_c}`

        :math:`\mathbf{B_z} = \mathbf{b_z} \: \mathbf{I_c}`

    Attributes
    ----------
        psi : array-like, shape(nT, nC)
            Poloidal flux matrix, :math:`\mathrm{WbA}^{-1}`.
        _psi : array-like, shape(nT, nP)
            Poloidal flux plasma unit filaments,
            :math:`\mathrm{WbA}^{-1}\mathrm{turn}^{-2}`.
        bx, bz : array-like, shape(nT, nC)
            Field matrix (radial, vertical), :math:`\mathrm{TA}^{-1}`.
        _bx, _bz : array-like, shape(nT, nP)
            Field matrix plasma unit filaments (radial, vertical),
            :math:`\mathrm{TA}^{-1}\mathrm{turn}^{-2}`.

    """

    # main class attribures
    _coilmatrix_attributes = ['psi', '_psi', 'bx', '_bx', 'bz', '_bz']

    def __init__(self):
        self._initialize_coilmatrix_attributes()

    def _initialize_coilmatrix_attributes(self):
        for attribute in self._coilmatrix_attributes:
            setattr(self, f'{attribute}', np.array([]))

    def solve_interaction(self, **biot_attributes):
        """
        Solve biot interaction.

        Calculate poloidal flux and field interaction matricies.

        Parameters
        ----------
        **biot_attributes : dict
            Optional keyword attributes.

        Returns
        -------
        None.

        """
        self.biot_attributes = biot_attributes  # update attributes
        self.update_biotset()  # assemble geometory matrices
        filament = Filament(self.source, self.target)
        self.flux_matrix(filament)  # assemble flux interaction matrix
        self.field_matrix(filament)  # assemble field interaction matricies
        self._solve_interaction = False
        self._update_plasma = True

    def flux_matrix(self, method):
        """Calculate filament flux (inductance) matrix."""
        psi = self._evaluate(method, 'scalar_potential')
        self.psi, self._psi = self.save_matrix(psi)

    def field_matrix(self, method):
        """Calculate subcoil field matrix."""
        field = {'x': 'radial_field', 'z': 'vertical_field'}
        for xz in field:  # save field matricies
            b, _b = self.save_matrix(self._evaluate(method, field[xz]))
            setattr(self, f'b{xz}', b)
            setattr(self, f'_b{xz}', _b)

    def _evaluate(self, method, attribute):
        """
        Compute and return attribute from biot method.

        Parameters
        ----------
        method : BiotElement.
            Method for calculating biot interactions.
        attribute : str
            Requested attribute.
            [scalar_potential, radial_field, vertical_field].

        Returns
        -------
        value : array-like, shape(nT*nS,)
            Requested attribute.

        """
        return getattr(method, attribute)()

    def save_matrix(self, M):
        """
        Save interaction matrix **M**.

        Extract plasma interaction from full matrix and save as _M.
        Apply source and target turns to full matrix only.

        Parameters
        ----------
        M : array-like, shape(nT*nS,)
            Unit turn source-target interaction matrix.

        Returns
        -------
        M : array-like, shape(nT, nS)
            Full interaction matrix.
        _M : array-like, shape(nT, nP)
            Unit plasma interaction matrix.

        """
        # extract plasma interaction
        _M = M.reshape(self.nT, self.nS)[:, self.source._plasma_index]
        if self.source_turns:
            M *= self.source._Nt_
        if self.target_turns:
            M *= self.target._Nt_
        M = M.reshape(self.nT, self.nS)  # source-target reshape (matrix)
        # reduce
        if self.reduce_source and len(self.source._reduction_index) < self.nS:
            M = np.add.reduceat(M, self.source._reduction_index, axis=1)
        if self.reduce_target and len(self.target._reduction_index) < self.nT:
            M = np.add.reduceat(M, self.target._reduction_index, axis=0)
        return M, _M  # turn-turn interaction, unit plasma interaction

    @property
    def Fx(self):
        return np.add.reduceat(2*np.pi*self.source.coilframe.x*self.source.coilframe.It*self.Bz,
                      self.source._reduction_index)

    def _update_plasma_turns(self, M, _M):
        """
        Update plasma turns in source interaction matrix **M**.

        Multiply unit plasma interaction matrix **_M** by source and target
        plasma turns (dependant on source_turns and target_turns flags).

        Parameters
        ----------
        M : array-like, shape(nT, nS)
            Full interaction matrix, augmented inplace.
        _M : array-like, shape(nT, nP)
            Unit plasma interaction matrix.

        Returns
        -------
        None.

        """
        if self.source.nP > 0:  # source plasma filaments
            _M = _M.copy()  # unlink
            if self.source_turns:
                _M *= self.source.coilframe.Np.reshape(1, -1)
            if self.target_turns:
                _M[self.target._plasma_index, :] *= \
                        self.target.coilframe.Np.reshape(-1, 1)
            if self.reduce_source and \
                    len(self.source._plasma_reduction_index) < self.source.nP:
                _M = np.add.reduceat(
                    _M, self.source._plasma_reduction_index, axis=1)
            if self.reduce_target and \
                    len(self.target._reduction_index) < self.nT:
                _M = np.add.reduceat(_M, self.target._reduction_index, axis=0)
            M[:, self.source._plasma_iloc] = _M

    def update_plasma(self):
        """
        Update plasma turns in psi, bx, and bx interactions.

        Returns
        -------
        None.

        """
        self._update_plasma_turns(self.psi, self._psi)
        self._update_plasma_turns(self.bx, self._bx)
        self._update_plasma_turns(self.bz, self._bz)
        self._update_plasma = False

    def _reshape(self, M):
        if hasattr(self, 'n2d'):
            M = M.reshape(self.n2d)
        return M

    """ split dot into two parts """
    def _dot(self, variable):
        if self._solve_interaction:  # solve interaction
            self.solve_interaction()
        if self._update_plasma:  # update plasma turns
            self.update_plasma()
        matrix = getattr(self, variable.lower())
        return self._reshape(np.dot(matrix, self.source.coilframe._Ic))

    @property
    def Psi(self):
        return self._dot('Psi')

    @property
    def Bx(self):
        return self._dot('Bx')

    @property
    def Bz(self):
        return self._dot('Bz')

    @property
    def B(self):
        return np.linalg.norm([self._dot('Bx'), self._dot('Bz')], axis=0)


if __name__ == '__main__':
    cm = CoilMatrix()



