import numpy as np

from nova.electromagnetic.biotelements import Filament


class CoilMatrix:
    '''
    container for coil_matrix and subcoil_matrix (filament) data

    Formulae:
        Psi = [psi][Ic] (Wb)
        Bx = [bx][Ic] (T)
        Bz = [bz][Ic] (T)


    Attributes:
        psi (np.array): flux matrix [nT, nC]
        _psi (np.array): plasma unit filaments [:, nP]

        bx (np.array): radial field matrix [nT, nC]
        bz (np.array): vertical field matrix [nT, nC]
        _b* (np.array): plasma unit filaments [:, nP]

    '''

    # main class attribures
    _coilmatrix_attributes = ['psi', '_psi',
                              'bx', '_bx', 'bz', '_bz']

    def __init__(self):
        self._initialize_coilmatrix_attributes()

    def _initialize_coilmatrix_attributes(self):
        for attribute in self._coilmatrix_attributes:
            setattr(self, f'{attribute}', np.array([]))

    def flux_matrix(self, method):
        'calculate filament flux (inductance) matrix'
        psi = self.calculate(method, 'scalar_potential')
        self.psi , self._psi = self.save_matrix(psi)

    def field_matrix(self, method):
        'calculate subcoil field matrix'
        field = {'x': 'radial_field', 'z': 'vertical_field'}
        for xz in field:  # save field matricies
            b, _b = self.save_matrix(self.calculate(method, field[xz]))
            setattr(self, f'b{xz}', b)
            setattr(self, f'_b{xz}', _b)

    def calculate(self, method, attribute):
        'calculate biot attributes (flux, radial_field, vertical_field)'
        return getattr(method, attribute)()

    def solve(self, **biot_attributes):
        self.biot_attributes = biot_attributes  # update attributes
        self.update_biotset()  # assemble geometory matrices
        filament = Filament(self.source, self.target)
        self.flux_matrix(filament)  # assemble flux interaction matrix
        self.field_matrix(filament)  # assemble field interaction matricies
        self._solve = False

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
        """Update plasma turns."""
        if self.source.nP > 0:  # source plasma filaments
            _M = _M.copy()  # unlink
            if self.source_turns:
                _M *= self.source.coilframe.Np.reshape(1, -1)
            if self.target_turns:
                _M[self.target._plasma_index, :] *= \
                        self.target.coilframe.Np.reshape(-1, 1)
        if self.reduce_source and len(self.source._reduction_index) < self.nS:
            _M = np.add.reduceat(
                _M, self.source._plasma_reduction_index, axis=1)
        if self.reduce_target and len(self.target._reduction_index) < self.nT:
            _M = np.add.reduceat(_M, self.target._reduction_index, axis=0)
        M[:, self.source._plasma_iloc] = _M

    def update_plasma(self):
        self._update_plasma_turns(self.psi, self._psi)
        self._update_plasma_turns(self.bx, self._bx)
        self._update_plasma_turns(self.bz, self._bz)
        self._update_plasma = False

    def _reshape(self, M):
        if hasattr(self, 'n2d'):
            M = M.reshape(self.n2d)
        return M

    def _dot(self, variable):
        if self._solve:  # solve interaction
            self.solve()
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



