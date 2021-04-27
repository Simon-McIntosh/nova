"""BiotMatix calculation methods."""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from nova.electromagnetic.biotfilament import BiotFilament
from nova.electromagnetic.biotset import BiotSet

# _coilmatrix_properties = ['Psi', 'Bx', 'Bz']


@dataclass
class BiotMatrix:
    r"""
    Calculation methods for Biot Savart instances.

    Formulae
    --------
        :math:`\pmb{\Psi} = \pmb{\psi} \: \mathbf{I_c}`

        :math:`\mathbf{B_x} = \mathbf{b_x} \: \mathbf{I_c}`

        :math:`\mathbf{B_z} = \mathbf{b_z} \: \mathbf{I_c}`

    Attributes
    ----------
        static : array-like, shape(target, source)
            static interaction matrix, :math:`*\mathrm{A}^{-1}`.
        plasma : array-like, shape(target, source)
            static interaction matrix (plasma unit filaments),
            :math:`*\mathrm{A}^{-1}\mathrm{turn}^{-2}`.

    """

    frameset: BiotSet
    variable: str
    static: npt.ArrayLike = field(init=False, repr=False)
    plasma: npt.ArrayLike = field(init=False, repr=False)

    def __post_init__(self):
        
        filament = BiotFilament(self.source, self.target)

    def flux_matrix(self, method):
        """Calculate filament flux (inductance) matrix."""
        psi = self._calculate(method, 'scalar_potential')
        self._psi, self._psi_ = self.save_matrix(psi)

    def field_matrix(self, method):
        """Calculate subcoil field matrix."""
        field = {'x': 'radial_field', 'z': 'vertical_field'}
        for xz in field:  # save field matricies
            _b, _b_ = self.save_matrix(self._calculate(method, field[xz]))
            setattr(self, f'_b{xz}', _b)
            setattr(self, f'_b{xz}_', _b_)

    def calculate(self, method, attribute):
        """
        Calculate and return attribute from biot method.

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
        Save interaction matrices (static and plasma.

        Extract plasma interaction from full matrix.
        Multiply static source and target turns to full matrix only.

        Parameters
        ----------
        M : array-like, shape(target*source,)
            Unit turn source-target interaction matrix.

        Returns
        -------
        None.

        """
        # extract plasma interaction
        _M_ = M.reshape(self.nT, self.nS)[:, self.source.plasma]
        if self.source_turns:
            M *= self.source._nturn_
        if self.target_turns:
            M *= self.target._nturn_
        _M = M.reshape(self.nT, self.nS)  # source-target reshape (matrix)
        # reduce
        if self.reduce_source and len(self.source._reduction_index) < self.nS:
            _M = np.add.reduceat(_M, self.source._reduction_index, axis=1)
        if self.reduce_target and len(self.target._reduction_index) < self.nT:
            _M = np.add.reduceat(_M, self.target._reduction_index, axis=0)
        return _M, _M_  # turn-turn interaction, unit plasma interaction

    def solve_interaction(self):
        """Solve biot interaction. """
        self.assemble_biotset()  # assemble geometory matrices
        filament = BiotFilament(self.source, self.target)
        self.flux_matrix(filament)  # assemble flux interaction matrix
        self.field_matrix(filament)  # assemble field interaction matricies
        self.update_biot = True
        self.update_interaction = False


    def solve_biot(self):
        """
        Evaluate all biot attributes.

        Returns
        -------
        None.

        """
        self.solve_interaction()
        if self.target.nT == 0:
            'Return with warning if targets not set.'
            warn('Targets not set in:\n'
                 f'{self.__class__}')
            return
        for variable in self._coilmatrix_properties:
            self.evaluate(variable)

    def evaluate(self, variable):
        """
        Return requested variable, re-calculate on only when necessary.

        Ensure relevant interaction matricies and plasma turn settings
        are up to date. Re-calculation triggered by update flags.

        Parameters
        ----------
        variable : str
            Variable name (capitalized).

        Returns
        -------
        variable : array-like, shape(*self.n2d) or shape(nT)
            Variable value.

        """
        self.solve_interaction()
        variable = variable.capitalize()
        self._dot(variable)
        return getattr(self, f'_{variable}') + getattr(self, f'_{variable}_')

    def _dot(self, variable):
        self._dot_plasma_turns(variable)
        self._dot_coil_current(variable)
        self._dot_plasma_current(variable)

    def _dot_plasma_turns(self, variable):
        """
        Update plasma turns in source interaction matrix **_M**.

        Multiply unit plasma interaction matrix **_M_** by source and target
        plasma turns (dependant on source_turns and target_turns flags).

        Parameters
        ----------
        variable : str
            Coilmatrix variable.
        _M : array-like, shape(nT, nS)
            Full interaction matrix, augmented inplace.
        _M_ : array-like, shape(nT, nP)
            Unit plasma interaction matrix.

        Returns
        -------
        None.

        """
        if self._update_plasma_turns[variable]:
            _M = getattr(self, f'_{variable.lower()}')
            _M_ = getattr(self, f'_{variable.lower()}_')
            if self.source.nP > 0:  # source plasma filaments
                _M_ = _M_.copy()  # unlink
                if self.source_turns:
                    _M_ *= self.source.coilframe.Np.reshape(1, -1)
                if self.target_turns:
                    _M_[self.target.plasma, :] *= \
                            self.target.nturn[self.target.plasma].reshape(-1, 1)
                if self.reduce_source and \
                        len(self.source._plasma_reduction_index) <\
                        self.source.nP:
                    _M_ = np.add.reduceat(
                        _M_, self.source._plasma_reduction_index, axis=1)
                if self.reduce_target and \
                        len(self.target._reduction_index) < self.nT:
                    _M_ = np.add.reduceat(
                        _M_, self.target._reduction_index, axis=0)
                _M[:, self.source._plasma_iloc] = _M_
                self._update_plasma_turns[variable] = False
                self._update_plasma_current[variable] = True

    def _dot_coil_current(self, variable):
        if self._update_coil_current[variable]:
            setattr(self, f'_{variable}', self._dot_current(variable, False))
            self._update_coil_current[variable] = False

    def _dot_plasma_current(self, variable):
        if self._update_plasma_current[variable]:
            setattr(self, f'_{variable}_', self._dot_current(variable, True))
            self._update_plasma_current[variable] = False

    def _dot_current(self, variable, plasma):
        index = self.source._plasma if plasma else ~self.source._plasma
        matrix = getattr(self, f'_{variable.lower()}')[:, index]
        current = self.source.coilframe.Ic[self.source.frameindex][index]
        vector = np.dot(matrix, current)
        return self._reshape(vector)

    def _reshape(self, M):
        if hasattr(self, 'n2d'):
            M = M.reshape(self.n2d)
        return M

    @property
    def Psi(self):
        return self.evaluate('Psi')

    @property
    def Bx(self):
        return self.evaluate('Bx')

    @property
    def Bz(self):
        return self.evaluate('Bz')

    @property
    def B(self):
        return np.linalg.norm([self.Bx, self.Bz], axis=0)

    @property
    def Fx(self):
        # TODO evaluate frame index for reduce soruce (passive)
        #coilframe.*[self.source.frameindex]
        return np.add.reduceat(
            2*np.pi*self.source.coilframe.x*self.source.coilframe.It*self.Bz,
            self.source._reduction_index)


if __name__ == '__main__':
    
    source = {'x': [3, 3.4, 3.6], 'z': [3.1, 3, 3.3],
              'dl': 0.3, 'dt': 0.3, 'section': 'hex'}
    biotset = BiotSet(source, source)



