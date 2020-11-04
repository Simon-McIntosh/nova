"""CoilMatix calculation methods."""

from warnings import warn

import numpy as np
import pandas as pd

from nova.electromagnetic.biotelements import Filament


class CoilMatrix():
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
        _psi : array-like, shape(nT, nC)
            Poloidal flux matrix, :math:`\mathrm{WbA}^{-1}`.
        _psi_ : array-like, shape(nT, nP)
            Poloidal flux matrix (plasma unit filaments),
            :math:`\mathrm{WbA}^{-1}\mathrm{turn}^{-2}`.
        _Psi : array-like, shape(nT,)
            Target poloidal flux(coil component), :math:`\mathrm{Wb}`
        _Psi_ : array-like, shape(nT,)
            Target poloidal flux(plasma component), :math:`\mathrm{Wb}`
        _bx, _bz : array-like, shape(nT, nC)
            Field matrix (radial, vertical), :math:`\mathrm{TA}^{-1}`.
        _bx_, _bz_ : array-like, shape(nT, nP)
            Field matrix plasma unit filaments (radial, vertical),
            :math:`\mathrm{TA}^{-1}\mathrm{turn}^{-2}`.

    """

    _coilmatrix_properties = ['Psi', 'Bx', 'Bz']
    _default_coilmatrix_attributes = {'source_turns': True,
                                      'target_turns': False,
                                      'reduce_source': True,
                                      'reduce_target': False}

    def __init__(self):
        self._initialize_coilmatrix_attributes()

    def _initialize_coilmatrix_attributes(self):
        _update_plasma_turns = {}
        _update_coil_current = {}
        _update_plasma_current = {}
        self._coilmatrix_attributes = []
        for variable in self._coilmatrix_properties:
            self._coilmatrix_attributes.extend(
                [f'_{variable.lower()}', f'_{variable}',
                 f'_{variable.lower()}_', f'_{variable}_'])
            _update_plasma_turns[variable] = True
            _update_coil_current[variable] = True
            _update_plasma_current[variable] = True
        self._default_coilmatrix_attributes.update(
                {'_update_interaction': True,
                 '_update_plasma_turns': _update_plasma_turns,
                 '_update_coil_current': _update_coil_current,
                 '_update_plasma_current': _update_plasma_current})

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

    def _calculate(self, method, attribute):
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
        Save interaction matrix **M**.

        Extract plasma interaction from full matrix and save as _M.
        Apply source and target turns to full matrix only.

        Parameters
        ----------
        M : array-like, shape(nT*nS,)
            Unit turn source-target interaction matrix.

        Returns
        -------
        _M : array-like, shape(nT, nS)
            Full interaction matrix.
        _M_ : array-like, shape(nT, nP)
            Unit plasma interaction matrix.

        """
        # extract plasma interaction
        _M_ = M.reshape(self.nT, self.nS)[:, self.source.plasma]
        if self.source_turns:
            M *= self.source._Nt_
        if self.target_turns:
            M *= self.target._Nt_
        _M = M.reshape(self.nT, self.nS)  # source-target reshape (matrix)
        # reduce
        if self.reduce_source and len(self.source._reduction_index) < self.nS:
            _M = np.add.reduceat(_M, self.source._reduction_index, axis=1)
        if self.reduce_target and len(self.target._reduction_index) < self.nT:
            _M = np.add.reduceat(_M, self.target._reduction_index, axis=0)
        return _M, _M_  # turn-turn interaction, unit plasma interaction

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
        if self.update_interaction:
            self.biot_attributes = biot_attributes  # update attributes
            self.assemble_biotset()  # assemble geometory matrices
            filament = Filament(self.source, self.target)
            self.flux_matrix(filament)  # assemble flux interaction matrix
            self.field_matrix(filament)  # assemble field interaction matricies
            self.update_biot = True
            self.update_interaction = False

    @property
    def update_biot(self):
        """
        Manage biot instance update status flags.

        - update interaction : Core matrix calculation.
        - update plasma turns : Update turn number in plasma sub-matrix.
        - update coil current : Calculate dot product (non-plasma turns).
        - update plasma current : Calculate dot product (plasma turns).

        Parameters
        ----------
        status : bool
            Set update flag.

        Returns
        -------
        status : DataFrame
            Matrix level update status for biot instance.

        """
        data = [{variable: self.update_interaction
                 for variable in self._coilmatrix_properties},
                self.update_plasma_turns,
                self.update_coil_current,
                self.update_plasma_current]
        return pd.DataFrame(data, index=['interaction', 'plasma turns',
                                         'coil current', 'plasma current'])

    @update_biot.setter
    def update_biot(self, status):
        self._confirm_boolean(status)
        self.update_interaction = status
        self.update_plasma_turns = status
        self.update_coil_current = status
        self.update_plasma_current = status

    def _flag_update(self, status):
        """
        Provide hook to set update flags in child class(es).

        Method to be extended by child class(es).

        >>> def _flag_update(self, status):
        >>>        super._flag_update(self, status)
        >>>        # set local update flags here


        Method called when setting status for:

            - interaction
            - plasma_turns
            - coil_current
            - plasma_current

        Parameters
        ----------
        status : bool
            Update status.

        Returns
        -------
        None.

        """
        pass

    @property
    def update_interaction(self):
        """
        Manage update status for solve_interaction method.

        Protect against multiple calls to core matrix calculation.

        Parameters
        ----------
        status : bool
            Update status.

        Returns
        -------
        status : bool
            Update status.

        """
        return self._update_interaction

    @update_interaction.setter
    def update_interaction(self, status):
        self._confirm_boolean(status)
        self._flag_update(status)
        self._update_interaction = status

    @property
    def update_plasma_turns(self):
        r"""
        Manage plasma_turn update status.

        Parameters
        ----------
        status : bool
            Set update flag for all variables in self._coilmatrix_properties.

        Returns
        -------
        update_status : dict
            Plasma_turn pdate flag for each variable in
            self._coilmatrix_properties.

        """
        return self._update_plasma_turns

    @update_plasma_turns.setter
    def update_plasma_turns(self, status):
        self._set_update_status(self._update_plasma_turns, status)

    @property
    def update_coil_current(self):
        r"""
        Manage coil_current update status.

        .. math::
            \_M = \_m \cdot I_c

        Parameters
        ----------
        status : bool
            Set update flag for all variables in self._coilmatrix_properties.

        Returns
        -------
        update_status : dict
            Coil current update flag for each variable in
            self._coilmatrix_properties.

        """
        return self._update_coil_current

    @update_coil_current.setter
    def update_coil_current(self, status):
        self._set_update_status(self._update_coil_current, status)

    @property
    def update_plasma_current(self):
        r"""
        Manage plasma_current update status.

        .. math::
            \_M\_ = \_m\_ \cdot I_c

        Parameters
        ----------
        status : bool
            Set plasma_current flag for all _coilmatrix_properties.

        Returns
        -------
        status : dict
            plasma_current flag status for for each variable in
            _coilmatrix_properties.

        """
        return self._update_plasma_current

    @update_plasma_current.setter
    def update_plasma_current(self, status):
        self._set_update_status(self._update_plasma_current, status)

    @staticmethod
    def _confirm_boolean(status):
        if not isinstance(status, bool):
            raise TypeError(f'type(status) {type(status)} must be boolean')

    def _set_update_status(self, update, status):
        self._confirm_boolean(status)
        self._flag_update(status)
        for attribute in update:
            update[attribute] = status

    def solve(self):
        """
        Evaluate all biot attributes.

        Returns
        -------
        None.

        """
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
        self._dot_plasma_turns(variable)
        self._dot_coil_current(variable)
        self._dot_plasma_current(variable)
        return getattr(self, f'_{variable}') + getattr(self, f'_{variable}_')

    def _dot(self, variable):
        self._dot_plasma_turns(variable)
        self._dot_coil_current(variable)
        self._dot_plasma_current(variable)

    def _dot_plasma_turns(self, variable):
        """
        Update plasma turns in source interaction matrix **M**.

        Multiply unit plasma interaction matrix **_M** by source and target
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
                            self.target.coilframe.Np.reshape(-1, 1)
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
        vector = np.dot(matrix, self.source.coilframe._Ic[index])
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
        return np.add.reduceat(2*np.pi*self.source.coilframe.x*self.source.coilframe.It*self.Bz,
                      self.source._reduction_index)


if __name__ == '__main__':
    cm = CoilMatrix()



