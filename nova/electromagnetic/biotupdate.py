"""Update methods for BiotFrame."""
from dataclasses import dataclass, field

import numpy as np
import pandas

from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.error import SubSpaceLockError


@dataclass
class BiotUpdate(MetaMethod):
    """Manage Biot update flags."""

    name = 'biotupdate'

    frame: DataFrame = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['Ic', 'nturn'])
    update: dict[str: bool] = field(init=False, default_factory=lambda: {
        'Ic': True, 'It': True, 'nturn': True})
    plasma_turns: bool = field(init=False, default=True)
    coil_current: bool = field(init=False, default=True)
    plasma_current: bool = field(init=False, default=True)

    def initialize(self):
        """Provide initialize interface."""
        pass


    def _set_item(self, indexer, key, value):
        if self.generate and self.frame.get_col(key) == 'It':
            if self.frame.lock('energize') is False \
                    and self.available['nturn']:
                value /= indexer.__getitem__(self._get_key(key, 'nturn'))
                try:
                    self.frame['Ic'] = value
                except SubSpaceLockError:
                    if not isinstance(value, pandas.Series):
                        index = self.frame.loc[key[0], key[1]].index
                        value = pandas.Series(value, index)
                    else:
                        index = value.index
                    index = index.intersection(self.frame.subspace.index)
                    self.frame.subspace.loc[index, 'Ic'] = value[index]
                return
        return indexer.__setitem__(key, value)
    
'''

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
        >>>        super()._flag_update(status)
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

'''

