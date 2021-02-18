"""Manage fast access dataframe attributes."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Union
from contextlib import contextmanager
import inspect

import numpy as np
from pandas import DataFrame, Index
from pandas.api.types import is_list_like, is_dict_like

from nova.electromagnetic.metadata import MetaData


@dataclass
class MetaArray(MetaData):
    """Manage CoilFrame metadata - accessed via CoilFrame['attrs']."""

    update: dict[str, str] = field(default_factory=lambda: {
        'required': 'replace', 'additional': 'extend', 'default': 'update'})
    coildata: dict = field(repr=False, default_factory=lambda: {})
    dataframe: dict = field(repr=False, default_factory=lambda: {})
    frame: dict[str, str] = field(
        repr=False, default_factory=lambda: {'current_update': 'full'})

    def validate_input(self):
        """Confirm that all additional attributes have a default value."""


@dataclass
class CoilArray(metaclass=ABCMeta):
    """
    Abstract base class enabling fast access to dynamic coil and subcoil data.

    Extended by CoilFrame. Inherited alongside DataFrame.
    Fast access variables stored as np.arrays _*
    Lazy data exchange implemented with parent DataFrame.

    Key Attributes
    --------------
    Ic : float, array-like
        Coil line current [A]
    It : float, array-like
        Coil turn curent [A.turns]
    Nt : float, array-like
        Coil turn number.
    active : bool, array-like
        Coil current control status.
    optimize : bool, array-like
        Optimization flag.
    plasma : bool, array-like
        Plasma flag.
    feedback : bool, array-like
        Feedback stabilization flag

    """

    def __init__(self):
        """Init metaarray."""
        self.metaarray = MetaArray()

    @abstractmethod
    def attrs(self) -> dict:
        """Return dictionary of global dataframe attributes."""

    @contextmanager
    def _write_dataframe(self):
        """
        Apply a local attribute lock via the _update_coilframe flag.

        Prevent local attribute write via __setitem__ during dataframe update.

        Yields
        ------
        None
            with self._write_dataframe(self):.

        """
        self._update_coilframe = False
        yield
        self._update_coilframe = True

    def refresh_dataframe(self):
        """Transfer data from coilframe attributes to dataframe."""
        if self.update_dataframe:
            _update_dataframe = self._update_dataframe.copy()
            self.update_dataframe = False
            with self._write_dataframe(self):
                for attribute in _update_dataframe:
                    if _update_dataframe[attribute]:
                        if attribute in ['Ic', 'It']:
                            current = self._Ic[self._mpc_referance] * \
                                self._mpc_factor
                            if attribute == 'It':
                                current *= self._Nt
                            self.loc[:, attribute] = current
                            _attr = next(attr for attr in ['Ic', 'It']
                                         if attr != attribute)
                            self._update_dataframe[_attr] = False
                        else:
                            self.loc[:, attribute] = getattr(self, attribute)

    def refresh_coilframe(self, key):
        """
        Transfer data from dataframe to coilframe attributes.

        Parameters
        ----------
        key : str
            CoilFrame column.

        Returns
        -------
        None.

        """
        if self._update_coilframe:  # protect against regressive update
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
            if key in self._update_dataframe:
                self._update_dataframe[key] = False

    def __setattr__(self, key, value):
        """Extend pandas.DataFrame.__setattr__."""
        if key in self._dataframe_attributes:
            self._update_dataframe[key] = True
            if key not in self._coildata_properties:
                # set as private variable
                if key in self._mpc_attributes:
                    nC = self._nC  # mpc variable
                else:
                    nC = self.coil_number  # coil number
                if not is_list_like(value):
                    value *= np.ones(nC, dtype=type(value))
                if len(value) != nC:
                    raise IndexError('Length of mpc vector does not match '
                                     'length of index')
                key = f'_{key}'
        return DataFrame.__setattr__(self, key, value)

    def __getattr__(self, key):
        """Extend pandas.DataFrame.__getattr__."""
        if key in self._dataframe_attributes:
            value = getattr(self, f'_{key}')
            if key in self._mpc_attributes:  # inflate
                value = value[self._mpc_referance]
            return value
        return DataFrame.__getattr__(self, key)

    def __setitem__(self, key, value):
        """Extend pandas.DataFrame.__setitem__."""
        self.refresh_dataframe()  # flush dataframe updates
        DataFrame.__setitem__(self, key, value)
        if key in self._dataframe_attributes:
            self.refresh_coilframe(key)
            if key in ['Nt', 'It', 'Ic']:
                self._It = self.It
            if key == 'Nt':
                self._update_dataframe['Ic'] = True
                self._update_dataframe['It'] = True
            if key in ['Ic', 'It']:
                _key = next(k for k in ['Ic', 'It'] if k != key)
                self._update_dataframe[_key] = True

    def __getitem__(self, key):
        """Extend pandas.DataFrame.__getitem__."""
        if key in self._dataframe_attributes:
            self.refresh_dataframe()
        return DataFrame.__getitem__(self, key)

    def _get_value(self, index, col, takeable=False):
        """Extend pandas.DataFrame._get_value."""
        if col in self._dataframe_attributes:
            self.refresh_dataframe()
        return DataFrame._get_value(self, index, col, takeable)

    def __repr__(self):
        """Extend pandas.DataFrame.__repr__."""
        self.refresh_dataframe()
        return DataFrame.__repr__(self)



