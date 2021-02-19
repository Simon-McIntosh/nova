"""Manage fast access dataframe attributes."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from contextlib import contextmanager
import inspect

import numpy as np
import pandas

from nova.electromagnetic.metadata import MetaData


@dataclass
class MetaArray(MetaData):
    """Manage CoilFrame metadata - accessed via CoilFrame['attrs']."""

    array: list[str] = field(default_factory=lambda: ['x'])
    update: list[bool] = field(default_factory=dict)
    frame: dict[str, str] = field(
        repr=False, default_factory=lambda: {'current_update': 'full'})

    def validate(self):
        """Extend MetaData.validate."""
        MetaData.validate(self)
        self.update |= {attr: False
                        for attr in self.array if attr not in self.update}


#@dataclass
class FrameArray(metaclass=ABCMeta):
    """
    Abstract base class enabling fast access to dynamic coil and subcoil data.

    Extended by CoilFrame. Inherited alongside DataFrame.
    Fast access variables stored as np.arrays _*
    Lazy data exchange implemented with parent DataFrame.

    """

    #data: dict[str, np.ndarray] = field(init=False, repr=False)
    #attrs: dict = field(init=False, repr=False)

    def __init__(self):
        """Build fast access data."""
        self.attrs['data'] = {}
        for attribute in self.metaarray.array:
            self.data[attribute] = self[attribute].to_numpy()
        # extract properties
        self.metaarray.properties = [p for p, __ in inspect.getmembers(
            FrameArray, lambda o: isinstance(o, property))]

    @property
    def data(self):
        return self.attrs['data']

    @abstractmethod
    def metaframe(self):
        """Return MetaFrame instance."""

    def __repr__(self):
        """Extend pandas.DataFrame.__repr__."""
        #self.refresh_dataframe()
        return pandas.DataFrame.__repr__(self)

    def _init_array(self):
        if 'metaarray' not in self.attrs:
            self.attrs['metaarray'] = MetaArray()

    @property
    def metaarray(self):
        """Return metaarray."""
        return self.attrs['metaarray']

    def __setattr__(self, key, value):
        """Extend pandas.DataFrame.__setattr__."""
        if 'metaarray' in self.attrs:
            if key in self.metaarray.array:
                self.metaarray.update[key] = True
                print(key)
                '''
                if key not in self.metaarray.properties:
                    # set as private variable
                    if key in self._mpc_attributes:
                        nC = self._nC  # mpc variable
                    else:
                        nC = self.coil_number  # coil number
                    if not pandas.api.types.is_list_like(value):
                        value *= np.ones(nC, dtype=type(value))
                    if len(value) != nC:
                        raise IndexError('Length of mpc vector does not match '
                                         'length of index')
                    key = f'_{key}'
                '''
                return None
        return pandas.DataFrame.__setattr__(self, key, value)

    def __getattr__(self, key):
        """Extend pandas.DataFrame.__getattr__."""
        if key in self.data:
            value = self.data[key]
            #if key in self._mpc_attributes:  # inflate
            #    value = value[self._mpc_referance]
            return value
        return pandas.DataFrame.__getattr__(self, key)


    def refresh_dataframe(self):
        """Transfer data from coilframe attributes to dataframe."""
        if self.update_dataframe:
            update = self.metaarray.update.copy()
            self.update_dataframe = False
            with self._write_dataframe():
                for attribute in update:
                    if update[attribute]:
                        if attribute in ['Ic', 'It']:
                            current = self._Ic[self._mpc_referance] * \
                                self._mpc_factor
                            if attribute == 'It':
                                current *= self._Nt
                            self.loc[:, attribute] = current
                            _attr = next(attr for attr in ['Ic', 'It']
                                         if attr != attribute)
                            self.metaarray.update[_attr] = False
                        else:
                            self.loc[:, attribute] = getattr(self, attribute)

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

    '''
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
            if key in self.metaarray.update:
                self.metaarray.update[key] = False

    def __setitem__(self, key, value):
        """Extend pandas.DataFrame.__setitem__."""
        self.refresh_dataframe()  # flush dataframe updates
        DataFrame.__setitem__(self, key, value)
        if key in self._dataframe_attributes:
            self.refresh_coilframe(key)
            if key in ['Nt', 'It', 'Ic']:
                self._It = self.It
            if key == 'Nt':
                self.metaarray.update['Ic'] = True
                self.metaarray.update['It'] = True
            if key in ['Ic', 'It']:
                _key = next(k for k in ['Ic', 'It'] if k != key)
                self.metaarray.update[_key] = True

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
    '''
