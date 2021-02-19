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

    array: list[str] = field(default_factory=lambda: ['x', 'z'])
    data: dict[str, np.ndarray] = field(default_factory=dict)
    update_array: list[bool] = field(default_factory=dict)
    update_frame: list[bool] = field(default_factory=dict)

    def __repr__(self):
        """Return __repr__."""
        repr_data = {field: getattr(self, field).values()
                     for field in ['update_array', 'update_frame']}
        return pandas.DataFrame(repr_data, index=self.array).__repr__()

    def validate(self):
        """Extend MetaData.validate."""
        MetaData.validate(self)
        # set default update flags
        self.update_flag('array', True)
        self.update_flag('frame', False)

    def update_flag(self, instance, default):
        attribute = getattr(self, f'update_{instance}')
        attribute |= {attr: default for attr in self.array
                      if attr not in attribute}
        setattr(self, f'update_{instance}',
                {attr: attribute[attr] for attr in self.array})


class FrameArray(metaclass=ABCMeta):
    """
    Abstract base class enabling fast access to dynamic coil and subcoil data.

    Extended by CoilFrame. Inherited alongside DataFrame.
    Fast access variables stored as np.arrays _*
    Lazy data exchange implemented with parent DataFrame.

    """

    #data: dict[str, np.ndarray] = field(init=False, repr=False)
    #attrs: dict = field(init=False, repr=False)

    '''
    def __init__(self):
        """Build fast access data."""
        #
        #for attribute in self.metaarray.array:
        #    self.data[attribute] = self[attribute].to_numpy()
        # extract properties
        #self.validate_array()
        self.metaarray.properties = [p for p, __ in inspect.getmembers(
            FrameArray, lambda o: isinstance(o, property))]
    '''

    def __repr__(self):
        """Extend pandas.DataFrame.__repr__."""
        #self.refresh_dataframe()
        return pandas.DataFrame.__repr__(self)

    def _init_array(self):
        if 'metaarray' not in self.attrs:
            self.attrs['metaarray'] = MetaArray()

    def validate_array(self):
        columns = self.metaframe.required + self.metaframe.additional
        unset = [attr not in columns for attr in self.metaarray.array]
        if np.array(unset).any():
            raise IndexError(
                f'metaarray attributes {np.array(self.metaarray.array)[unset]} '
                f'not set in metaframe.required {self.metaframe.required} '
                f'or metaframe.additional {self.metaframe.additional}')

    @property
    def data(self):
        """Return fast access data dictionary."""
        return self.metaarray.data

    @abstractmethod
    def metaframe(self):
        """Return MetaFrame instance."""

    @property
    def metaarray(self):
        """Return metaarray."""
        return self.attrs['metaarray']

    def __getattr__(self, key):
        """Extend pandas.DataFrame.__getattr__."""
        if key in self.metaarray.array:
            if self.metaarray.update_array[key]:
                self.data[key] = \
                    pandas.DataFrame.__getattr__(self, key).to_numpy()
                self.metaarray.update_array[key] = False
            #if key in self._mpc_attributes:  # inflate
            #    value = value[self._mpc_referance]
            return self.data[key]
        return pandas.DataFrame.__getattr__(self, key)

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
