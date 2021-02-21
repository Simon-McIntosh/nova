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
    """Manage Frame metadata - accessed via Frame['attrs']."""

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
        """Set flag defaults for new attributes."""
        attribute = getattr(self, f'update_{instance}')
        attribute |= {attr: default for attr in self.array
                      if attr not in attribute}
        setattr(self, f'update_{instance}',
                {attr: attribute[attr] for attr in self.array})


class Array(metaclass=ABCMeta):
    """
    Abstract base class enabling fast access to dynamic Frame fields.

    Extended by Frame. Inherited alongside DataFrame.
    Fast access variables stored as np.arrays _*
    Lazy data exchange implemented with parent DataFrame.

    """

    def __repr__(self):
        """Extend pandas.DataFrame.__repr__."""
        #self.refresh_dataframe()
        return pandas.DataFrame.__repr__(self)

    @property
    @abstractmethod
    def metaframe(self):
        """Return MetaFrame instance."""

    @property
    @abstractmethod
    def metaarray(self):
        """Return metaarray instance."""

    def _checkvalue(self, key, value):
        #if key not in self.metaarray.properties:
        if key in self._mpc_attributes:
            shape = self.unique_coil_number  # mpc variable
        else:
            shape = self.coil_number  # coil number
        if not pandas.api.types.is_list_like(value):
            value *= np.ones(nC, dtype=type(value))
        if len(value) != shape:
            raise IndexError('Length of mpc vector does not match '
                             'length of index')

    def __getattr__(self, key):
        """Extend pandas.DataFrame.__getattr__."""
        if 'metaarray' in self.attrs:
            if key in self.metaarray.array:
                if self.metaarray.update_array[key]:
                    self.metaarray.data[key] = \
                        pandas.DataFrame.__getattr__(self, key).to_numpy()
                    self.metaarray.update_array[key] = False
                #if key in self._mpc_attributes:  # inflate
                #    value = value[self._mpc_referance]
                return self.metaarray.data[key]
        return pandas.DataFrame.__getattr__(self, key)

    def __setattr__(self, key, value):
        """Extend pandas.DataFrame.__setattr__."""
        if 'metaarray' in self.attrs:
            if key in self.metaarray.array:
                self.metaarray.update_array[key] = False
                self.metaarray.update_frame[key] = True
                self.metaarray.data[key] = value
                print(key)

                return None
        return pandas.DataFrame.__setattr__(self, key, value)

    def refresh_dataframe(self):
        """Transfer data from frame attributes to dataframe."""
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
        Apply a local attribute lock via the _update_frame flag.

        Prevent local attribute write via __setitem__ during dataframe update.

        Yields
        ------
        None
            with self._write_dataframe(self):.

        """
        self._update_frame = False
        yield
        self._update_frame = True

    '''
    def refresh_frame(self, key):
        """
        Transfer data from dataframe to frame attributes.

        Parameters
        ----------
        key : str
            Frame column.

        Returns
        -------
        None.

        """
        if self._update_frame:  # protect against regressive update
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
            self.refresh_frame(key)
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
