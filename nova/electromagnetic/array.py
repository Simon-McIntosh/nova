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
    @property
    @abstractmethod
    def metaframe(self):
        """Return MetaFrame instance."""

    @property
    def metaarray(self):
        """Return metaarray instance."""
        return self.attrs['metaarray']

    def update_attrs(self, attrs=None):
        """Extend Frame.update_attrs with metaarray instance."""
        super().update_attrs(attrs)
        if 'metaarray' not in self.attrs:
            self.attrs['metaarray'] = MetaArray()
            self.attrs['metadata'].append('metaarray')

    def validate(self):
        """Extend Frame.validate to validate metaarray."""
        super().validate()
        unset = [attr not in self.metaframe.columns
                 for attr in self.metaarray.array]
        if np.array(unset).any():
            raise IndexError(
                f'metaarray attributes {np.array(self.metaarray.array)[unset]}'
                f' already set in metaframe.required {self.metaframe.required}'
                f'or metaframe.additional {self.metaframe.additional}')

    def __repr__(self):
        """Extend pandas.DataFrame.__repr__."""
        #self.refresh_dataframe()
        return pandas.DataFrame.__repr__(self)

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

    def _extract_reduction_index(self):
        """Extract reduction incices (reduceat)."""
        if 'coil' in self:  # subcoil
            coil = self.coil.to_numpy()
            if (coil == self._default_attributes['coil']).all():
                _reduction_index = np.arange(self._nC)
            else:
                _name = coil[0]
                _reduction_index = [0]
                for i, name in enumerate(coil):
                    if name != _name:
                        _reduction_index.append(i)
                        _name = name
            self._reduction_index = np.array(_reduction_index)
            self._plasma_iloc = np.arange(len(self._reduction_index))[
                self.plasma[self._reduction_index]]
            filament_indices = np.append(self._reduction_index, self.coil_number)
            plasma_filaments = filament_indices[self._plasma_iloc+1] - \
                filament_indices[self._plasma_iloc]
            self._plasma_reduction_index = \
                np.append(0, np.cumsum(plasma_filaments)[:-1])
        else:  # coil, reduction only applied to subfilaments
            self._reduction_index = None
            self._plasma_iloc = None
            self._plasma_reduction_index = None

    def mpc_index(self, mpc_flag):
        """
        Return subset of _mpc_index based on mpc_flag.

        Parameters
        ----------
        mpc_flag : str
            Selection flag. Full description given in
            :meth:`~coildata.CoilData.mpc_select`.

        Returns
        -------
        index : pandas.DataFrame.Index
            Subset of mpc_index based on mpc_flag.

        """
        return self._mpc_index[self.mpc_select(mpc_flag)]

    def mpc_select(self, mpc_flag):
        """
        Return selection boolean for _mpc_index based on mpc_flag.

        Parameters
        ----------
        mpc_flag : str
            Selection flag.

            - 'full' : update full current vector (~feedback)
            - 'active' : update active coils (active & ~plasma & ~feedback)
            - 'passive' : update passive coils (~active & ~plasma & ~feedback)
            - 'free' : update free coils (optimize & ~plasma & ~feedback)
            - 'fix' : update fix coils (~optimize & ~plasma & ~feedback)
            - 'plasma' : update plasma (plasma & ~feedback)
            - 'coil' : update all coils (~plasma & ~feedback)
            - 'feedback' : update feedback stabilization coils


        Raises
        ------
        IndexError
            mpc_flag not in
            [full, active, passive, free, fix, plasma, coil, feedback].

        Returns
        -------
        mpc_select : array-like, shape(_nC,)
            Boolean selection array.

        """
        if self.coil_number > 0 and self._mpc_iloc is not None:
            if mpc_flag == 'full':
                mpc_select = np.full(self._nC, True) & ~self._feedback
            elif mpc_flag == 'active':
                mpc_select = self._active & ~self._plasma & ~self._feedback
            elif mpc_flag == 'passive':
                mpc_select = ~self._active & ~self._plasma & ~self._feedback
            elif mpc_flag == 'free':
                mpc_select = self._optimize & ~self._plasma & ~self._feedback
            elif mpc_flag == 'fix':
                mpc_select = ~self._optimize & ~self._plasma & ~self._feedback
            elif mpc_flag == 'plasma':
                mpc_select = self._plasma & ~self._feedback
            elif mpc_flag == 'coil':
                mpc_select = ~self._plasma & ~self._feedback
            elif mpc_flag == 'feedback':
                mpc_select = self._feedback
            else:
                raise IndexError(f'flag {mpc_flag} not in '
                                 '[full, actitve, passive, free, fix, '
                                 'plasma, coil, feedback]')
            return mpc_select
