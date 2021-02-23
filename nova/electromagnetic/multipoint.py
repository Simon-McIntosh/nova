
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas

from nova.electromagnetic.metadata import MetaData
from nova.electromagnetic.metamethod import MetaMethod

if TYPE_CHECKING:
    from nova.electromagnetic.frame import Frame


@dataclass
class MetaLink(MetaData):
    """Manage MultiPoint metadata."""

    link: list[str] = field(default_factory=lambda: ['Ic'])


@dataclass
class MultiPoint(MetaMethod):
    """Manage multi-point constraints applied across frame.index."""

    frame: Frame = field(repr=False)
    key_attributes: list[str] = field(default_factory=lambda: ['mpc'])
    additional_attributes: list[str] = field(default_factory=lambda: [
        'factor'])
    '''
    iloc: list[int] = field(init=False)
    index: pandas.Index = field(init=False)
    referance: list[int] = field(init=False)
    factor: list[float] = field(init=False)
    link_index: list[int, int] = field(init=False)
    link_factor: list[float] = field(init=False)
    '''

    def __post_init__(self):
        """Generate multi-point constraints."""
        super().__post_init__()
        self.generate()

    def generate(self):
        """
        Generate multipoint.frame constraints if key_attributes in columns.

            - mpc is none or NaN:

                - mpc = ''
                - factor = 0

            - mpc is bool:

                - mpc = '' if False else 'index[0]'
                - factor = 0 if False else 1

            - mpc is int or float:

                - mpc = 'index[0]'
                - factor = value

        """
        if self.enable:
            self.frame.loc[pandas.isna(self.frame.mpc), ['mpc', 'factor']] = \
                self.frame.metaframe.default['mpc'], \
                self.frame.metaframe.default['factor']
            isnumeric = np.array([isinstance(mpc, (int, float)) &
                                  ~isinstance(mpc, bool)
                                  for mpc in self.frame.mpc], dtype=bool)
            istrue = np.array([mpc is True for mpc in self.frame.mpc],
                              dtype=bool)
            isstr = np.array([isinstance(mpc, str)
                              for mpc in self.frame.mpc], dtype=bool)
            self.frame.loc[~istrue & ~isnumeric & ~isstr, 'mpc'] = ''
            index = self.frame.index[istrue | isnumeric]
            if index.empty:
                return
            factor = np.ones(len(self.frame))
            factor[isnumeric] = self.frame.mpc[isnumeric]
            factor = factor[istrue | isnumeric][1:]
            self.add_multipoint(index, factor)

    def add_multipoint(self, index, factor=1):
        """
        Define multi-point constraint linking a set of coils.

        Parameters
        ----------
        index : list[str]
            List of coil names (present in self.frame.index).
        factor : float, optional
            Inter-coil coupling factor. The default is 1.

        Raises
        ------
        IndexError

            - index must be list-like
            - len(index) must be greater than l
            - len(factor) must equal 1 or len(name)-1.

        Returns
        -------
        None.

        """
        if not pandas.api.types.is_list_like(index):
            raise IndexError(f'index: {index} is not list like')
        self.frame.loc[index[0], ['mpc', 'factor']] = '', 1
        index_number = len(index)
        if index_number == 1:
            return
        if not pandas.api.types.is_list_like(factor):
            factor = factor * np.ones(index_number-1)
        elif len(factor) != index_number-1:
            raise IndexError(f'len(factor={factor}) must == 1 '
                             f'or == len(index={index})-1')
        for i in np.arange(1, index_number):
            self.frame.loc[index[i], ['mpc', 'factor']] = index[0], factor[i-1]

    '''
    def drop_multipoint(self, index):
        """Drop multi-point constraints referancing dropped coils."""
        if 'mpc' in self.frame.columns:
            if not pandas.api.types.is_list_like(index):
                index = [index]
            name = [mpc[0] if mpc else '' for mpc in self.frame.mpc]
            drop = [n in index for n in name]
            self.remove_multipoint(drop)

    def remove_multipoint(self, index):
        """Remove multi-point constraint on indexed coils."""
        if not pandas.api.types.is_list_like(index):
            index = [index]
        self.frame.loc[index, 'mpc'] = ''

    def reduce_multipoint(self, matrix):
        """Apply multipoint constraints to coupling matrix."""
        _matrix = matrix[:, self._mpc_iloc]  # extract primary coils
        if len(self._mpl_index) > 0:  # add multi-point links
            _matrix[:, self._mpl_index[:, 0]] += \
                matrix[:, self._mpl_index[:, 1]] * \
                np.ones((len(matrix), 1)) @ self._mpl_factor.reshape(-1, 1)
        return _matrix



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
        #if key in self._mpc_attributes:  # inflate
        #    value = value[self._mpc_referance]

    def __len__(self) -> int:
        """Return frame rank, the number of independant coils."""
        return len(self.iloc)

    def update(self):
        """Update multi-point parameters."""
        mpc = self.get('mpc', np.array([self.metaframe.default['mpc']
                                        for __ in range(self.coil_number)]))
        self._mpc_iloc = [i for i, _mpc in enumerate(mpc) if not _mpc]
        self._mpc_index = self.index[self._mpc_iloc]
        self._mpc_referance = np.zeros(self.coil_number, dtype=int)
        self._mpc_factor = np.ones(self.coil_number, dtype=float)
        _mpc_list = list(self._mpc_index)
        _mpc_array = np.arange(len(_mpc_list))
        mpc_index = mpc != self.metaframe.default['mpc']
        self._mpc_referance[~mpc_index] = _mpc_array
        if sum(mpc_index) > 0:
            _mpc = np.array([[name, factor]
                             for name, factor in mpc[mpc_index].values],
                            dtype=object)
            _mpc_name = [_mpc[i, 0] for i in
                         sorted(np.unique(_mpc[:, 0], return_index=True)[1])]
            _mpc_dict = {name: index for name, index in
                         zip(_mpc_name,
                             _mpc_array[np.isin(_mpc_list, _mpc_name)])}
            self._mpc_referance[mpc_index] = [_mpc_dict[name]
                                              for name in _mpc[:, 0]]

            self._mpc_factor[mpc_index] = _mpc[:, 1]
        # link subcoil to coil referance
        if 'coil' in self:
            self._mpc_index = Index(self.loc[self._mpc_index, 'coil'])
        # construct multi-point link ()
        mpl = np.array([
            [referance, couple, factor] for couple, (referance, _mpc, factor)
            in enumerate(zip(self._mpc_referance, mpc, self._mpc_factor))
            if _mpc])
        if len(mpl) > 0:
            self._mpl_index = mpl[:, :2].astype(int)  # (refernace, couple)
            self._mpl_factor = mpl[:, 2]  # coupling factor
        else:
            self._mpl_index = []
            self._mpl_factor = []
        self._relink_mpc = True

    @property
    def _nC(self):
        """
        Return mpc coil number.

        Returns
        -------
        _nC : int
            Number of coils without mpc constraints.

        """
        return len(self._mpc_iloc)

    @property
    def _nI(self):
        """Return number of indexed coils."""
        return sum(self._current_index)


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
'''
