
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas

from nova.electromagnetic.metadata import MetaData

if TYPE_CHECKING:
    from nova.electromagnetic.frame import Frame


@dataclass
class MetaConstraint(MetaData):
    """Manage Frame metadata - accessed via Frame['attrs']."""

    constrain: list[str] = field(default_factory=lambda: ['Ic'])


@dataclass
class MultiPoint:

    frame: Frame

    '''
    iloc: list[int] = field(init=False)
    index: pandas.Index = field(init=False)
    referance: list[int] = field(init=False)
    factor: list[float] = field(init=False)
    link_index: list[int, int] = field(init=False)
    link_factor: list[float] = field(init=False)
    '''

    def __post_init__(self):
        """Configure frame for multi-point constraints."""
        #if 'mpc' not in self.frame.metaframe.columns:
        #    self.frame.add_column('mpc')

    def link(self):
        """
        Apply multi-point constraints to frame.

            - if 'mpc' in self.columns
                Format mpc attribute:

                - mpc is none or NaN: mpc = ''
                - mpc is bool: mpc = '' if False else 1
                - mpc is int or float: mpc[0] = '', mpc[1:] = factor

            - elif self.link
                Add mpc attrbute and set mpc vector to Ture

            - else
                mpc unset.

        """
        if 'mpc' in self.frame.columns:
            isnan = np.array([pandas.isna(mpc)
                              for mpc in self.frame.mpc], dtype=bool)
            self.frame.loc[isnan, 'mpc'] = self.frame.metaframe.default['mpc']
            isnumeric = np.array([isinstance(mpc, (int, float)) &
                                  ~isinstance(mpc, bool)
                                  for mpc in self.frame.mpc], dtype=bool)
            istrue = np.array([mpc is True for mpc in self.frame.mpc],
                              dtype=bool)
            istuple = np.array([isinstance(mpc, tuple)
                                for mpc in self.frame.mpc], dtype=bool)
            self.frame.loc[~istrue & ~isnumeric & ~istuple, 'mpc'] = ''
            index = self.frame.index[istrue | isnumeric]
            if index.empty:
                return
            factor = np.ones(len(self.frame))
            factor[isnumeric] = self.frame.mpc[isnumeric]
            factor = factor[istrue | isnumeric][1:]
            if len(index) > 1:
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
        index_number = len(index)
        if index_number == 1:
            raise IndexError(f'len({index}): {index_number} '
                             'is not greater > 1')
        if not pandas.api.types.is_list_like(factor):
            factor = factor * np.ones(index_number-1)
        elif len(factor) != index_number-1:
            raise IndexError(f'len(factor={factor}) must == 1 '
                             f'or == len(index={index})-1')
        self.frame.at[index[0], 'mpc'] = ''
        for i in np.arange(1, index_number):
            self.frame.at[index[i], 'mpc'] = (index[0], factor[i-1])
        #self.rebuild_CoilArray()

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


    '''
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
    '''
