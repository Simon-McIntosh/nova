
from dataclasses import dataclass, field

import numpy as np

from nova.electromagnetic.coilframe import CoilFrame


@dataclass
class MultiPointConstraint:

    frame: CoilFrame

    def __post_init__(self):
        if 'mpc' not in self.frame.metaframe.columns:
            self.frame.metaframe.metadata = {'additional': ['mpc']}
            self.frame['mpc'] = self.frame.metaframe.default['mpc']

    def extract(self):
        """Extract mpc interger index and factor."""
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


if __name__ == '__main__':

    frame = CoilFrame({'x': [1, 3], 'z': 0}, metadata={
        'Required': ['x', 'z'], 'Additional': []})
    mpc = MultiPointConstraint(frame)
    frame.add_frame(4, 5)
    print(frame)
