"""Manage matmul operations and svd reductions on BiotData."""
from dataclasses import dataclass, field, InitVar

import numba
import numpy as np
import xarray

from nova.biot.biotdata import BiotData
from nova.frame.framesetloc import ArrayLocIndexer

'''
@numba.njit(fastmath=True, parallel=True)
def matmul(A, B):
    """Perform fast matmul operation."""
    row_number = len(A)
    vector = np.empty(row_number, dtype=numba.float64)
    for i in numba.prange(row_number):  # pylint: disable=not-an-iterable
        vector[i] = np.dot(A[i], B)
    return vector
'''


@dataclass
class BiotOp:
    """Fast array opperations for Biot Data arrays."""

    aloc: ArrayLocIndexer
    saloc: ArrayLocIndexer
    dataset: InitVar[xarray.Dataset]

    def __post_init__(self, dataset):
        """Extract matrix, plasma_matrix and plasma_index from dataset."""
        data_vars = list(dataset.data_vars)
        self.matrix = dataset[data_vars[0]].data
        self.plasma_matrix = dataset[data_vars[1]].data
        self.plasma_index = dataset.attrs['plasma_index']

        '''
        #  perform svd order reduction
        self.svd_rank = min([len(plasma_s), svd_rank])

        # TODO fix svd_rank == -1 bug - crop plasma_U
        self.plasma_U = plasma_U.copy()#[:, :self.svd_rank].copy()
        self.plasma_s = plasma_s.copy()#[:self.svd_rank].copy()
        self.plasma_V = plasma_V.copy()#[:self.svd_rank, :].copy()
        '''

    def evaluate(self):
        """Return interaction."""
        return self.matrix @ self.saloc['Ic']

    @property
    def plasma_nturn(self):
        """Return plasma turns."""
        return self.aloc['nturn'][self.aloc['plasma']]

    def update_turns(self, svd=True):
        """Update plasma turns."""
        '''
        if svd:
            self.matrix[:, self.plasma_index] = self.plasma_U @ \
                (self.plasma_s * (self.plasma_V @ self.plasma_nturn))
            return
        print('svd == -1')
        '''
        self.matrix[:, self.plasma_index] = \
            self.plasma_matrix @ self.plasma_nturn


@dataclass
class BiotOperate(BiotData):
    """Multi-attribute interface to numba Biot Evaluate methods."""

    version: dict[str, int | None] = field(
        init=False, repr=False, default_factory=dict)
    _svd_rank: int = field(init=False, default=-1)
    operator: dict[str, BiotOp] = field(init=False, default_factory=dict,
                                        repr=False)
    target_number: int = field(init=False, default=0)
    array: dict = field(init=False, repr=False, default_factory=dict)

    @property
    def svd_rank(self):
        """Manage svd rank. Set to -1 to disable svd plasma turn update."""
        return self._svd_rank

    @svd_rank.setter
    def svd_rank(self, svd_rank: int):
        if svd_rank != self._svd_rank:
            self._svd_rank = svd_rank
            self.load_operators()

    @property
    def shape(self):
        """Return target shape."""
        return (self.target_number,)

    def post_solve(self):
        """Solve biot interaction - extened by subclass."""
        super().post_solve()
        self.load_operators()
        for attr in self.attrs:
            self.update_turns(attr)

    def load(self):
        """Extend netCDF load."""
        super().load()
        self.load_operators()

    def load_operators(self):
        """Link fast biot operators."""
        self.operator = {}
        if 'attributes' not in self.data.attrs:
            return
        self.attrs = self.data.attrs['attributes']
        for attr in self.attrs:
            self.operator[attr] = BiotOp(self.aloc, self.saloc,
                                         self.data[[attr, f'_{attr}']])
        self.load_version()
        self.load_arrays()

    def load_version(self):
        """Initialize biot version identifiers."""
        self.version |= {attr: self.data.attrs.get(attr, None)
                         for attr in self.attrs}
        self.version |= {attr.lower(): None for attr in self.attrs}
        if 'Br' in self.attrs and 'Bz' in self.attrs:
            self.version['bn'] = None

    def load_arrays(self):
        """Link data arrays."""
        self.target_number = self.data.dims['target']
        for attr in self.version:
            if attr.capitalize() in self.attrs or attr == 'bn':
                if attr.islower():
                    self.array[attr] = np.zeros(self.target_number)
                    if len(self.shape) == 1:
                        continue
                    ndarray = self.array[attr].reshape(self.shape)
                    self.array[f'{attr}_'] = ndarray
                    continue
                self.array[attr] = self.operator[attr].matrix

    def update_turns(self, Attr: str, svd=True):
        """Update plasma turns."""
        if self.data.attrs['plasma_index'] == -1:
            return
        self.operator[Attr].update_turns(svd)
        self.version[Attr] = self.data.attrs[Attr] = \
            self.subframe.version['nturn']
        self.version[Attr.lower()] = None

    def calculate_norm(self):
        """Return calculated L2 norm."""
        return np.linalg.norm([self.br, self.bz], axis=0)

    def get_norm(self):
        """Return cached field L2 norm."""
        if (version := self.aloc_hash['Ic']) != self.version['bn']:
            self.version['bn'] = version
            self.array['bn'][:] = self.calculate_norm()
        return self.array['bn']

    def __getattr__(self, attr):
        """Return variable data - lazy evaluation - cached."""
        attr = attr.replace('_field_', '')
        if attr.islower() and attr[-1] == '_':  # return shaped array
            if len(self.shape) == 1:
                return getattr(self, attr[:-1])
            self.array[attr][:] = getattr(self, attr[:-1]).reshape(self.shape)
            return self.array[attr]
        if attr not in (avalible := [attr for attr in self.version
                                     if attr.capitalize() in self.attrs
                                     or attr == 'bn']):
            raise AttributeError(f'Attribute {attr} '
                                 f'not defined in {avalible}.')
        if len(self.data) == 0:
            return self.array[attr]
        if attr == 'bn':
            return self.get_norm()
        Attr = attr.capitalize()
        if self.version[Attr] != self.subframe.version['nturn']:
            self.update_turns(Attr)
        if attr == Attr:
            return self.array[attr]
        if self.version[attr] != (version := self.aloc_hash['Ic']):
            self.version[attr] = version
            self.array[attr][:] = self.operator[Attr].evaluate()
        return self.array[attr]

    def __getitem__(self, attr):
        """Return array attribute via dict-like access."""
        return getattr(self, attr)
