"""Manage matmul operations and svd reductions on BiotData."""
from abc import abstractmethod
from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt

from nova.electromagnetic.biotdata import BiotData


@numba.njit(fastmath=True, parallel=True)
def matmul(A, B):
    """Perform fast matmul operation."""
    row_number = len(A)
    vector = np.empty(row_number, dtype=numba.float64)
    for i in numba.prange(row_number):
        vector[i] = np.dot(A[i], B)
    return vector


@numba.experimental.jitclass(dict(
    current=numba.float64[::1],
    nturn=numba.float64[::1],
    plasma=numba.boolean[::1],
    plasma_index=numba.int32,
    matrix=numba.float64[:, ::1],
    plasma_matrix=numba.float64[:, ::1],
    plasma_U=numba.float64[:, ::1],
    plasma_s=numba.float64[::1],
    plasma_V=numba.float64[:, ::1]))
class BiotOp:
    """Fast array opperations for Biot Data arrays."""

    def __init__(self, current, nturn, plasma,
                 plasma_index, matrix, plasma_matrix,
                 svd_factor, plasma_U, plasma_s, plasma_V):
        self.current = current
        self.nturn = nturn
        self.plasma = plasma
        self.plasma_index = plasma_index
        self.matrix = matrix
        self.plasma_matrix = plasma_matrix
        #  perform svd order reduction
        rank = int(len(plasma_s) / svd_factor)
        print(rank)
        self.plasma_U = plasma_U[:, :rank].copy()
        self.plasma_s = plasma_s[:rank].copy()
        self.plasma_V = plasma_V[:rank, :].copy()

    def evaluate(self):
        """Return interaction."""
        return matmul(self.matrix, self.current)

    def update_turns(self, svd=True):
        """Update plasma turns."""
        if svd:
            self.matrix[:, self.plasma_index] = \
                matmul(self.plasma_U,
                       self.plasma_s * matmul(self.plasma_V,
                                              self.nturn[self.plasma]))
            return
        self.matrix[:, self.plasma_index] = matmul(
            self.plasma_matrix, self.nturn[self.plasma])


@dataclass
class BiotOperate(BiotData):
    """Multi-attribute interface to numba Biot Evaluate methods."""

    svd_factor: float = 6.
    version: dict[str, int] = field(
        init=False, repr=False, default_factory=dict)
    operator: dict[str, BiotOp] = field(init=False, default_factory=dict)

    #array: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Initialize version identifiers."""
        self.version |= {attr: id(None) for attr in self.attrs}
        self.version['Bn'] = id(None)
        super().__post_init__()

    @abstractmethod
    def solve(self, *args):
        """Solve biot interaction - extened by subclass."""
        super().solve()
        self.load_operators()
        #self.link_array()

    def load(self, file: str, path=None):
        """Extend BiotData load."""
        super().load(file, path)
        self.load_operators()
        #self.link_array()

    def load_operators(self, svd_factor=None):
        """Load fast biot operators."""
        if svd_factor is not None:
            self.svd_factor = svd_factor
        self.x_coordinate = self.data.x.data
        self.z_coordinate = self.data.z.data
        for attr in self.data.attrs['attributes']:
            self.operator[attr] = BiotOp(
                self.saloc['Ic'], self.aloc['nturn'], self.aloc['plasma'],
                self.data.attrs['plasma_index'],
                self.data[attr].data, self.data[f'_{attr}'].data,
                self.svd_factor, self.data[f'_U{attr}'].data,
                self.data[f'_s{attr}'].data, self.data[f'_V{attr}'].data)

    ##@@property
    #def bnorm(self):
    #    """Return L2 norm of poloidal magnetic field."""

    '''
    def link_array(self):
        """Update array attributes."""
        for attr in self.data.attrs['attributes']:
            self.array[attr] = self.data[attr].data
            self.array[f'_{attr}'] = self.data[f'_{attr}'].data
            rank = int(len(self.data[f'_s{attr}']) / self.svd_factor)
            self.array[f'_U{attr}'] = self.data[f'_U{attr}'].data[:, :rank]
            self.array[f'_s{attr}'] = self.data[f'_s{attr}'].data[:rank]
            self.array[f'_V{attr}'] = self.data[f'_V{attr}'].data[:rank, :]
        self.update_indexer()
        try:
            self.plasma_index = next(
                self.frame.subspace.index.get_loc(name) for name in
                self.subframe.frame[self.aloc['plasma']].unique())
        except StopIteration:
            pass

    def link_data(self):
        """Update data attributes."""
        for attr in self.data.attrs['attributes']:
            if attr[0] == '_':
                continue
            self.data[attr].data = self.array[attr]

    '''

    def __getattr__(self, attr):
        """Return variable data."""
        if (Attr := attr.capitalize()) in self.version:
            self.update_indexer()
            if Attr == 'Bn':
                return self.get_norm()
            if self.version[Attr] != self.subframe.version['plasma']:
                self.update_turns(Attr)
                self.version[Attr] = self.subframe.version['plasma']
            #return self.array[Attr] @ self.current
            return self.operator[Attr].evaluate()
        raise AttributeError(f'attribute {Attr} not specified in {self.attrs}')

    def update_turns(self, attr: str, svd=True):
        """Update plasma turns."""
        if self.data.attrs['plasma_index'] is None:
            return
        self.operator[attr].update_turns(svd)
        '''
        nturn = self.aloc['nturn'][self.aloc['plasma']]
        index = self.plasma_index
        if svd:
            self.array[attr][:, index] = \
                self.array[f'_U{attr}'] @ (self.array[f'_s{attr}'] *
                                           (self.array[f'_V{attr}'] @ nturn))
            return
        self.array[attr][:, index] = self.array[f'_{attr}'] @ nturn
        #self.array[attr][:, index] = self.matmul(self.array[f'_{attr}'], nturn)
        '''

    def get_norm(self):
        """Return cached field L2 norm."""
        version = hash(self.current.data.tobytes())
        if self.version['Bn'] != version or 'Bn' not in self.array:
            self.array['Bn'] = self.calculate_norm()
            self.version['Bn'] = version
        return self.array['Bn']

    def calculate_norm(self):
        """Return calculated L2 norm."""
        return np.linalg.norm([self.Br, self.Bz], axis=0)

'''

    @staticmethod
    @numba.njit(parallel=True)
    def _update_turns(matrix, nturn):
        row_number, col_number = matrix.shape
        vector = np.empty(row_number)
        for j in numba.prange(col_number):
            if nturn[j] == 0:
                continue
            vector += nturn[j] * matrix[:, j]
        return vector

    def update_turns(self, attr: str, solver='cpu'):
        """Update plasma turns."""
        if self.data.attrs['plasma_index'] is None:
            return
        #self.biotattrs[attr].update_turns()

        nturn = self.aloc['nturn'][self.aloc['plasma']]
        index = self.plasma_index
        #self._array[attr][:, index] = \
        #    np.einsum('ij,j', self._array[f'_{attr}'][:, turn_index],
        #              nturn[turn_index])

        #self._array[attr][:, index] = np.dot(self._array[f'_{attr}'], nturn)
        self._array[attr][:, index] = self._array[f'_{attr}'] @ nturn

        if solver == 'cpu':
            self._array[attr][:, index] = self._array[f'_{attr}'] @ nturn
            return
        if solver == 'jit':
            self._array[attr][:, index] = self._update_turns(
                self._array[f'_{attr}'], nturn)
            return

        #raise NotImplementedError(f'solver <{solver}> not implemented')


    def update(self):
        """Update data attributes."""
        for attr in self.data.attrs['attributes']:
            self._array[attr] = self.data[attr].data
            self._array[f'_{attr}'] = self.data[f'_{attr}'].data
        self.update_indexer()
        self.plasma_index = next(
            self.frame.subspace.index.get_loc(name) for name in
            self.subframe.frame[self.aloc.plasma].unique())

        try:
            self.data.attrs['plasma_index'] = next(
                self.frame.subspace.index.get_loc(name) for name in
                self.subframe.frame[self.aloc.plasma].unique())
            #self.biotattrs = BiotAttrs(self.current, self.aloc['nturn'],
            #                           self.aloc['plasma'], self.data)
        except StopIteration:
            pass

            def __getattr__(self, attr):
                """Return variales data."""
                if (Attr := attr.capitalize()) in self.attrs:
                    self.update_indexer()
                    if Attr == 'Bn':
                        return self.get_norm()
                    if self.version[Attr] != self.subframe.version['plasma']:
                        self.update_turns(Attr)
                        self.version[Attr] = self.subframe.version['plasma']
                    return self._array[Attr] @ self.current
                    #return self._biot.evaluate()
                    #return self.biotattrs[Attr].evaluate()
                raise AttributeError(f'attribute {Attr} not specified '
                                     f'in {self.attrs}')
'''
