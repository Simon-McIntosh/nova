"""Manage matmul operations and svd reductions on BiotData."""
from abc import abstractmethod
from dataclasses import dataclass, field

import numba
import numpy as np

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
        rank = int(np.ceil(len(plasma_s) / svd_factor))
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

    svd_factor: float = 10.
    version: dict[str, int] = field(
        init=False, repr=False, default_factory=dict)
    operator: dict[str, BiotOp] = field(init=False, default_factory=dict,
                                        repr=False)

    def __post_init__(self):
        """Initialize version identifiers."""
        self.version |= {attr: id(None) for attr in self.attrs}
        self.version['Bn'] = id(None)
        self.version['null'] = id(None)
        super().__post_init__()

    @abstractmethod
    def solve(self, *args):
        """Solve biot interaction - extened by subclass."""
        super().solve()
        self.load_operators()

    def load(self, file: str, path=None):
        """Extend BiotData load."""
        super().load(file, path)
        self.load_operators()

    def load_operators(self, svd_factor=None):
        """Load fast biot operators."""
        if svd_factor is not None:
            self.svd_factor = svd_factor
        self.x_coordinate = self.data.x.data
        self.z_coordinate = self.data.z.data
        self.update_loc_indexer()
        for attr in self.data.attrs['attributes']:
            self.operator[attr] = BiotOp(
                self.saloc['Ic'], self.aloc['nturn'], self.aloc['plasma'],
                self.data.attrs['plasma_index'],
                self.data[attr].data, self.data[f'_{attr}'].data,
                self.svd_factor, self.data[f'_U{attr}'].data,
                self.data[f'_s{attr}'].data, self.data[f'_V{attr}'].data)

    def __getattr__(self, attr):
        """Return variable data."""
        if (Attr := attr.capitalize()) in self.version:
            if Attr == 'Bn':
                return self.get_norm()
            if self.version[Attr] != self.subframe.version['plasma'] or \
                    self.version[Attr] == self.version['null']:
                self.update_turns(Attr)
                self.version[Attr] = self.subframe.version['plasma']
            return self.operator[Attr].evaluate()
        raise AttributeError(f'attribute {Attr} not specified in {self.attrs}')

    def update_turns(self, attr: str, svd=True):
        """Update plasma turns."""
        if self.data.attrs['plasma_index'] == -1:
            return
        self.operator[attr].update_turns(svd)

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
