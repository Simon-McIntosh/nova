#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 10:39:04 2022

@author: mcintos
"""


@numba.njit(fastmath=True, parallel=True)
def matmul(A, B):
    row_number = len(A)
    vector = np.empty(row_number)
    for i in numba.prange(row_number):
        vector[i] = np.dot(A[i], B)
    return vector


@numba.experimental.jitclass(dict(
    current=float64[::1], nturn=float64[::1], plasma=boolean[::1],
    matrix=float64[:, ::1], _matrix=float64[:, ::1],
    _U=float64[:, ::1], _s=float64[::1], _V=float64[:, ::1]))
class BiotEval:
    """Fast array opperations for Biot Data arrays."""

    plasma_index: int

    def __init__(self, current, nturn, plasma, plasma_index, matrix, _matrix,
                 _U, _s, _V):
        self.current = current
        self.nturn = nturn
        self.plasma = plasma
        self.plasma_index = plasma_index
        self.matrix = matrix
        self._matrix = _matrix

        rank = len(_s) / 1

        self._U = _U[:, :rank].copy()
        self._s = _s[:rank].copy()
        self._V = _V[:rank, :].copy()

        '''
        U, s, V = np.linalg.svd(self._matrix, True)
        rank = len(s) / 1
        self._U = U[:, :rank].copy()
        self._s = s[:rank].copy()
        self._V = V[:rank, :].copy()
        '''

        #self._U, self._s, self._V = \
        #    np.linalg.svd(self._matrix, full_matrices=False)

    '''
    def update_turns()
        nturn = self.aloc['nturn'][self.aloc['plasma']]
        index = self.plasma_index
        if solver == 'cpu':
            self.array[attr][:, index] = self.array[f'_{attr}'] @ nturn
            return
        if solver == 'jit':
            self.array[attr][:, index] = self._update_turns(
                self.array[f'_{attr}'], nturn)
            return

        @staticmethod
        @numba.njit(parallel=True)
        def _update_turns(matrix, nturn):

            for i in numba.prange(self.row_number):
                vector[i] = np.dot(matrix[i], nturn)
            return vector
    '''

    def update_turns(self):
        """Update plasma turns."""
        #self.matrix[:, self.plasma_index] = \
        #    self._matrix @ self.nturn[self.plasma]

        self.matrix[:, self.plasma_index] = matmul(
            self._matrix, self.nturn[self.plasma])

        #self.matrix[:, self.plasma_index] = \
        #    self._U @ (self._s * (self._V @ self.nturn[self.plasma]))

        #matrix = self._V @ self.nturn[self.plasma]
        #matrix *= self._s
        #self.matrix[:, self.plasma_index] = self._U @ matrix

    def matmul(self):
        return self._matrix @ self.nturn[self.plasma]

    def svd(self):
        return self._U @ (self._s * (self._V @ self.nturn[self.plasma]))


        #row_number = len(self._matrix)
        #nturn = self.nturn[self.plasma]
        #for i in numba.prange(row_number):
        #    self.matrix[i, self.plasma_index] = self._matrix[i].dot(nturn)

        #col_number = self._matrix.shape[1]
        ##nturn = self.nturn[self.plasma]
        #for i in numba.prange(col_number):
        #    if nturn[i] == 0:
        #        continue
        #    self.matrix[:, self.plasma_index] += nturn[i] * self._matrix[:, i]

        '''
        col_number = self._matrix.shape[0]
        vector = np.zeros(col_number)
        nturn = self.nturn[self.plasma]
        for i in numba.prange(col_number):
            print(nturn[i])
            vector += (self._matrix[:, i] * nturn[i])
        self.matrix[:, self.plasma_index] = vector
        '''

    def evaluate(self):
        """Return interaction."""
        return self.matrix @ self.current


@dataclass
class BiotAttrs:
    """Multi-attribute interface to numba Biot Evaluate methods."""

    current: npt.ArrayLike
    nturn: npt.ArrayLike = field(repr=False)
    plasma: npt.ArrayLike = field(repr=False)
    data: xarray.Dataset = field(repr=False)
    calc: dict[str, BiotEval] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Extract attrs for dataset + update calculation interface."""
        for attr in self.data.attrs['attributes']:
            self.calc[attr] = BiotEval(
                self.current,
                self.nturn,
                self.plasma,
                self.data.attrs['plasma_index'],
                self.data[attr].data,
                self.data[f'_{attr}'].data,
                self.data[f'_U{attr}'].data,
                self.data[f'_s{attr}'].data,
                self.data[f'_V{attr}'].data)

    def __getitem__(self, key):
        return self.calc[key]

    def __getattr__(self, attr):
        return self.calc[attr]

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
        #'''
        nturn = self.aloc['nturn'][self.aloc['plasma']]
        index = self.plasma_index
        #self._array[attr][:, index] = \
        #    np.einsum('ij,j', self._array[f'_{attr}'][:, turn_index],
        #              nturn[turn_index])

        #self._array[attr][:, index] = np.dot(self._array[f'_{attr}'], nturn)
        self._array[attr][:, index] = self._array[f'_{attr}'] @ nturn
        '''
        if solver == 'cpu':
            self._array[attr][:, index] = self._array[f'_{attr}'] @ nturn
            return
        if solver == 'jit':
            self._array[attr][:, index] = self._update_turns(
                self._array[f'_{attr}'], nturn)
            return
        '''
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
