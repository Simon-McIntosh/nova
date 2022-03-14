"""Manage singluar value decomposition."""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class Decompose:
    """Provide svd methods."""

    matrix: npt.ArrayLike = None
    shape: tuple[int, int] = None
    svd: bool = True
    rank: int = None
    _svd: dict = field(init=False, repr=False)

    def __post_init__(self):
        """Calculate svd decomposition."""
        super().__post_init__()
        self.initialize_rank()
        self.decompose()

    def initialize_rank(self):
        """Init svd rank."""
        if self.rank == 0:
            raise ValueError('rank == 0, set rank=None for full rank.')
        full_rank = min(self.shape)
        if self.rank is None:
            self.rank = full_rank
        if self.rank < 0:
            self.rank += full_rank
            if self.rank <= 0:
                raise ValueError(f'rank {full_rank-self.rank} >= full rank '
                                 f'{full_rank}')
        self.rank = min(self.rank, full_rank)

    def decompose(self):
        """Perform SVD order reduction."""
        UsVh = np.linalg.svd(self.matrix, full_matrices=False)
        self._svd = dict(U=UsVh[0], s=UsVh[1], Vh=UsVh[2])
        self.reduce()
        self.transpose()
        assert (self._svd['s'] > 0).all()

    def transpose(self):
        """Transpose derived svd arrays."""
        self._svd |= dict(Uh=self._svd['U'].T, V=self._svd['Vh'].T)

    def reduce(self):
        """Apply rank reduction to svd data."""
        if self.rank < min(self.shape):
            self._svd['U'] = self._svd['U'][:, :self.rank]
            self._svd['s'] = self._svd['s'][:self.rank]
            self._svd['Vh'] = self._svd['Vh'][:self.rank, :]
