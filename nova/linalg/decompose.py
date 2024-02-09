"""Manage singluar value decomposition."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Decompose:
    """Provide access to svd methods."""

    matrix: np.ndarray
    rank: int = 0
    matrices: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        """Calculate svd decomposition."""
        self.initialize_rank()
        self.decompose()

    @property
    def shape(self):
        """Return matrix shape."""
        return self.matrix.shape

    def __getitem__(self, key: str):
        """Return array from matrices dict."""
        return self.matrices[key]

    def initialize_rank(self):
        """Init svd rank."""
        full_rank = min(self.shape)
        if self.rank == 0:
            self.rank = full_rank
        if self.rank < 0:
            self.rank += full_rank
            if self.rank <= 0:
                raise ValueError(
                    f"rank {full_rank-self.rank} >= full rank " f"{full_rank}"
                )
        self.rank = min(self.rank, full_rank)

    def decompose(self):
        """Perform SVD order reduction."""
        UsVh = np.linalg.svd(self.matrix, full_matrices=False)
        self.matrices = dict(U=UsVh[0].copy("C"), s=UsVh[1], Vh=UsVh[2].copy("C"))
        self.reduce()
        self.transpose()
        assert (self.matrices["s"] > 0).all()

    def transpose(self):
        """Transpose derived svd arrays."""
        self.matrices |= dict(
            Uh=self.matrices["U"].T.copy(order="C"),
            V=self.matrices["Vh"].T.copy(order="C"),
        )

    def reduce(self):
        """Apply rank reduction to svd data."""
        if self.rank < min(self.shape):
            self.matrices["U"] = self.matrices["U"][:, : self.rank]
            self.matrices["s"] = self.matrices["s"][: self.rank]
            self.matrices["Vh"] = self.matrices["Vh"][: self.rank, :]
