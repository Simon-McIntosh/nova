"""Manage Berstein polynomials."""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.special

from nova.utilities.pyplot import plt


@dataclass
class Bernstein:
    """Generate 1D Berstein polynomials of a given order."""

    length: int
    degree: int
    coordinate: npt.ArrayLike = field(init=False, repr=False)
    matrix: npt.ArrayLike = field(init=False, repr=False)

    def __post_init__(self):
        """Generate linear spaced coordinate array from space parameter."""
        self.coordinate = np.linspace(0, 1, self.length)
        self.matrix = np.c_[[self.basis(i) for i in range(self.degree+1)]].T

    def basis(self, term: int):
        """Return Bernstein basis polynomial, 0<=term<degree."""
        assert term >= 0 & term <= self.degree
        return scipy.special.binom(self.degree, term) * \
            self.coordinate**term * (1 - self.coordinate)**(self.degree - term)

    def plot(self):
        """Plot set of Berstein basis polynomials."""
        for i in range(self.degree+1):
            plt.plot(self.coordinate, self.basis(i))


if __name__ == '__main__':

    berstein = Bernstein(501, 13)
    berstein.plot()
