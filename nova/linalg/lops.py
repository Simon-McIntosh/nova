"""Extend pylops regression class."""

from dataclasses import dataclass

from pylops import LinearOperator
from pylops.optimization.leastsquares import RegularizedInversion

from nova.linalg.regression import RegressionBase


@dataclass
class Lops(RegressionBase, LinearOperator):
    """Extend Pylops linear operator and Nova regression classes."""

    dtype: type = float
    explicit: bool = True

    def __post_init__(self):
        """Link matrix attribute to Pylops LinearOperator.A."""
        LinearOperator.__init__(self)
        super().__post_init__()
        self.A = self.matrix

    def _matvec(self, model):
        """Return results of forward calculation."""
        return self.forward(model)

    def _rmatvec(self, data):
        """Return results of adjoint calculation."""
        return self.adjoint(data)

    def _inverse(self):
        """Retun solution to least squares problem using default solver."""
        return RegularizedInversion(self).solve(self.data, None)[0]
