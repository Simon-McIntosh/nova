"""Manage 1D basis functions."""

import abc
from dataclasses import dataclass, field
from functools import cached_property
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from nova.graphics.plot import Plot1D
from nova.jax.tree_util import Pytree


@dataclass
@jax.tree_util.register_pytree_node_class
class Basis(Pytree):
    """Evaluate and plot interpolant."""

    coordinate: jnp.ndarray = field(repr=False, default_factory=lambda: jnp.array([]))
    data: jnp.ndarray = field(repr=False, default_factory=lambda: jnp.array([]))

    @jax.jit
    def __call__(self, coordinate: jnp.ndarray):
        """Evaluate forward model."""
        return self.forward(coordinate)

    @abc.abstractmethod
    def forward(self, coordinate: jnp.ndarray):
        """Evaluate forward model."""

    @cached_property
    def plot_coordinate(self):
        """Return high resolution plot coordinate."""
        if (ncoord := len(self.coordinate)) > 0:
            return np.linspace(self.coordinate[0], self.coordinate[-1], 5 * ncoord)
        return np.linspace(0, 1)

    def plot(self, axes=None):
        """Plot model."""
        axes = Plot1D().set_axes(axes=axes)
        if len(self.coordinate) > 0 and len(self.data > 0):
            axes.plot(self.coordinate, self.data, "o", label="data")
        axes.plot(self.plot_coordinate, self(self.plot_coordinate), "-", label="fit")
        axes.set_xlabel("coordinate")
        axes.set_ylabel("value")
        axes.legend()
        return axes

    def tree_flatten(self):
        """Return flattened pytree."""
        children, aux_data = super().tree_flatten()
        children += (self.coordinate, self.data)
        return (children, aux_data)


@dataclass
@jax.tree_util.register_pytree_node_class
class Fit(Pytree):
    """Fit interpolant to data."""

    coordinate: jnp.ndarray = field(repr=False)

    def __truediv__(self, data):
        """Solve inverse model given data."""
        return self.inverse(data)

    @abc.abstractmethod
    def inverse(self, data):
        """Calculate inverse and update model coeffcents."""

    def tree_flatten(self):
        """Return flattened pytree."""
        children, aux_data = super().tree_flatten()
        children += (self.coordinate,)
        return (children, aux_data)


@dataclass
@jax.tree_util.register_pytree_node_class
class Linear(Basis, Pytree):
    """Evaluate linear model."""

    @jax.jit
    def forward(self, coordinate: jnp.ndarray):
        """Evaluate forward model."""
        return jax.numpy.interp(coordinate, self.coordinate, self.data)


@dataclass
@jax.tree_util.register_pytree_node_class
class Interp(Fit):
    """Fit linear interpolant to data."""

    @jax.jit
    def inverse(self, data) -> Basis:
        """Return interpolating polynomial."""
        return Linear(self.coordinate, data)


@dataclass(kw_only=True)
@jax.tree_util.register_pytree_node_class
class Bernstein(Pytree):
    """Generate Bernstein basis functions."""

    order: int

    @jax.jit
    def binom(self, term):
        """Return Binomial cooefcient (order term)."""
        return jsp.special.gamma(self.order + 1) / (
            jsp.special.gamma(term + 1) * jsp.special.gamma(self.order - term + 1)
        )

    @jax.jit
    def basis(self, coordinate: jnp.ndarray, term: int):
        """Return Bernstein basis polynomial."""
        return (
            self.binom(term)
            * coordinate**term
            * (1 - coordinate) ** (self.order - term)
        )

    @jax.jit
    def coefficent_matrix(self, coordinate: jnp.ndarray):
        """Return coefficent matrix."""

        def basis(_, term):
            return _, self.basis(coordinate, term)

        return jax.lax.scan(basis, None, jnp.arange(self.order + 1))[1].T

    def tree_flatten(self):
        """Return flattened pytree."""
        children = ()
        aux_data = {"order": self.order}
        return (children, aux_data)


@dataclass
@jax.tree_util.register_pytree_node_class
class Polynomial(Basis, Bernstein):
    """Interpolating Bernstein polynomial."""

    model: jnp.ndarray = field(repr=False, default_factory=lambda: jnp.array([]))

    @jax.jit
    def forward(self, coordinate: jnp.ndarray):
        """Evaluate forward model."""

        def product(result, term):
            return result + self.basis(coordinate, term) * self.model[term], None

        init = jnp.zeros_like(coordinate)
        return jax.lax.scan(product, init, jnp.arange(self.order + 1))[0]

    def plot_basis(self, axes=None, **kwargs):
        """Plot basis functions."""
        axes = Plot1D().set_axes(axes=axes)
        basis = np.stack(
            [
                coef * self.basis(self.plot_coordinate, i)
                for i, coef in enumerate(self.model)
            ],
            axis=1,
        )
        kwargs = {"color": "gray", "lw": 1.5} | kwargs
        axes.plot(self.plot_coordinate, basis[:, 0], **kwargs, label="basis")
        axes.plot(self.plot_coordinate, basis[:, 1:], **kwargs)
        return axes

    def plot(self, *, axes=None, basis=False):
        """Plot model."""
        if basis:
            axes = self.plot_basis(axes)
        return super().plot(axes)

    def tree_flatten(self):
        """Return flattened pytree."""
        children, aux_data = super().tree_flatten()
        children += (self.model,)
        return (children, aux_data)


@dataclass
@jax.tree_util.register_pytree_node_class
class BSpline(Fit, Bernstein):
    """Fit a Bernstein polynomial to data."""

    @cached_property
    def matrix(self):
        """Return coefficent matrix."""
        return self.coefficent_matrix(self.coordinate)

    @jax.jit
    def inverse(self, data) -> Basis:
        """Return interpolating polynomial."""
        model = jnp.linalg.lstsq(self.matrix, data, rcond=None)[0]
        return Polynomial(self.coordinate, data, model, order=self.order)


if __name__ == "__main__":

    bspline = BSpline(x := jnp.linspace(0, 1, 21), order=7)
    data = -(3 * x**5) + 3 * x**2 + -4 * x + 4
    poly = bspline / data
    poly.plot(basis=True)

    linear = Interp(x) / data
    linear.plot()
