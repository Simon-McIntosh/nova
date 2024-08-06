"""Manage equilibrium flux functions."""

from functools import cached_property
from typing import Callable, ClassVar


class Flux:
    """Manage equilibrium flux functions."""

    flux_attrs: ClassVar[list[str]] = ["p_prime", "ff_prime"]

    def fluxfunctions(self, attr) -> Callable:
        """Return flux function interpolant for requested attr."""
        raise NotImplementedError(f"{attr} interpolant is not set")

    @cached_property
    def p_prime(self) -> Callable:
        """Return p_prime interpolant."""
        return self.fluxfunctions("p_prime")

    @cached_property
    def ff_prime(self) -> Callable:
        """Return ff_prime interpolant."""
        return self.fluxfunctions("ff_prime")

    def _clear_flux_function_cache(self):
        """Clear flux function cached properties."""
        for attr in self.flux_attrs:
            try:
                delattr(self, f"{attr}")
            except AttributeError:
                pass

    def update(self):
        """Clear flux function cache."""
        self._clear_flux_function_cache()
        if hasattr(super(), "update"):
            super().update()
