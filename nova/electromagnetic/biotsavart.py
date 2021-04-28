"""Biot-Savart calculation base class."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import numpy.typing as npt

from nova.electromagnetic.biotset import BiotSet


@dataclass
class BiotSavart(ABC, BiotSet):
    """Biot calculation base-class, Define calculaiton interface."""

    name = 'base'

    update: Union[str, list[str]] = field(default_factory=lambda: ['psi'])
    attrs: list[str] = field(init=False, default_factory=lambda: [
        'phi', 'psi', 'radial_field', 'vertical_field'])
    phi: npt.ArrayLike = field(init=False, repr=False, default=None)
    psi: npt.ArrayLike = field(init=False, repr=False, default=None)
    radial_field: npt.ArrayLike = field(init=False, repr=False, default=None)
    vertical_field: npt.ArrayLike = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Check update vector."""
        super().__post_init__()
        if isinstance(self.update, str):
            self.update = [self.update]
        self.check_update()

    def check_update(self):
        """Check avalibility of update attributes."""
        if any(notimplemented := [attr not in self.attrs
                                  for attr in self.update]):
            raise NotImplementedError(
                f'update fields {np.array(self.update)[notimplemented]} '
                'not implemented')

    @abstractmethod
    def calculate(self):
        """Calculate dependant variables."""
