"""AbstractBaseClass Extended Frame._methods."""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nova.electromagnetic.frame import Frame


@dataclass
class MetaMethod(metaclass=ABCMeta):
    """Manage Frame._methods, subclass with dataclass."""

    frame: Frame
    key_attributes: list[str]
    additional_attributes: list[str]

    def __post_init__(self):
        """Generate multi-point constraints."""
        self.generate()

    @abstractmethod
    def generate(self):
        """Generate method attributes."""

    @property
    def enable(self) -> bool:
        """Return enable flag, update additional attributes if True."""
        update = np.array([attr in self.frame.columns
                           for attr in self.key_attributes]).any()
        if update:
            self.frame.metadata = {'additional': self.additional_attributes}
            return True
        return False
