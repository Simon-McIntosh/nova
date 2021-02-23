"""AbstractBaseClass Extended Frame._methods."""
from abc import ABCMeta
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MetaMethod(metaclass=ABCMeta):
    """Manage Frame._methods."""

    frame: Any = field(init=False)
    key_attributes: list = field(init=False)
    additional_attributes: list[str] = field(init=False)
    enable: bool = field(init=False)

    def __post_init__(self):
        """Check for key_attribute in frame.columns."""
        self.update()

    def update(self):
        """Check for key_attribute in frame.columns, update additional."""
        self.enable = np.array([attr in self.frame.columns
                                for attr in self.key_attributes]).all()
        if self.enable:
            self.frame.metadata = {'additional': self.additional_attributes}
