"""AbstractBaseClass Extended Frame._methods."""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nova.electromagnetic.frame import Frame


@dataclass
class MetaMethod(metaclass=ABCMeta):
    """Manage Frame._methods, subclass with dataclass."""

    frame: Frame
    required: list[str]
    additional: list[str]

    def __post_init__(self):
        """Generate multi-point constraints."""
        self.update()
        self.initialize()

    @abstractmethod
    def initialize(self):
        """Init method."""

    @property
    def required_attributes(self):
        """Return boolean status of attributes found in frame.columns."""
        return np.array([attr in self.frame.columns
                         for attr in self.required])

    def update(self):
        """Update additional attributes if subset exsists in frame.columns."""
        if self.enable:
            self.frame.metadata = {'additional': self.additional}

    @property
    def enable(self):
        """Return status of required attributes in frame.columns."""
        return self.required_attributes.all()
