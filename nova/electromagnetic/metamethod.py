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

    frame: Frame = field(repr=False)
    required: list[str]
    additional: list[str]
    require_all: bool = True

    def __post_init__(self):
        """Generate multi-point constraints."""
        if self.generate:
            self.update_additional()
            if not self.frame.empty:
                self.initialize()

    @abstractmethod
    def initialize(self):
        """Init method."""

    @property
    def generate(self):
        """Return initialization flag."""
        if self.require_all:
            return self.required_attributes.all()
        return self.required_attributes.any()

    @property
    def required_attributes(self):
        """Return boolean status of attributes found in frame.columns."""
        return np.array([attr in self.frame.metaframe.columns
                         for attr in self.required])

    def update_additional(self):
        """Update additional attributes if subset exsists in frame.columns."""
        self.frame.metadata = {'additional': self.additional}
        if not self.require_all:
            self.frame.metadata = {'additional': self.required}
