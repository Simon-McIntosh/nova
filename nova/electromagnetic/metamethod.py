"""AbstractBaseClass Extended FrameSpace._methods."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas


@dataclass
class MetaMethod(ABC):
    """Manage DataFrame methods."""

    frame: pandas.DataFrame = field(repr=False)
    required: list[str] = field(default_factory=list)
    additional: list[str] = field(default_factory=list)
    require_all: bool = True

    def __post_init__(self):
        """Generate multi-point constraints."""
        if self.generate:
            self.update_additional()

    @abstractmethod
    def initialize(self):
        """Init metamethod."""

    @property
    def generate(self):
        """Return True if required attributes set else False."""
        if not self.required:  # required attributes empty
            return True
        if self.require_all:
            return self.required_attributes.all()
        return self.required_attributes.any()

    @property
    def required_attributes(self):
        """Return boolean status of attributes found in frame.columns."""
        return np.array([attr in self.frame
                         or attr in self.frame.metaframe.available
                         for attr in self.required])

    def unset(self, attributes: list[str]) -> list[str]:
        """Return unset attributes."""
        return [attr for attr in list(dict.fromkeys(attributes))
                if attr not in self.frame.metaframe.columns]

    def update_additional(self):
        """Update additional attributes if subset exsists in frame.columns."""
        additional = self.unset(self.required + self.additional)
        if not self.require_all:
            additional.extend(self.unset(self.required))
        if additional:
            self.frame.metaframe.metadata = {'additional': additional}

    def update_available(self, attrs):
        """Update metaframe.available if attrs unset and available."""
        additional = [attr for attr in self.unset(attrs)
                      if attr in self.frame.metaframe.available]
        if additional:
            self.frame.metaframe.metadata = {'additional': additional}
