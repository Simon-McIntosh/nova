"""Provide IdsData baseclass."""
from abc import ABC
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Attrs(ABC):
    """Methods for updating ids attributes."""

    attributes: ClassVar[list[str]] = []

    def __call__(self, ids: object):
        """Update code metadata."""
        for attr in self.attributes:
            try:
                attribute = getattr(self, attr)
            except AttributeError:
                continue
            if attribute is None:
                continue
            setattr(ids, attr, attribute)
