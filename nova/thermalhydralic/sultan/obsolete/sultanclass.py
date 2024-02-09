"""Generic Sultan DataClass methods."""

import numpy as np


class Trigger:
    """Reload abstract base class."""

    @property
    def trigger(self):
        """Manage trigger status."""
        return np.fromiter(vars(self).values(), dtype=bool).any()

    @trigger.setter
    def trigger(self, status):
        for attribute in self.__dict__:
            setattr(self, attribute, status)


class SultanClass:
    """Provide generic methods to Sultan classes."""

    def __repr__(self):
        """Return string representation of dataclass."""
        _vars = vars(self)
        attributes = ", ".join(
            f"{name.replace('_', '')}={_vars[name]!r}" for name in _vars
        )
        return f"{self.__class__.__name__}({attributes})"
