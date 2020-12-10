"""Generic Sultan DataClass methods."""
# pylint: disable=no-member


class SultanClass:
    """Provide generic methods to Sultan classes."""

    def __repr__(self):
        """Return string representation of dataclass."""
        _vars = vars(self)
        attributes = ", ".join(f"{name.replace('_', '')}={_vars[name]!r}"
                               for name in _vars)
        return f"{self.__class__.__name__}({attributes})"
