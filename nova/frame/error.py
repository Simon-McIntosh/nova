"""Collection of frameset error classes."""


class ColumnError(IndexError):
    """Prevent column creation."""

    def __init__(self, name):
        super().__init__(
            "Column access via a new attribute name " f"{name} is not allowed."
        )


class SpaceKeyError(KeyError):
    """Prevent frame access to subspace attributes."""

    def __init__(self, name, col):
        super().__init__(
            f"{name}['{col}'] access is restricted for "
            f"subspace attributes use s{name}['{col}'] or "
            f"subspace.{name}['{col}']"
        )


class SubSpaceKeyError(KeyError):
    """Prevent direct access to variables not listed in metaframe.subspace."""

    def __init__(self, col, subspace):
        super().__init__(f"{col} not specified as a subspace attribute {subspace}")


class GridError(IndexError):
    """Restrict access to grid solvers when requisite attributes are absent."""

    def __init__(self, attribute):
        super().__init__(f"Items with required attribute {attribute} are absent.")


class PlasmaGridError(AttributeError):
    """Ensure that plasma rejoin is meshed with hexagonal elements."""

    def __init__(self, turn):
        super().__init__(f"Plasma must be meshed with hexagonal elements not {turn}.")
