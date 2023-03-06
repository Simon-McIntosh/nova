"""Collection of biot error classes."""


class PlasmaTopologyError(IndexError):
    """Raise for degenerate plasma topologies."""

    def __init__(self, reason: str):
        super().__init__(f'{reason}.')
