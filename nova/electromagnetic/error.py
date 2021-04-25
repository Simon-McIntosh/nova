"""Collection of electromagnetic error classes."""


class ColumnError(IndexError):
    """Prevent column creation."""

    def __init__(self, name):
        super().__init__('Column access via a new attribute name '
                         f'{name} is not allowed.')


class SpaceKeyError(KeyError):
    """Prevent frame access to subspace attributes."""

    def __init__(self, name, col):
        super().__init__(
            f'{name}[\'{col}\'] access is restricted for '
            f'subspace attributes use s{name}[\'{col}\']')


class SubSpaceKeyError(KeyError):
    """Prevent direct access to variables not listed in metaframe.subspace."""

    def __init__(self, col, subspace):
        super().__init__(
            f'{col} not specified as a subspace attribute {subspace}')


class SubSpaceLockError(IndexError):
    """Prevent direct access to frame's subspace variables."""

    def __init__(self, name, col):
        super().__init__(
            f'{name} access is restricted for subspace attributes. '
            f'Use frame.subspace.{name}[:, \'{col}\'] = *.\n\n'
            'Lock may be overridden via the following context manager '
            'but subspace will still overwrite (Cavieat Usor):\n'
            'with frame.setlock(True, \'subspace\'):\n'
            f'    frame.{name}[:, {col}] = *')
