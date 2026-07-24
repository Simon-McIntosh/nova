"""Label and positional accessors over the columnar frame store.

These replace the runtime-manufactured pandas ``_LocIndexer`` subclasses. Row
selection resolves a label, label-slice, boolean-column name, ``part`` value,
label list or boolean mask to integer positions against the store index; the
column is resolved by name (``loc``) or position (``iloc``).
"""

from __future__ import annotations

import numpy as np

from nova.frame.columnar import Vector, is_list_like
from nova.frame.error import SpaceKeyError


class Accessor:
    """Base row/column accessor bound to a frame."""

    positional_columns = False

    def __init__(self, frame):
        """Bind the accessor to its frame."""
        self.frame = frame

    def _split(self, key):
        """Return (row selector, column selector) from an index key."""
        if isinstance(key, tuple):
            return key[0], key[1]
        return key, slice(None)

    def _column(self, col):
        """Resolve a column selector to a name."""
        if self.positional_columns and isinstance(col, (int, np.integer)):
            return self.frame.columns.to_list()[col]
        if isinstance(col, (int, np.integer)) and not self.positional_columns:
            return self.frame.columns.to_list()[col]
        return col

    def _rows(self, selector):
        """Resolve a row selector to positions, a slice or a boolean mask."""
        index = self.frame.index
        if isinstance(selector, slice):
            start = None if selector.start is None else self._position(selector.start)
            stop = selector.stop
            if isinstance(stop, str):
                # label slices are inclusive of the stop label
                stop = index.get_loc(stop) + 1
            return slice(start, stop, selector.step)
        if isinstance(selector, str):
            if selector in self.frame:  # boolean column name (e.g. "plasma")
                return np.asarray(self.frame[selector], dtype=bool)
            try:
                return index.get_loc(selector)
            except KeyError:
                if "part" in self.frame:
                    return np.asarray(self.frame["part"]) == selector
                raise
        if is_list_like(selector):
            mask = np.asarray(selector)
            if mask.dtype == bool:
                return mask
            return index.get_indexer(mask)
        return selector

    def _position(self, label):
        """Return the integer position for a label (or the int itself)."""
        if isinstance(label, str):
            return self.frame.index.get_loc(label)
        return label

    def __getitem__(self, key):
        """Return the selected column values."""
        rows, col = self._split(key)
        if is_list_like(col):
            return self._multi_column_get(rows, col)
        column = self.frame[self._column(col)]
        rows = self._rows(rows)
        if isinstance(rows, slice) and rows == slice(None, None, None):
            return column  # the store's id-stable cached view, not a new slice
        return column[rows]

    def _multi_column_get(self, rows, cols):
        """Return several columns stacked along the last axis.

        Serves ``loc[rows, [c1, c2, ...]]``: a scalar row yields a 1-D vector
        across the columns, a row selection a 2-D ``(rows, columns)`` block.
        """
        rows = self._rows(rows)
        whole = isinstance(rows, slice) and rows == slice(None, None, None)
        selected = [
            np.asarray(self.frame[self._column(col)])[slice(None) if whole else rows]
            for col in cols
        ]
        return Vector(np.stack(selected, axis=-1))

    def __setitem__(self, key, value):
        """Assign to the selected column values, in place on the store."""
        rows, col = self._split(key)
        if is_list_like(col):
            self._multi_column_set(rows, col, value)
            return
        col = self._column(col)
        rows = self._rows(rows)
        if isinstance(rows, slice) and rows == slice(None, None, None):
            # route through the frame setter without touching the store first,
            # so subspace / column guards raise before any read
            self.frame[col] = value
            return
        subspace_active = getattr(self.frame, "_subspace_active", None)
        if subspace_active is not None and subspace_active(col):
            raise SpaceKeyError("loc", col)
        column = self.frame[col]
        column[rows] = value  # in-place view write-back to the store

    def _multi_column_set(self, rows, cols, value):
        """Assign several columns from a per-column last axis, in place."""
        rows = self._rows(rows)
        value = np.asarray(value)
        subspace_active = getattr(self.frame, "_subspace_active", None)
        for position, col in enumerate(cols):
            name = self._column(col)
            if subspace_active is not None and subspace_active(name):
                raise SpaceKeyError("loc", name)
            column = self.frame[name]
            column[rows] = value[..., position] if value.ndim else value


class LabelAccessor(Accessor):
    """Label-oriented accessor (``loc``)."""


class PositionAccessor(Accessor):
    """Integer-position accessor (``iloc``)."""

    positional_columns = True

    def __getitem__(self, key):
        """Return a column selection, or a whole row for a bare integer."""
        if not isinstance(key, tuple):
            return RowView(self.frame, key)
        return super().__getitem__(key)


class ScalarLabelAccessor(Accessor):
    """Single-element label accessor (``at``)."""

    def __getitem__(self, key):
        """Return a single element."""
        rows, col = self._split(key)
        return self.frame[self._column(col)][self._position(rows)]

    def __setitem__(self, key, value):
        """Set a single element in place."""
        rows, col = self._split(key)
        self.frame[self._column(col)][self._position(rows)] = value


class ScalarPositionAccessor(ScalarLabelAccessor):
    """Single-element integer accessor (``iat``)."""

    positional_columns = True

    def _position(self, label):
        """Return the integer position unchanged."""
        return label


class RowView:
    """A single row exposed as an ordered, list-convertible record."""

    def __init__(self, frame, position):
        """Bind the view to a frame row position."""
        self._values = [frame[name][position] for name in frame.columns]

    def to_list(self):
        """Return the row values as a python list."""
        return [
            value.item() if isinstance(value, np.generic) else value
            for value in self._values
        ]

    def __iter__(self):
        """Iterate over the row values."""
        return iter(self.to_list())

    def __eq__(self, other):
        """Compare against a list of values."""
        return self.to_list() == list(other)
