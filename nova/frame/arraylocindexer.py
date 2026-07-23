"""Fast array access into the columnar frame store.

``ArrayLocIndexer`` exposes the array-group columns as live ndarray views so
callers (notably ``nova.biot`` and the frameset Loc indexers) can read and
write bulk column data without label bookkeeping. It holds a reference to the
frame, so views stay current as the store mutates.
"""

from __future__ import annotations

import numpy as np

from nova.frame.columnar import is_list_like


class ArrayLocIndexer:
    """Live array-column accessor over a frame's columnar store."""

    def __init__(self, name, frame):
        """Bind the indexer to a frame and its array-column group name."""
        self.name = name
        self.frame = frame

    @property
    def attrs(self) -> list[str]:
        """Return the array-group columns currently present in the frame."""
        return [attr for attr in self.frame.metaframe.array if attr in self.frame]

    def __call__(self) -> list[str]:
        """Return the list of exposed array columns."""
        return self.attrs

    def __len__(self) -> int:
        """Return the number of exposed array columns."""
        return len(self.attrs)

    def _rows(self, selector):
        """Resolve a row selector to positions against the frame index."""
        if isinstance(selector, str):
            return self.frame.index.get_loc(selector)
        return selector

    def __getitem__(self, key):
        """Return a column view, or a positional selection of one."""
        if isinstance(key, tuple):
            rows, col = key
            return self.frame[col][self._rows(rows)]
        return self.frame[key]

    def __setitem__(self, key, value):
        """Write a column (or a positional slice of one) in place."""
        if isinstance(key, tuple):
            rows, col = key
            self.frame[col][self._rows(rows)] = value
            return
        col = key
        if col in self.frame:
            column = self.frame[col]
            if is_list_like(value):
                column[:] = np.asarray(value)
            else:
                column[:] = value
        else:
            self.frame[col] = value
