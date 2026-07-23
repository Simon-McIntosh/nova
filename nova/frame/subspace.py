"""Frame subspace: the independent degrees of freedom of a linked frame.

Multi-point links reduce a frame to a smaller set of independent rows (the
link heads). ``SubSpace`` holds the subspace columns for those rows; the
parent :class:`~nova.frame.framespace.FrameSpace` projects them back onto the
full frame through the ``subref`` / ``factor`` mapping.
"""

from __future__ import annotations

import numpy as np

from nova.frame.columnar import Index
from nova.frame.error import SubSpaceKeyError
from nova.frame.framelink import FrameLink
from nova.frame.metaframe import MetaFrame

# pylint: disable=too-many-ancestors


class SubSpace(FrameLink):
    """Independent-row projection of a frame's subspace columns."""

    def __init__(self, frame):
        """Build the subspace frame from a parent frame's independent rows."""
        index = self.get_subindex(frame)
        columns = self.get_subcolumns(frame)
        array = self.get_subarray(frame, columns)
        metaframe = MetaFrame(
            Index(index),
            required=[],
            additional=list(columns),
            available=[],
            subspace=[],
            array=array,
            lock=frame.metaframe.lock,
        )
        labels = list(np.asarray(index))
        if columns and labels:
            data = {col: np.asarray(frame.loc[list(labels), col]) for col in columns}
        else:
            data = {}
        super().__init__(
            data,
            index=labels if labels else None,
            columns=list(columns),
            attrs={"metaframe": metaframe},
        )
        self.update_subspace(frame)

    def __setitem__(self, col, value):
        """Reject writes to columns not declared as subspace attributes."""
        if self.lock("subspace") is False and not self.hascol("subspace", col):
            raise SubSpaceKeyError(col, self.metaframe.subspace)
        super().__setitem__(col, value)

    @staticmethod
    def get_subindex(frame):
        """Return the independent-row index (multipoint link heads)."""
        if not frame.hasattrs("multipoint"):
            return frame.index
        if len(frame.multipoint.index) == 0:
            return frame.index
        return frame.multipoint.index

    @staticmethod
    def get_subcolumns(frame):
        """Return the subspace columns present in the frame."""
        if frame.columns.empty:
            return list(frame.metaframe.subspace)
        subspace = frame.metaframe.subspace
        if any(attr in frame for attr in subspace):
            with frame.setlock(None, "subspace"):
                frame.metaframe.metadata = {"additional": subspace}
            frame.update_columns()
            return [attr for attr in subspace if attr in frame]
        return []

    @staticmethod
    def get_subarray(frame, columns):
        """Return the array-group columns present in the subspace columns."""
        return [attr for attr in frame.metaframe.array if attr in columns]

    def update_subspace(self, frame):
        """Record the subspace column set on both frames."""
        subspace = list(self.columns)
        subspace = [col for col in subspace if col in self.metaframe.additional]
        if subspace:
            self.metaframe.metadata = {"Subspace": subspace}
            frame.metaframe.metadata = {"subspace": subspace}
