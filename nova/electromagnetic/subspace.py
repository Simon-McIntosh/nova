"""Manage frame subspace."""
import pandas
import numpy as np

from nova.electromagnetic.frameset import FrameSet

# pylint: disable=too-many-ancestors


class SubSpace(FrameSet):
    """Manage frame subspace, extract independent rows for subspace columns."""

    def __init__(self, frame):
        index = self.get_subindex(frame)
        columns = self.get_subcolumns(frame)
        array = self.get_subarray(frame, columns)
        super().__init__(pandas.DataFrame(frame),
                         index=index, columns=columns,
                         Required=[], Additional=columns,
                         Available=columns, Array=array)
        self.metaframe._lock = frame.metaframe._lock  # link locks

    @staticmethod
    def get_subindex(frame):
        """Return subspace index."""
        if not hasattr(frame, 'multipoint'):
            return None
        if frame.multipoint.index.empty:
            return frame.index
        return frame.multipoint.index

    @staticmethod
    def get_subcolumns(frame):
        """Return subspace columns."""
        if frame.columns.empty:
            return frame.metaframe.subspace
        if np.array([attr in frame.metaframe.subspace
                     for attr in frame.columns]).any():
            with frame.metaframe.setlock(None, 'subspace'):  # update metaframe
                frame.metadata = {'additional': frame.metaframe.subspace}
            return frame.metaframe.subspace
        return []

    @staticmethod
    def get_subarray(frame, columns):
        """Return subarray - fast access variables."""
        return [attr for attr in frame.metaarray.array if attr in columns]
