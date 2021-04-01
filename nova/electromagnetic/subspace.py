"""Manage frame subspace."""
import pandas
import numpy as np

from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.dataframe import ColumnError
from nova.electromagnetic.framearray import (
    FrameArray,
    FrameArrayLocMixin,
    FrameArrayIndexer
    )

# pylint: disable=too-many-ancestors


class SubSpaceAccessError(IndexError):
    """Prevent direct access to variables not listed in metaframe.subspace."""

    def __init__(self, col, subspace):
        super().__init__(
            f'{col} not specified as a subspace attribute '
            f'metaframe.subspace {subspace}')


class SubspaceLocMixin(FrameArrayLocMixin):
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        col = self.obj.get_col(key)
        if self.obj.lock('subspace') is False:
            if not self.obj.metaframe.hascol('subspace', col):
                raise SubSpaceAccessError(col, self.obj.metaframe.subspace)
        return super().__setitem__(key, value)


class SubSpaceIndexer(FrameArrayIndexer):
    """Extend pandas indexer."""

    @property
    def loc_mixin(self):
        """Return LocIndexer mixins."""
        return SubspaceLocMixin


class SubSpace(SubSpaceIndexer, FrameArray):
    """Manage frame subspace, extract independent rows for subspace columns."""

    def __init__(self, frame):
        index = self.get_subindex(frame)
        columns = self.get_subcolumns(frame)
        array = self.get_subarray(frame, columns)
        metaframe = MetaFrame(
            index, required=[], additional=columns, available=[],
            subspace=[], array=array, _lock=frame.metaframe._lock)
        super().__init__(pandas.DataFrame(frame.loc[index, columns]),
                         index=index, columns=columns,
                         attrs={'metaframe': metaframe})
        self.update_subspace(frame)

    def __getattr__(self, name):
        """Extend pandas.DataFrame.__getattr__. (frame.*)."""
        if name not in self.attrs and name not in self.columns:
            raise ColumnError(name)
        return super().__getattr__(name)

    def __setitem__(self, col, value):
        """Raise error when subspace variable is set directly from frame."""
        if self.lock('subspace') is False:
            if not self.metaframe.hascol('subspace', col):
                raise SubSpaceAccessError(col, self.metaframe.subspace)
        return super().__setitem__(col, value)

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
        subspace = frame.metaframe.subspace
        if np.array([attr in subspace for attr in frame.columns]).any():
            with frame.setlock(None, 'subspace'):  # update metaframe
                frame.metadata = {'additional': frame.metaframe.subspace}
            return frame.metaframe.subspace
        return []

    @staticmethod
    def get_subarray(frame, columns):
        """Return subarray - fast access variables."""
        return [attr for attr in frame.metaframe.array if attr in columns]

    def update_subspace(self, frame):
        """Update frame and subspace metadata."""
        subspace = list(self.columns)
        if subspace:
            self.metaframe.metadata = {'Subspace': subspace}
            frame.metaframe.metadata = {'subspace': subspace}
