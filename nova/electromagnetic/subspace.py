
import pandas
import numpy as np

#from nova.electromagnetic.frame import SuperFrame
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.dataarray import DataArray


class SubSpace(DataArray):
    """Manage frame subspace (all independent rows)."""

    def __init__(self, frame):
        index = self.get_index(frame)
        columns = self.get_columns(frame)
        array = [attr for attr in frame.metaarray.array
                 if attr in columns]
        super().__init__(pandas.DataFrame(frame), index=index, columns=columns,
                         Avalible=None, Array=array)

    def get_index(self, frame):
        """Return subspace index."""
        if not hasattr(frame, 'multipoint'):
            return None
        if frame.multipoint.index.empty:
            return frame.index
        return frame.multipoint.index

    def get_columns(self, frame):
        """Return subspace columns."""
        if frame.empty:
            return frame.metaframe.subspace
        if np.array([attr in frame.metaframe.subspace
                     for attr in frame.columns]).any():
            with frame.metaframe.setlock(None, 'subspace'):  # update metaframe
                frame.metadata = {'additional': frame.metaframe.subspace}
            return frame.metaframe.subspace
        return []

