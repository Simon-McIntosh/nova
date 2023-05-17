"""Manage sultan timeseries data."""
from dataclasses import dataclass, field
from typing import Union

import numpy as np


from nova.thermalhydralic.sultan.pointdata import PointData
import matplotlib.pyplot as plt


@dataclass
class Profile:
    """Offset and normalize sultan timeseries data."""

    sample: Union[Sample, Trial, Campaign, str]
    _offset: Union[bool, tuple[float]] = True
    _normalize: bool = True
    _pointdata: PointData = field(init=False, repr=False)

    def __post_init__(self):
        """Build data pipeline."""
        if not isinstance(self.sample, Sample):
            self.sample = Sample(self.sample)
        self.offset = self._offset
        self.normalize = self._normalize

    @property
    def offset(self):
        """Manage data offset flag."""
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset
        self.pointdata.offset = offset

    @property
    def normalize(self):
        """Manage normalization flag."""
        return self._normalize

    @normalize.setter
    def normalize(self, normalize):
        self._normalize = normalize
        self.pointdata.normalize = normalize

    @property
    def pointdata(self):
        """Return pointdata, read-only."""
        if self.sample.data.reload.profile:
            self._pointdata = PointData(
                self.lowpassdata, self.heatindex.start,
                self.offset, self.normalize)
            self.sample.data.reload.profile = False
        return self._pointdata


if __name__ == '__main__':

    profile = Profile('CSJA13')
    profile.sample.shot = -3
    profile.plot()
