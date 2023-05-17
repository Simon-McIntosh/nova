"""Index heating in sultan shots."""
from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas
import numpy as np


@dataclass
class HeatIndex:
    """Index external heating."""

    data: pandas.DataFrame = field(repr=False)
    _threshold: float = 0.25
    _index: slice = field(init=False, default=None)
    reload: SimpleNamespace = field(init=True, repr=False,
                                    default_factory=SimpleNamespace)

    def __post_init__(self):
        """Init reload namespace."""
        self.reload.__init__(threshold=True, index=True)

    @property
    def threshold(self):
        """
        Manage heat threshold parameter.

        Parameters
        ----------
        threshold : float
            Heating idexed as current.abs > threshold * current.abs.max.

        Raises
        ------
        ValueError
            threshold must lie between 0 and 1.

        Returns
        -------
        threshold : float

        """
        if self.reload.threshold:
            self.threshold = self._threshold
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        if threshold < 0 or threshold > 1:
            raise ValueError(f'heat threshold {threshold} '
                             'must lie between 0 and 1')
        self._threshold = threshold
        self.reload.threshold = False
        self.reload.index = True

    @property
    def index(self):
        """
        Return heat index, slice, read-only.

        Evaluated as current.abs() > threshold * current.abs().max()

        """
        if self.reload.index:
            current = self.data.loc[:, ('Ipulse', 'A')]
            abs_current = current.abs()
            max_current = abs_current.max()
            threshold_index = np.where(abs_current >=
                                       self.threshold*max_current)[0]
            self._index = slice(threshold_index[0], threshold_index[-1]+1)
            self.reload.index = False
        return self._index

    @property
    def start(self):
        """Return start index."""
        return self.index.start

    @property
    def stop(self):
        """Return stop index."""
        return self.index.stop

    @property
    def time(self):
        """Return start / end time tuple of input heating period, read-only."""
        return self.data.loc[[self.start, self.stop], ('t', 's')].values

    @property
    def time_delta(self):
        """Return heating period, read-only."""
        return np.diff(self.time)[0]
