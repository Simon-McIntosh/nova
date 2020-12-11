
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from nova.thermalhydralic.Sultan.sultanshot import SultanShot
from nova.thermalhydralic.Sultan.sultanprofile import SultanProfile
from nova.thermalhydralic.Sultan.sultanplot import SultanPlot

from nova.utilities.pyplot import plt


@dataclass
class Reload:
    """Reload datastructure for post process."""

    threshold: bool = True
    iQdot: bool = True


@dataclass
class Threshold(SultanPlot):

    profile: SultanProfile
    _threshold: float = 0.95
    _iQdot: Tuple[int, int] = field(init=False)
    reload: Reload = field(init=False, default=Reload(), repr=False)

    @property
    def threshold(self):
        """
        Manage heat threshold parameter.

        Parameters
        ----------
        threshold : float
            Heating idexed as Ipulse.abs > threshold * Ipulse.abs.max.

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
        self.reload.iQdot = True

    @property
    def iQdot(self):
        """Return heat index, slice, read-only."""
        if self.reload.iQdot:
            self._evaluate_iQdot()
        return self._iQdot

    def _evaluate_iQdot(self):
        """
        Return slice of first and last indices meeting threshold condition.

        Evaluated as Ipulse.abs() > threshold * Ipulse.abs().max()

        Parameters
        ----------
        data : array-like
            Data vector.
        threshold : float, optional property
            Threshold factor applied to data.abs().max(). The default is 0.95.

        Returns
        -------
        index : slice
            Threshold index.

        """
        Ipulse = self.profile.rawdata[('Ipulse', 'A')]
        Imax = Ipulse.abs().max()
        threshold_index = np.where(Ipulse.abs() >= self.threshold*Imax)[0]
        self._iQdot = slice(threshold_index[0], threshold_index[-1]+1)
        self.reload.iQdot = False

    def _zero_offset(self):
        """Correct start of heating offset in Qdot_norm."""
        zero_offset = self.profile.lowpassdata.loc[self.iQdot.start,
                                                   ('Qdot_norm', 'W')]
        if not np.isclose(zero_offset, 0):
            for attribute in ['rawdata', 'lowpassdata']:
                data = getattr(self.profile, attribute)
                data['Qdot_norm'] -= zero_offset

    def extract_response(self, transient_factor=1.05, plot=False, ax=None):
        """
        Extract heating response at end of heat and max heat.

        Flag transient when max heat >> end of heat.

        Parameters
        ----------
        transient_factor : float, optional
            Limit factor applied to ratio of eoh and max heat.
            Heating is considered transient of ratio exceeds factor.
            The default is 1.05.
        plot : bool, optional
            plotting flag. The default is False
        ax : axis, optional
            plot axis. The default is None (plt.gca())

        Returns
        -------
        t_eoh : float
            end of heating time.
        Qdot_eoh : float
            end of heating value (Qdot_norm).
        t_max : float
            max heating time.
        Qdot_max : float
            max heating value (Qdot_norm).
        steady : bool
            transient flag, False if Qdot_max/Qdot_eoh > transient_factor.

        """
        # extract lowpass data
        self._zero_offset()
        t = self.profile.lowpassdata[('t', 's')]
        Qdot_norm = self.profile.lowpassdata[('Qdot_norm', 'W')]
        # end of heating
        t_eoh = t[self.iQdot.stop-1]
        Qdot_eoh = Qdot_norm[self.iQdot.stop-1]
        dQdot_heat = Qdot_norm[self.iQdot].max() - Qdot_norm[self.iQdot].min()
        argmax = Qdot_norm.argmax()
        t_max = t[argmax]
        Qdot_max = Qdot_norm[argmax]

        steady = True
        if Qdot_max/Qdot_eoh > transient_factor:
            steady = False
        elif Qdot_norm[self.iQdot.start] > Qdot_norm[self.iQdot.stop-1]:
            steady = False
        elif Qdot_norm[self.iQdot.stop-1] - Qdot_norm[self.iQdot].min() < \
                0.95 * dQdot_heat:
            steady = False
        if plot:
            if ax is None:
                ax = plt.gca()
            ax.plot(t_eoh, Qdot_eoh, **self._get_marker(steady, 'eoh'))
            ax.plot(t_max, Qdot_max, **self._get_marker(steady, 'max'))
        return t_eoh, Qdot_eoh, t_max, Qdot_max, steady


if __name__ == '__main__':

    shot = SultanShot('CSJA_3')
    profile = SultanProfile(shot)
    post = SultanPostProcess(profile)
    post.extract_response(plot=True)