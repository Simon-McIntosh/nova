"""Example implementation of CS force limit."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt


@dataclass
class CSLimit:
    """
    Define limit loads [MN] and model coefficients.

    Parameters
    ----------
    preload : float
        Residule tie-plate preload at 4k [MN]

    limit_load: float
        Axial force upper limit load acting across module gaps [MN]

    module_weight: float
        Module weight [MN]

    alpha: float
        Fx coefficient (Poisson)

    beta: ArrayLike
        Fz coefficient

    gamma: float
        Fc coefficient (crush)

    """

    preload: float = 190.
    limit_load: float = -26.
    module_weight: float = 1.18

    alpha: ClassVar[float] = -0.0019
    beta: ClassVar[npt.ArrayLike] = np.array([0.0389, 0.1161, 0.1933,
                                              0.2696, 0.3468, 0.4239])
    gamma: ClassVar[float] = 0.0739
    ncoil: ClassVar[int] = 6

    def Faxial(self, FxCS, FzCS, FcCS):
        """Return gap axial force [MN]."""
        Ftp = -self.preload
        Ftp += self.alpha * np.sum(FxCS)
        Ftp += np.sum(self.beta * FzCS)
        Ftp += self.gamma * np.sum(FcCS)
        Faxial = np.ones(self.ncoil+1)
        Faxial[-1] = Ftp
        for i in np.arange(1, self.ncoil+1):  # Faxial for each gap top-bottom
            Faxial[-(i+1)] = Faxial[-i] + FzCS[-i] - self.module_weight
        return Faxial


if __name__ == '__main__':

    # generate Faxial test vector using CORSICA input
    FxCS = '7.3551D+1 7.5189D+2 1.6606D+3 1.6235D+3 5.8907D+2 7.8573D+1'
    FzCS = '1.5332D+2 3.7810D+2 2.5249D+2 -3.2667D+2 -3.0269D+2 -4.8665D+1'
    FcCS = '-3.0263D+0 3.0505D+1 1.0676D+2 1.1928D+2 8.2709D+0 -5.0205D+0'

    # translate
    FxCS = np.array([float(Fx.replace('D', 'E')) for Fx in FxCS.split()])
    FzCS = np.array([float(Fz.replace('D', 'E')) for Fz in FzCS.split()])
    FcCS = np.array([float(Fc.replace('D', 'E')) for Fc in FcCS.split()])

    cslimit = CSLimit()
    Faxial = cslimit.Faxial(FxCS, FzCS, FcCS)

    assert np.allclose(
        Faxial, [-196.30075561, -348.44075561, -725.36075561, -976.67075561,
                 -648.82075561, -344.95075561, -295.10575561])
