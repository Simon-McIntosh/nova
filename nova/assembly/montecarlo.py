from dataclasses import dataclass

import numpy as np

from nova.assembly.fieldline import FieldLine
from nova.utilities.pyplot import plt


@dataclass
class Trial(FieldLine):
    """Run stastistical analysis on trial vault assemblies."""

    samples: int = 100000
    sead: int = 2025

    def __post_init__(self):
        super().__post_init__()
        self.rng = np.random.default_rng(self.sead)

        radial = self.sample()
        peaktopeak = self.predict(radial)
        peaktopeak += 0.11

        print(np.quantile(peaktopeak, 0.99))

        plt.hist(peaktopeak, rwidth=0.8, bins=31)

    def sample(self):
        delta = 5
        radial = self.rng.uniform(-delta, delta, (self.samples, 18))
        return radial

    def predict(self, radial):
        return self.electromagnetic.peaktopeak(radial=radial)


if __name__ == '__main__':

    trial = Trial()
