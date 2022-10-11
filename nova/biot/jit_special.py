

import numpy as np
import numba
from numba.experimental import jitclass
import scipy.special


#@jitclass
class Biot:

    #k2: numba.float64[:]

    def __init__(self, k2):
        self.k2 = k2


    def elip(self):
        return scipy.special.ellipe(self.k2)
        data = np.empty_like(self.k2)
        for i, k2 in enumerate(self.k2):
            data[i] = scipy.special.ellipe(k2)
        return data

if __name__ == '__main__':


    rng = np.random.default_rng(2025)
    k2 = rng.random(10000)

    biot = Biot(k2)
    print(biot.elip())
