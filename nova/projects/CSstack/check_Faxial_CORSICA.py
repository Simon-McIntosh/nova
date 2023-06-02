from pprint import pprint

import numpy as np

from nova.limits.tieplate import get_tie_plate

tie_plate = get_tie_plate(wp="Built")


F = {
    "r": [73.6, 751.9, 1660.6, 1623.6, 589.1, 78.6],
    "z": [153.3, 378.1, 252.5, -326.7, -302.7, -48.7],
    "c": [-3.0, 30.3, 106.8, 119.3, 8.3, -5.0],
}

pprint(F, sort_dicts=False)
pprint(tie_plate, sort_dicts=False)


def calculate_Faxial(tie_plate, F):
    # calculate tie-plate load (Eq.3)
    Ftp = tie_plate["preload"]
    Ftp += tie_plate["alpha"] * np.sum(F["r"])
    Ftp += np.sum(tie_plate["beta"] * F["z"])
    Ftp += tie_plate["gamma"] * np.sum(F["c"])
    nCS = len(F["r"])  # number of CS modules
    Faxial = np.ones(nCS + 1)
    Faxial[-1] = -Ftp  # set upper gap to negated tie-plate load (Eq.4)
    for i in np.arange(1, nCS + 1):  # Faxial for each gap top-bottom (Eq.5)
        Faxial[-(i + 1)] = Faxial[-i] + F["z"][-i] - tie_plate["mg"]
    return Faxial


Faxial = calculate_Faxial(tie_plate, F)

print(Faxial)
