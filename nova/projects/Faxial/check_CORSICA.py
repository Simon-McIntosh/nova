import numpy as np

from nova.limits.tieplate import get_tie_plate

tie_plate = get_tie_plate(wp='Built')
nCS = 6

FxCS = '7.3551D+01 7.5189D+02 1.6606D+03 1.6235D+03 5.8907D+02 7.8573D+01'
FzCS = '1.5332D+02	3.7810D+02	2.5249D+02	-3.2667D+02 -3.0269D+02 -4.8665D+01'
FcCS = '-3.0263D+00 3.0505D+01 1.0676D+02 1.1928D+02 8.2709D+00 -5.0205D+00'

FxCS = np.array([float(Fx.replace('D', 'E')) for Fx in FxCS.split()])
FzCS = np.array([float(Fz.replace('D', 'E')) for Fz in FzCS.split()])
FcCS = np.array([float(Fc.replace('D', 'E')) for Fc in FcCS.split()])

Ftp = -tie_plate['preload'] 
Ftp += tie_plate['alpha'] * np.sum(FxCS)
Ftp += np.sum(tie_plate['beta'] * FzCS)
Ftp += tie_plate['gamma'] * np.sum(FcCS)
Faxial = np.ones(nCS+1)
Faxial[-1] = Ftp
for i in np.arange(1, nCS + 1):  # Faxial for each gap top-bottom
    Faxial[-(i+1)] = Faxial[-i] + FzCS[-i] - tie_plate['mg']
    
print(', '.join([f'{Fax:1.1f}' for Fax in Faxial]))
