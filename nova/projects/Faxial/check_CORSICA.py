import numpy as np

from nova.limits.tieplate import get_tie_plate

tie_plate = get_tie_plate(wp='Built')
nCS = 6

FxCS = '7.35515D+01 7.51896D+02 1.66062D+03 1.62355D+03 5.89071D+02 7.85736D+01'
FzCS = '1.53323D+02	3.78106D+02	2.52493D+02	-3.26672D+02 -3.02693D+02 -4.86656D+01'
FcCS = '-3.02636D+00 3.05059D+01 1.06762D+02 1.19286D+02 8.27095D+00 -5.02058D+00'

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
