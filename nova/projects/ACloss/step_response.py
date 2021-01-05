import scipy
import numpy as np

from nova.utilities.pyplot import plt
from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.thermalhydralic.sultan.shotprofile import ShotProfile

x = [-0.01, 0.001, -0.0001, 5]
nz = 0  # number of zeros


lti = scipy.signal.ZerosPolesGain(x[:nz], x[nz:-1], x[-1])

t = np.linspace(0, 10000)

y = scipy.signal.step(lti, T=t)[1]

#plt.plot(t, y)

profile = ShotProfile(ShotInstance('CSJA_3', -3))
t, Qdot = profile.shotresponse.step
plt.plot(t, Qdot)