import numpy as np

from nova.thermalhydralic.sultan.fluidresponse import FluidResponse
import matplotlib.pyplot as plt

fluid = FluidResponse("CSJA_6", -1, "Left")

omega, gain = fluid.response(2)

gain = gain**2

dBgain = 10 * np.log10(gain)
log_omega = np.log10(omega)
rolloff = (dBgain[-1] - dBgain[0]) / (log_omega[-1] - log_omega[0])

"""
nzero = 1
npole = 1

zero_vectors = [np.exp(-1j * (k+1) * omega).reshape(-1, 1)
                for k in range(nzero)]
pole_vectors = [(-gain * np.exp(-1j * (k+1) * omega)).reshape(-1, 1)
                for k in range(npole)]
gain_vector = [-np.ones((len(omega), 1))]

response_matrix = np.concatenate(zero_vectors + pole_vectors + gain_vector,
                                 axis=1)
zeropolegain = np.linalg.lstsq(response_matrix, gain, rcond=None)[0]

lti = scipy.signal.ZerosPolesGain(zeropolegain[:nzero],
                                  zeropolegain[nzero:-1],
                                  zeropolegain[-1])
lti_gain = scipy.signal.bode(lti, omega)[1]
"""


# response_matrix =
print(rolloff)

plt.plot(omega, gain)
plt.plot(omega, 10 ** (lti_gain / 20))
plt.xscale("log")
plt.yscale("log")
