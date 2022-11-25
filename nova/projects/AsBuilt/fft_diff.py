
import numpy as np

from nova.plot import plt

ndiv = 500
wavenumber = 3
amplitude = 2.5
phase = np.pi/2

period = 2*np.pi

phi, dphi = np.linspace(-period/2, period/2, ndiv, retstep=True, endpoint=True)
wave = amplitude*np.cos(wavenumber*phi + phase)

kappa = np.fft.fftfreq(ndiv, dphi)[:ndiv // 2 + 1]

wave_hat = np.fft.rfft(wave)

dwave_hat = 1j*kappa*wave_hat * period

dwave = np.fft.irfft(dwave_hat)

plt.plot(phi, -amplitude*wavenumber*np.sin(wavenumber*phi + phase))
plt.plot(phi, dwave)
