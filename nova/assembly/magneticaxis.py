
import numpy as np
import scipy.integrate
import scipy.interpolate

from nova.assembly.dataset import FieldData
from nova.utilities.pyplot import plt



field = FieldData('v3')

reference_field = -8.0045 * 1e4
radius = 4.1053


shift_x = radius / (np.pi * reference_field) * \
    np.trapz(field.data.field[1] * np.cos(field.data.phi), field.data.phi)
shift_y = radius / (np.pi * reference_field) * \
        np.trapz(field.data.field[1] * np.sin(field.data.phi), field.data.phi)
offset = np.sqrt(shift_x**2 + shift_y**2)

print(1e3*offset)


'''

_field= scipy.interpolate.interp1d(field.data.phi.data,
                                   field.data.field[:2].data, axis=1)
def fun(t, y):
    return _field(t)

sol = scipy.integrate.solve_ivp(fun, (0, 2*np.pi), (radius, 0),
                                t_eval=np.linspace(0, 2*np.pi, 200))

plt.plot(sol.y[0], sol.y[1])
plt.axis('equal')


number = 151
theta = np.linspace(0, 2*np.pi, number, endpoint=False)
radius = 3.7
offset = 0.3

waveform = np.linalg.norm([radius*np.cos(theta),
                           -offset + radius*np.sin(theta)], axis=0)

n1_offset = np.abs(np.fft.rfft(waveform)[1]) / (number // 2)

print(offset, n1_offset)

plt.plot(theta, waveform)
'''
