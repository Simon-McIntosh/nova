
import numpy as np
import scipy.integrate
import scipy.interpolate

from nova.assembly.dataset import Ansys, FieldData
import matplotlib.pyplot as plt

Bo = -8.0045
radius = 4.1053

simulation = 'c2'

field = 1e-4 * FieldData('v3').data.field
field /= Bo

deviation = Ansys().data.deviation.sel(simulation='v3')

zeta_t, zeta_r = np.zeros(2), np.zeros(2)
zeta_t[0] = radius / np.pi * np.trapz(field[1] * np.cos(field.phi), field.phi)
zeta_t[1] = radius / np.pi * np.trapz(field[1] * np.sin(field.phi), field.phi)

zeta_r[0] = -radius / np.pi * np.trapz(field[0] * np.sin(field.phi), field.phi)
zeta_r[1] = radius / np.pi * np.trapz(field[0] * np.cos(field.phi), field.phi)

zeta_r *= 1e3
zeta_t *= 1e3

_delta = scipy.interpolate.interp1d(field.phi, field.data[:2])

def fun(t, y, radial_only=False):
    phi = np.arctan2(y[1], y[0])
    if phi < 0:
        phi += 2*np.pi
    delta = _delta(phi)
    if radial_only:
        Bx = delta[0]*np.cos(phi) - np.sin(phi)
        By = delta[0]*np.sin(phi) + np.cos(phi)
        return Bx, By
    Bx = delta[0]*np.cos(phi) - delta[1]*np.sin(phi) - np.sin(phi)
    By = delta[0]*np.sin(phi) + delta[1]*np.cos(phi) + np.cos(phi)
    return Bx, By

def loop(t, y):
    if t == 0:
        return 1
    return y[1]

loop.terminal = True
loop.direction = 1

sol = scipy.integrate.solve_ivp(fun, (0, 4*np.pi*radius), (radius, 0),
                                t_eval=np.linspace(0, 4*np.pi*radius, 200),
                                rtol=1e-4, events=loop, max_step=0.5)
phi_sol = np.arctan2(sol.y[1], sol.y[0])
circle = np.array([radius*np.cos(phi_sol), radius*np.sin(phi_sol)])
delta_sol = sol.y - circle

phi = np.linspace(0, 2*np.pi)
plt.plot(radius*np.cos(phi), radius*np.sin(phi), '-.', color='gray',
         label='wall')

delta_r = radius + deviation
plt.plot(delta_r*np.cos(deviation.phi), delta_r*np.sin(deviation.phi),
         '-', color='C2', label=r'$h(\phi)$')

factor = 1e3
plt.plot(circle[0] + factor*delta_sol[0], circle[1] + factor*delta_sol[1],
         color='C1', label='RK45(B)')
plt.axis('equal')

modes = 1
fft = np.fft.rfft(deviation)
# fft[0] = radius / len(deviation)
fft[modes+1:] = 0
n1_deviation = np.fft.irfft(fft)

n1_delta_r = radius + n1_deviation
plt.plot(n1_delta_r*np.cos(deviation.phi), n1_delta_r*np.sin(deviation.phi),
         '--', color='C2', label=fr'$n_{{0to{modes}}}$ fit')

n1_mode = fft[1] / (len(deviation) // 2)

offset = np.abs(n1_mode)
angle = np.angle(np.conj(n1_mode))

plt.plot([0, n1_mode.real], [0, -n1_mode.imag], '-o', ms=6, color='C0',
         label=f'offset {offset:1.2f}')

plt.plot(*zeta_r, 'C4X', ms=6,
         label=fr'$\zeta_{{B_r}} ${np.linalg.norm(zeta_r):1.2f}')
plt.plot(*zeta_t, 'C5X', ms=6,
         label=fr'$\zeta_{{B_\phi}} ${np.linalg.norm(zeta_t):1.2f}')

plt.plot(0, 0, 'o', ms=6, color='gray')
plt.axis('off')
plt.legend(fontsize='small', bbox_to_anchor=(0.71, 1))

plt.title('n1 offset for simulation v3')
'''
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
