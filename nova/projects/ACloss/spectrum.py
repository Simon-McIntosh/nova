
import os.path

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.signal

from nova.definitions import root_dir
from nova.utilities.png_tools import data_mine, data_load
from nova.utilities.pyplot import plt

path = os.path.join(root_dir, 'input/ACLoss/')
#data_mine(path, 'CS_twist_pitch',
#          xlim=(0.05, 10), ylim=(0.5, 10), scale='log')
data = data_load(path, 'CS_twist_pitch', data='2020_11_20')[0][0]
data['x'] *= 2*np.pi  # to rad/s

omega, p = data['x'], data['y']/data['x']

lnp = scipy.interpolate.interp1d(np.log10(omega), np.log10(p))
omega = 10**np.linspace(*np.log10(np.array([omega[0], omega[-1]])), 50)
p = 10**lnp(np.log10(omega))

def _H(x):
    H = x[0]
    for bp in x[1:]:
        H /= (omega+bp)
    return H

def log_rms(x):
    H = _H(x)
    ##### implement here
    w, mag = scipy.signal.bode(sys, w=omega)[:2]

    err = np.sqrt(np.mean((np.log10(p) - np.log10(H))**2))
    return err

xo = [1, 1, 0.01]

bounds = [(1e-2, 1e4) for __ in range(len(xo))]
bounds[0] = (1e-3, None)
res = scipy.optimize.minimize(log_rms, xo, bounds=bounds)
tau = np.sort(2*np.pi/res.x[1:])


H = _H(res.x)
#H = 2000 / ((omega+0.3) * (omega+10)* (omega+5))

plt.plot(omega, p, '-', label='data')
plt.plot(omega, H, '-', label='fit')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$H(\omega$)')
plt.despine()
plt.legend()

tau_text = ', '.join(f"{t:1.2f}" for t in tau)
plt.title(rf'$\tau$={tau_text}s')
plt.xscale('log')
plt.yscale('log')

ax = plt.subplots(2, 1)[1]
ax[0].plot(data['x'], data['y'], label='data')
ax[0].plot(omega, H*omega, label='fit')
ax[0].set_xlabel(r'$\omega$')
ax[0].set_ylabel(r'$\omega H(\omega)$')
ax[1].plot(data['x'], data['y']/data['x'], label='data')
ax[1].plot(omega, H, label='fit')
plt.despine()
ax[1].set_xlabel(r'$\omega$')
ax[1].set_ylabel(r'$H(\omega$)')

lti = scipy.signal.ZerosPolesGain([], -res.x[1:], res.x[0])
sys = scipy.signal.StateSpace(lti)

#plt.figure()
#t, p = scipy.signal.impulse(lti, N=100)
#plt.plot(t, p)

#omega = omega[::5]
ncycle, nmuilt = 20, 10
index = slice(nmuilt*ncycle // 2, None)
P = np.zeros(len(omega))
for i, _omega in enumerate(omega):
    t_max = ncycle * 2*np.pi/_omega
    t = np.linspace(0, t_max, 10*ncycle)
    Bdot = np.cos(_omega*t)
    t, _p = scipy.signal.lsim(sys, Bdot, t)[:2]
    P[i] = np.sqrt(np.mean(_p[index]**2))
    #P[i] = 1/np.sqrt(2) * np.max(_p)

plt.figure()
plt.plot(data['x'], data['y'], label='data')
plt.plot(omega, H*omega, label='fit')
plt.plot(omega, P*omega, label='LTI')


#plt.xscale('log')
#plt.yscale('log')

#plt.ylim([0.1, 100])

w, mag = scipy.signal.bode(sys, w=omega)[:2]
plt.figure()
plt.plot(omega, H, label='fit')
plt.plot(w, 10**(mag/20), label='bode')
plt.xscale('log')
plt.yscale('log')
plt.legend()