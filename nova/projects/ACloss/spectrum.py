
import os.path

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.signal
import nlopt

from nova.definitions import root_dir
from nova.utilities.png_tools import data_mine, data_load
import matplotlib.pyplot as plt

path = os.path.join(root_dir, 'input/ACLoss/')
#data_mine(path, 'CS_twist_pitch',
#          xlim=(0.05, 10), ylim=(0.5, 10), scale='log')
data = data_load(path, 'CS_twist_pitch', date='2020_11_20')[0][0]
data['x'] *= 2*np.pi  # to rad/s

omega, p = data['x'], data['y']/data['x']

lnp = scipy.interpolate.interp1d(np.log10(omega), np.log10(p))
omega = 10**np.linspace(*np.log10(np.array([omega[0], omega[-1]])), 200)
p = 10**lnp(np.log10(omega))

def _H(x):
    H = x[0]
    for bp in x[1:]:
        H /= (omega+bp)
    return H

def bode(x):
    zeros, poles, gain = [], -x[:-1], x[-1]
    mag = scipy.signal.bode((zeros, poles, gain), w=omega)[1]
    H = 10**(mag/20)
    return H

def bode_err(x):
    H = bode(x)
    return np.sqrt(np.mean((np.log10(p) - np.log10(H))**2))

def log_rms(x, grad):
    err = bode_err(x)
    print(err)
    if len(grad) > 0:
        grad[:] = scipy.optimize.approx_fprime(x, bode_err, 1e-6)
    return err
'''
xo = [6, 10, 10]
opt = nlopt.opt(nlopt.LD_MMA, len(xo))
opt.set_min_objective(log_rms)
opt.set_ftol_rel(1e-6)
#opt.set_xtol_rel(self.xtol_rel)
x = opt.optimize(xo)
print(x, opt.last_optimize_result())

#opt.set_ftol_rel(self.ftol_rel)
#opt.set_xtol_rel(self.xtol_rel)
#opt.set_lower_bounds(self.grid_boundary[::2])
#opt.set_upper_bounds(self.grid_boundary[1::2])
'''


xo = [0.1, 2, 20]
bounds = [(np.min(omega), np.max(omega)) for __ in range(len(xo))]
bounds[-1] = (1e-6, None)
res = scipy.optimize.minimize(bode_err, xo, method='L-BFGS-B',
                              bounds=bounds, options={'gtol': 1e-9})
tau = np.sort(2*np.pi/res.x[:-1])

print(res)

H = bode(res.x)
plt.plot(omega, p, '-', label='data')
plt.plot(omega, H, '-', label='fit')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$H(\omega$)')
plt.despine()
plt.legend()


tau_text = ', '.join(f"{t:1.4f}" for t in tau)
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

lti = scipy.signal.ZerosPolesGain([], -res.x[:-1], res.x[-1])
sys = scipy.signal.StateSpace(lti)

#plt.figure()
#t, p = scipy.signal.impulse(lti, N=100)
#plt.plot(t, p)

#omega = omega[::5]
ncycle, nmuilt = 20, 10
index = slice(nmuilt*ncycle // 2, None)
P = np.zeros(len(omega))
for i, w in enumerate(omega):
    t_max = ncycle * 2*np.pi / w
    t = np.linspace(0, t_max, nmuilt*ncycle)
    Bdot = np.cos(w*t)
    t, _p = scipy.signal.lsim(sys, Bdot, t)[:2]
    #P[i] = np.sqrt(np.mean(_p[index]**2))
    #P[i] = 1/np.sqrt(2) * np.max(_p)
    P[i] = np.max(_p[index])

plt.figure()
plt.plot(data['x'], data['y']/data['x'], label='data')
plt.plot(omega, H, label='fit')
plt.plot(omega, P, label='LTI')
plt.xscale('log')
plt.yscale('log')
plt.legend()
# "FTPTrans@PSI"
http://www.psi.ch

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

