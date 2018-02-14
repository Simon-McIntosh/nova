import os
import nep
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from amigo.png_tools import data_mine, data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
from amigo.addtext import linelabel
from read_dina import timeconstant

path = os.path.join(class_dir(nep), '../Data/Energopul/')

# data_mine(path, 'upperVS_discharge', [0, 1], [-5e3, 65e3])

points = data_load(path, 'upperVS_discharge', date='2018_02_06')[0][0]

tp = points['x']
Ip = points['y']



t = np.linspace(0, tp[-1], 500) # equal spacing, s
I = interp1d(tp, Ip)(t)  # A
I_lp = savgol_filter(I, 11, 3, mode='mirror')  # lowpass filter


imax = np.argmax(I_lp)
tmax = t[imax]
dt_exp = t[-1]-tmax

io = np.argmin(abs(t-tmax-0.1*dt_exp))

t_exp = t[io:]
I_exp = I_lp[io:]
n_exp = len(t_exp)


tc = timeconstant(t_exp, I_exp)
to, Io, tau = tc.get_tau()
'''
def fit(x, *args):
    to, Io, tau = x
    t_exp, I_exp = args
    I_fit = Io*np.exp(-(t_exp-to)/tau)
    err = np.linalg.norm(I_exp-I_fit)
    return err


to, Io, tau = minimize(fit, [0.1, 1e4, 100e-3], args=(t_exp, I_exp)).x
'''

I_fit = Io*np.exp(-(t-to)/tau)

text = linelabel(postfix='', value='', loc='max')
plt.plot(1e3*t, 1e-3*I_lp)
plt.plot(1e3*t_exp, 1e-3*I_exp)
plt.plot(1e3*t, 1e-3*I_fit, '--')
txt = r'$\tau_{approx}=$'+'{:1.1f}ms'.format(1e3*tau)
txt += '\n$t_s$='+'{:1.0f}ms'.format(1e3*to)
txt += '\n$I_s$='+'{:1.1f}kA'.format(1e-3*Io)
text.add(txt)
text.plot()
plt.xlabel('$t$ ms')
plt.ylabel('$I_{VS3}$ kA')
plt.despine()


trip_t = 0
R = 17.66e-3
L = 1.52e-3

R, L = 17.62*1e-3, 1.354*1e-3
tau_o = L/R

plt.figure()
text = linelabel(postfix='kA', value='1.1f', loc='max')
plt.plot(1e3*t, 1e-3*I_lp, label='$I_o$=0kA')
text.add('$I_{max}$=')
plt.plot(1e3*t, 60*np.exp(-(t-trip_t)/tau_o)+1e-3*I_lp, label='$I_o$=+60kA')
text.add(r'$I_{max}$=')
plt.plot(1e3*t, -60*np.exp(-(t-trip_t)/tau_o)+1e-3*I_lp, label='$I_o$==60kA')
text.add(r'$I_{max}$=')
text.plot()
plt.legend()
plt.xlabel('$t$ ms')
plt.ylabel('$I_{VS3}$ kA')
plt.ylim([0, 100])
plt.xlim([0, 300])
plt.despine()
