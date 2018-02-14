import numpy as np
from amigo.pyplot import plt
from nep.DINA.read_tor import read_tor
from nep.DINA.read_plasma import read_plasma
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import minimize

Rvs3=17.66e-3
Lvs3=1.52e-3
tau_vs3 = Lvs3 / Rvs3  # vs3 timeconstant

pl = read_plasma('disruptions')
tor = read_tor('disruptions')

nf = 1  # pl.dina.nf
nt = 50  #  pl.nt-i_cq

I_dict = {'t': [], 'Ivs': [], 'Ivv': [], 'Ip': [], 'zpl': [],
          'Ivs_dot': [], 'Ivv_dot': [], 'Ip_dot': [], 'zpl_dot': []}
Iwf = [I_dict.copy() for _ in range(nf)]  # current waveforms

nfile = 0

ax = plt.subplots(3, 1, sharex=True)[1]
for i in range(nf):
    tor.read_file(i+nfile)
    pl.read_file(i+nfile)
    i_cq, t_cq = pl.get_quench()[:2]  # quench index, time

    Iwf[i]['t'] = np.linspace(t_cq, pl.t[-1], nt)  # even spaced time
    vs_index = ~np.isnan(pl.Ivs_o)
    Iwf[i]['Ivs'] = interp1d(pl.t[vs_index], pl.Ivs_o[vs_index])(Iwf[i]['t'])
    vv_index = ~np.isnan(tor.Ibar['vv'])
    Iwf[i]['Ivv'] = interp1d(tor.t[vv_index],
                             tor.Ibar['vv'][vv_index])(Iwf[i]['t'])
    Iwf[i]['Ipl'] = interp1d(pl.t, pl.Ip)(Iwf[i]['t'])
    Iwf[i]['zpl'] = interp1d(pl.t, pl.z)(Iwf[i]['t'])  # plasma position

    for var in ['Ivs', 'Ivv', 'Ipl', 'zpl']:
        Iwf[i][var+'_dot'] = np.gradient(Iwf[i][var], Iwf[i]['t'])

    ax[0].plot(1e-3*Iwf[i]['t'], 1e-6*Iwf[i]['Ipl'])
    ax[0].invert_yaxis()

    ax[1].plot(1e-3*Iwf[i]['t'], 1e-3*Iwf[i]['Ivv'], 'C0')
    ax[1].plot(1e-3*Iwf[i]['t'], 1e-3*Iwf[i]['Ivs'], 'C1')

    ax[2].plot(1e-3*Iwf[i]['t'], Iwf[i]['Ivs_dot'], 'C0')
    ax[2].plot(1e-3*Iwf[i]['t'], Iwf[i]['Ivv_dot'], 'C1')
    # ax[2].plot(1e-3*Iwf[i]['t'], Iwf[i]['zpl_dot'], 'C2')


def dIdt(I, t, *args):
    tau_inv = args[0]  # inverse of coupling matrix
    tau_pl = args[1]  # plasma coupling timeconstant [tau_vv_pl, tau_vs_pl]
    lambda_pl = args[2]  # vertical velocity terms
    pl_fun = args[3]  # plasma functions
    wpl, Ipl, Ipl_dot = pl_fun  # vertical speed, current, current rate

    Idot = np.dot(-tau_inv, I)
    Idot += np.dot(-tau_inv, Ipl_dot(t)*tau_pl)  # timeconstants
    Idot += np.dot(-tau_inv, Ipl(t)*wpl(t)*lambda_pl)  # vertical velocity
    return Idot


def get_waveform(x, *args):
    # tau_vv, tau_vv_vs, tau_vv_pl, tau_vs_pl = x[:4]  # unpack timeconstants
    # lambda_vv_pl, lambda_vz_pl = x[4:]  # unpack vertical velocity terms
    xf = np.zeros(6)  # full variable list
    xf[:len(x)] = x  # populate in order
    tau, lam = {}, {}
    for i, var in enumerate(['vv', 'vv_vs', 'vv_pl', 'vs_pl']):
        tau[var] = xf[i]  # label timeconstants
    for i, var in enumerate(['vv_pl', 'vs_pl']):
        lam[var] = xf[i+4]  # label timeconstants

    tau['vs'] = args[0]  # vs3 timeconstant
    t = args[1]  # time (constant spacing)
    pl_fun = args[2]  # plasma functions, speed, current, current rate
    Io = args[3]  # inital current [Ivv[0], Ivs3[0]]

    tau_pl = np.array([tau['vv_pl'], tau['vs_pl']])  # plasma coupling
    lambda_pl = np.array([lam['vv_pl'], lam['vs_pl']])  # velocity coupling
    tau = np.array([[tau['vv'], tau['vv_vs']],  # coupling matix
                    [tau['vv_vs'], tau['vs']]])
    tau_inv = np.linalg.inv(tau)  # inverse of coupling matrix
    Iode = odeint(dIdt, Io, t, args=(tau_inv, tau_pl, lambda_pl, pl_fun))
    return Iode, tau, lam


def fit_waveform(x, *args):
    # args[:3] = tau_vs3, t, pl_fun
    Ivv_ref = args[3]
    Ivs_ref = args[4]
    Io = np.array([Ivv_ref[0], Ivs_ref[0]])
    Iode = get_waveform(x, *args[:3], Io)[0]

    err = np.linalg.norm(Iode[:, 1] - Ivs_ref) / np.linalg.norm(Ivs_ref)
    err += np.linalg.norm(Iode[:, 0] - Ivv_ref) / np.linalg.norm(Ivv_ref)
    print(err, x)
    return err


wpl = interp1d(Iwf[i]['t'], Iwf[i]['zpl_dot'], fill_value='extrapolate')
Ipl = interp1d(Iwf[i]['t'], Iwf[i]['Ipl'], fill_value='extrapolate')
Ipl_dot = interp1d(Iwf[i]['t'], Iwf[i]['Ipl_dot'], fill_value='extrapolate')
pl_fun = [wpl, Ipl, Ipl_dot]

xo = 1e-3*np.array([881.89, 73.97, 9.58, 0.19, 1.37, -0.45])

# tau_vv, tau_vv_vs, tau_vv_pl, tau_vs_pl

# 0) [338.81, -2.09, 2.99, 0.77, 0.0, -0.25]
# 1) [173.06, 9.26, 1.78, 0.66, -0.08, -0.17]
# 2) [173.38, 9.45, 1.81, 0.64, -0.08, -0.17]
# 3) [385.39, 7.87, 3.24, 0.23, -0.09, -0.11]
# 4) [392.51, 3.37, 3.33, -0.01, -0.06, -0.24]
# 5) [392.18, 3.38, 3.34, -0.01, -0.06, -0.24]
# 6) [496.45, 4.8, 4.44, -0.01, -0.07, 0.17]

# 7) [426.76, 99.32, 6.97, 0.17, 1.46, 1.21]
# 8) [288.23, 52.05, 0.06, 0.2, 2.36, 0.71]

# 9) [881.89, 73.97, 9.58, 0.19, 1.37, -0.45]
# 10) [268.86, 21.26, 2.66, 0.45, -0.14, -0.67]
# 11) [863.27, 101.49, 7.89, 0.05, 1.53, -0.77]

x = minimize(fit_waveform, xo, method='Nelder-Mead',
             options={'xatol': 1e-2},
             args=(tau_vs3, Iwf[i]['t'], pl_fun,
                   Iwf[i]['Ivv'], Iwf[i]['Ivs']),).x

Io = np.array([Iwf[i]['Ivv'][0], Iwf[i]['Ivs'][0]])
Iode = get_waveform(x, tau_vs3, Iwf[i]['t'], pl_fun, Io)[0]

plt.figure()
plt.plot(Iwf[i]['t'], 1e-3*Iwf[i]['Ivv'], 'C0-')
plt.plot(Iwf[i]['t'], 1e-3*Iode[:, 0], 'C0--')
plt.plot(Iwf[i]['t'], 1e-3*Iwf[i]['Ivs'], 'C1-')
plt.plot(Iwf[i]['t'], 1e-3*Iode[:, 1], 'C1--')

print([float('{:1.2f}'.format(1e3*n)) for n in x])

# ax[1].plot(1e-3*Iwf[i]['t'], 1e-3*Iode[:, 0], 'C0--')
# ax[1].plot(1e-3*Iwf[i]['t'], 1e-3*Iode[:, 1], 'C1--')

