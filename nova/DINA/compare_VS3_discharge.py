import numpy as np
from amigo.pyplot import plt


t = np.linspace(0, 0.15, 500)
Io = 60e3

Nova = {'alpha': [0.64754104, 0.13775032, 0.21254934],
        'tau': [0.04390661, 0.01073328, 0.19240964]}

LTC = {'alpha': [0.0629583, 0.3059171, 0.63042031],
       'tau': [0.00068959, 0.01501697, 0.09481686]}

ENP = {'alpha': [0.4003159, 0.36687173, 0.23344466],
       'tau': [0.0283991, 0.12646493, 0.04291896]}


def Idecay(t, Io, coef):
    Id = np.zeros(len(t))
    for alpha, tau in zip(coef['alpha'], coef['tau']):
        Id += alpha*np.exp(-t/tau)
    Id *= Io
    return Id


plt.plot(1e3*t, 1e-3*Io * np.exp(-t / 0.0769), '-.', color='gray',
         label='bare conductor')
plt.plot(1e3*t, 1e-3*Idecay(t, Io, Nova), label='Nova')
plt.plot(1e3*t, 1e-3*Idecay(t, Io, LTC), label='LTC')
plt.plot(1e3*t, 1e-3*Idecay(t, Io, ENP), label='Energopul')
plt.legend(loc=3)
plt.despine()
plt.xlabel('$t$ ms')
plt.ylabel('$I$ kA')
