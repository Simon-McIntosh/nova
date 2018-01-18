import pandas as pd
from amigo.pyplot import plt
import numpy as np
from amigo.addtext import linelabel


path = 'C:/Users/mcintos/Downloads/'
file = 'plasma.dat_MD_UP_lin50ms_Cat.I_2010_Int_3F9EHM_v1_0.dat'

data = pd.read_csv(path+file, delim_whitespace=True, skiprows=40,
                   na_values='NAN')
data = data.dropna()  # remove NaN values

columns = {}
for c in list(data):
    columns[c] = c.split('[')[0]
data = data.rename(index=str, columns=columns)

data.loc[:, 'I_dw'] /= 4  # per turn
data.loc[:, 'I_up'] /= 4

t = np.copy(data.loc[:, 't'])
Ivs3 = abs(np.copy(data.loc[:, 'I_dw']))

trip_current = 0.01*np.max(Ivs3)
trip_index = next(i for i, Ivs in enumerate(Ivs3) if Ivs > trip_current)
trip_t = t[trip_index]

Io = 60*np.ones(len(t))


tau = 1.53/(17.66)*1e3

# tau = 80

Io[trip_index:] = 60*np.exp(-(t[trip_index:]-trip_t)/tau)

print('dI {:1.1f}kA'.format(np.max(Ivs3+Io[0])-np.max(Ivs3+Io)))
print('tau {:1.0f}ms'.format(tau))

text = linelabel(loc='max', postfix='kA')


plt.plot(t, Ivs3+Io, '-')
text.add('decay')

plt.plot(t, Ivs3+np.max(Io), '-')
text.add('constant')

plt.plot(t, Io, '--', label='$I_{decay}$')
plt.plot(t, Ivs3, '--', label='$I_{VDE}$')


ax = plt.gca()
ylim = ax.get_ylim()
plt.plot(trip_t*np.ones(2), ylim, ':', color='gray')
plt.ylabel(r'$I_{VS3}$ kA')
plt.xlabel(r'$t$ ms')
text.plot(Ralign=True)
plt.legend(loc=4)
plt.despine()

'''
# file = 'Case_04__Eddy_current_in_Lower_VS_coil_d_U63UG9_v1_0.dat'
file = 'Case_08__Eddy_current_in_Lower_VS_coil_d_UMES6A_v1_0.dat'
data_ansys = pd.read_csv(path+file, delim_whitespace=True)
columns = {}
for c in list(data_ansys):
    columns[c] = c.split(',')[0]
data_ansys = data_ansys.rename(index=str, columns=columns)
for var in ['I_cond', 'I_cond+jacket']:
    data_ansys.loc[:,var] *= 1e-3
data_ansys.loc[:,'t'] *= 1e3

data_ansys.plot('t', 'I_cond+jacket', ax=ax)

plt.ylabel(r'$I$, kA')
plt.xlabel(r'$t$, ms')
plt.despine()
plt.show()
'''