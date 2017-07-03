import pylab as pl
import pandas as pd
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 8)

pl.style.use('seaborn-poster')


path = 'C:/Users/mcintos/Downloads/'
file = 'plasma.dat_MD_DW_exp16ms_2010_Int_3F7FBJ_v1_0.dat'

cindex = []
cindex.extend(range(6))
cindex.extend(range(9, 11))
cindex.extend(range(27, 37))
cindex.extend(range(33, 62))

data = pd.read_csv(path+file, delim_whitespace=True,
                   skiprows=40)  # , usecols=cindex

columns = {}
for c in list(data):
    columns[c] = c.split('[')[0]
data = data.rename(index=str, columns=columns)

data.loc[:,'I_dw'] /= 4
data.loc[:,'I_up'] /= 4

ax = data.plot('t', ['I_up', 'I_dw'])

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

pl.ylabel(r'$I$, kA')
pl.xlabel(r'$t$, ms')
sns.despine()