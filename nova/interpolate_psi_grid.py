import pylab as pl
import numpy as np
from itertools import cycle
import seaborn as sns
sns.set_context('poster')
sns.set_style('ticks')
sns.set_palette('Set2')
from streamfunction import SF

  
filename = './eqlib_data/Equil_AR3d1_bt_1d03li_0d8_Ipl_20d25_26co_2L2R9C_v1_0.eqdsk'
dataname = 'SFD'

pl.figure(figsize=(14,16))
pl.axis('equal')
sf = SF(filename,dataname)
#sf.contour()

from scipy.interpolate import RectBivariateSpline as rbs

interp_psi = rbs(sf.rm,sf.zm,sf.psi,s=0.05)

rm = np.linspace(sf.rm[0],sf.rm[-1],5*len(sf.rm))
zm = np.linspace(sf.zm[0],sf.zm[-1],5*len(sf.zm))
psi = interp_psi(rm,zm,dx=0,dy=0)

#pl.contour(sf.rm, sf.zm, sf.psi.T,levels=[0],colors='b')
pl.contour(rm, zm, psi.T,levels=[0],colors='r')