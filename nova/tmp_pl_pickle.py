'''
import pylab as pl
from streamfunction import SF
from elliptic import EQ
from inverse import INV
from eqConfig import Config
from itertools import cycle
import numpy as np
from radial_build import RB
from shelf import PKL

import matplotlib.pyplot as plt

import seaborn as sns
rc = {'figure.figsize':[7*12/16,7],'savefig.dpi':175, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2'))
fig = plt.figure()

#ax = plt.subplot(111)
plt.axis('equal')
plt.axis('off')

pkl = PKL('moveSX')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])

sf.contour()

inv.plot_coils()
sf.plot_coils(next(Color),coils=sf.coil,label=True,plasma=False,current=True) 
sf.plot_coils(next(Color),coils=eq.coil,label=False,plasma=False) 
inv.plot_fix()


pkl_tmp = PKL('pl_test')

pkl_tmp.write(data={'ax':fig})  # pickle data


fig = pkl_tmp.fetch(['ax'])

'''



import numpy as np
import matplotlib.pyplot as plt
import pickle as pl

# Plot simple sinus function
fig_handle = plt.figure()
x = np.linspace(0,2*np.pi)
y = np.sin(x)
plt.plot(x,y)

# Save figure handle to disk

pl.dump(fig_handle,file('sinus.pickle','w'))
