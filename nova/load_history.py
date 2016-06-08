import pylab as pl
from shelf import PKL

rc = {'figure.figsize':[3*3.14,3*3.14*1/3],'savefig.dpi':100,
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      "lines.linewidth": 2}
sns.set(context='poster',style='white',font='serif',palette='Set2',
        font_scale=1,rc=rc,)
color = sns.color_palette('Set2',5)

pkl = PKL('moveSX_Dev')
#inv = pkl.fetch(['inv'])[0]
'''
pl.figure()
pl.plot(inv.log['dTpol'])
#pl.ylim([0,20])
pl.yscale('linear')
'''

pl.figure()
pl.plot(inv.log['rms'])
pl.ylim([0,0.5])
pl.yscale('linear')
pl.xlabel('Position update')
pl.ylabel('RMS error')
sns.despine()
pl.savefig('../Figs/rms_error')

'''
pl.figure()
pl.plot(np.array(inv.log['FzPF'])*1e-6)
#pl.ylim([0,20])
pl.yscale('linear')
'''