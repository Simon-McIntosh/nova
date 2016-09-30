import numpy as np
import pylab as pl
from surface import bernstein
import scipy.optimize as op
import seaborn as sns
rc = {'figure.figsize':[3.14,3.14*10/12],'savefig.dpi':200, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='paper',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
color = cycle(sns.color_palette('Set2'))


psi = np.linspace(0,1,500)      
bern = bernstein(psi,n=28)

sf = SF(Config('vde'))

Pfit,b = bern.fit(sf.Pprime(psi))
bo = np.append(np.ones(bern.n+1),np.arange(0,bern.n+1))
bounds = (bern.n+1)*[(None,None)]+(bern.n+1)*[(0,bern.n)]

bo = np.ones(bern.n+1)
bounds = (bern.n+1)*[(None,None)]
opp = op.minimize(bern.error,bo,options={'disp':True},method='L-BFGS-B',
                  bounds=bounds)
Popp_fit = bern.gen(opp.x)
print(opp.fun)

'''
Ffit,b = bern.fit(sf.FFprime(psi))

Fknots = bern.A*b

bo = np.ones(len(b))
opp = op.minimize(bern.error,bo,options={'disp':False},method='SLSQP',
                  bounds=[(None,None),(None,None),(None,None),(0.1,0.1)])
Fopp_fit = bern.gen(opp.x)  
'''

'''
for method in ['Nelder-Mead','Powell','BFGS',
               'Anneal','L-BFGS-B','COBYLA','SLSQP']:
    opp = op.minimize(bern.error,bo,options={'disp':False},
                      method=method,tol=1e-2)
    opp_fit = bern.gen(opp.x)                               
    print(method,abs(bern.error(opp.x)),opp.nfev)
'''

fig,ax = pl.subplots(2,sharex=True)
ax[0].plot(psi,sf.Pprime(psi),'.',markersize=3)
ax[0].plot(psi,Pfit)
ax[0].plot(psi,Popp_fit)

'''
ax[1].plot(psi,sf.FFprime(psi),'.',markersize=3)
ax[1].plot(psi,Ffit)
ax[1].plot(psi,Fopp_fit)
for i in range(bern.n+1):
    ax[1].plot(bern.t,Fknots[:,i])
    '''
sns.despine()