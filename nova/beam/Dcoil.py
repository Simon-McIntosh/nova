import pylab as pl
import numpy as np
from scipy.special import iv as besl  
from scipy.interpolate import interp1d
from amigo import geom

import seaborn as sns
rc = {'figure.figsize':[4,4*8/16],'savefig.dpi':120, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

def pD(r1,r2,npoints=200):
    ro=np.sqrt(r1*r2)
    k=0.5*np.log(r2/r1)
    theta=np.linspace(-0.5*np.pi,1.5*np.pi,2*npoints)
    r,z = np.zeros(2*npoints),np.zeros(2*npoints)
    s = np.zeros(2)
    for m,t in enumerate(theta):
        s = 0
        for n in np.arange(1,20):  # sum convergent series              
            ds = 1j/n*(np.exp(-1j*n*t)-1)*(1+np.exp(1j*n*(t+np.pi)))*\
            np.exp(1j*n*np.pi/2)*(besl(n-1,k)+besl(n+1,k))/2
            s += ds
            if abs(ds) < 1e-6:
                break
        z[m]=abs(ro*k*(besl(1,k)*t+s))
        r[m]=ro*np.exp(k*np.sin(t))
    z -= np.mean(z)
    r,z = geom.space(r,z,npoints)
    return r,z

if __name__ is '__main__':  # test shape function
    import seaborn as sns
    r1,r2 = 4.486,15.708  # DEMO baseline
    r,z = pD(r1,r2)
    
    pl.figure()
    pl.plot(r,z)
    pl.xlabel('$r$ [m]')
    pl.ylabel('$z$ [m]')
    pl.axis('equal')
    sns.despine()
    
    # check profile gradient
    ro=np.sqrt(r1*r2)
    k=0.5*np.log(r2/r1)
    
    l = np.linspace(1e-2,0.5-1e-2,100)
    L = np.append(0,np.cumsum(np.sqrt(np.diff(r)**2+np.diff(r)**2)))
    L /= L[-1]
    ro=np.sqrt(r1*r2)
    x,y = interp1d(L,r)(l)/ro,interp1d(L,z)(l)/ro
    dydx = np.gradient(y)/np.gradient(x)  # profile
    dydx_ = 1/k*np.log(x)/np.sqrt(1-1/k**2*np.log(x)**2)  # theory
    
    pl.figure()
    pl.plot(l,dydx)
    pl.plot(l,dydx_,'--')
    pl.legend(['profile','theory'],loc=2)
    pl.xlabel(r'normilized length, $l$')
    pl.ylabel(r'$\frac{dy}{dx}$')
    sns.despine()
    
    print('height',z.max()-z.min())
