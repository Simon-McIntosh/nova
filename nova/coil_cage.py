import nova.cross_coil as cc
import pylab as pl
import numpy as np
import seaborn as sns
from nova.config import Setup
from nova.streamfunction import SF
from amigo import geom
from scipy.optimize import minimize_scalar
from nova import geqdsk
from nova.inductance import neumann
import time
from nova.coils import TF
from nova.DEMOxlsx import DEMO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class coil_cage(object):
    def __init__(self,nTF=18,rc=0.5,ny=3,nr=1,alpha=1-1e-4,**kwargs):
        self.nTF = nTF  # TF coil number
        self.ny = ny  #  winding pack depth discritization
        self.nr = nr  # winding pack radial discritization
        self.rc = rc
        if 'plasma' in kwargs:
            self.get_seperatrix(alpha=alpha,**kwargs['plasma'])
        if 'coil' in kwargs:
            self.set_TFcoil(kwargs['coil'])
            
    def get_seperatrix(self,nplasma=80,alpha=1-1e-3,plot=False,**kwargs):
        self.nplasma = nplasma
        self.plasma_loop = np.zeros((self.nplasma,3))  # initalise loop array
        if 'sf' in kwargs:  # seperatrix directly from sf object
            sf = kwargs['sf']
            r,z = sf.get_boundary(alpha=alpha)
            self.eqdsk = sf.eqdsk
        elif 'setup' in kwargs:
            setup = kwargs.get('setup')
            sf = SF(setup.filename)
            r,z = sf.get_boundary(alpha=alpha)
            self.eqdsk = sf.eqdsk
        elif 'config' in kwargs:
            setup = Setup(kwargs.get('config'))
            sf = SF(setup.filename)
            r,z = sf.get_boundary(alpha=alpha)
            self.eqdsk = sf.eqdsk
        elif 'eqdsk' in kwargs:  # seperatrix from eqdsk
            self.eqdsk = geqdsk.read(kwargs.get('eqdsk'))
            r,z = self.eqdsk['rbdry'],self.eqdsk['zbdry']
        elif 'r' in kwargs and 'z' in kwargs:  # separatrix from input
            r,z = kwargs.get('r'),kwargs.get('z')
            self.eqdsk = {'rcentr':9.0735,'zmagx':0.15295,'bcentr':-5.6211}
        else:
            errtxt = '\n'
            errtxt += 'Require plasma={} input of following types:\n'
            errtxt += '1) configuration flag, {\'config\':\'SN\'} \n'
            errtxt += '2) eqdsk file, {\'eqdsk\':\'\'} \n'
            errtxt += '3) seperatrix profile, {\'r\':[],\'z\':[]}'
            raise ValueError(errtxt)
        r,z = geom.clock(r,z)
        (self.plasma_loop[:,0],self.plasma_loop[:,2]) = \
        geom.rzSLine(r,z,npoints=self.nplasma)
        self.plasma_length = geom.length(self.plasma_loop[:,0],
                                         self.plasma_loop[:,2])
        rfun,zfun = geom.rzfun(r,z)
        self.plasma_interp = {'r':rfun,'z':zfun}
        if plot:
            pl.plot(self.plasma_loop[:,0],self.plasma_loop[:,2])
 
    def set_TFcoil(self,cl,smooth=False):
        r,z = geom.clock(cl['r'],cl['z'])
        self.npoints = len(r)
        self.coil_loop = np.zeros((self.npoints,3))
        self.coil_loop[:,0],self.coil_loop[:,2] = \
        geom.rzSLine(r,z,npoints=self.npoints)  # coil centerline
        self.gfl = cc.GreenFeildLoop(self.coil_loop,rc=self.rc,smooth=smooth)
        self.pattern()
        self.amp_turns()  # set coil cage amp-turns

    def pattern(self,plot=False):  # generate winding-pack pattern
        ro = self.coil_loop[0,0]  # inboard centreline
        rlim = 0.6*np.max(self.coil_loop[:,0])
        dL = ro*np.tan(np.pi/self.nTF)  # toroidal winding-pack width
        if self.ny > 1:
            self.dYwp = np.linspace(dL*(1/self.ny-1),dL*(1-1/self.ny),self.ny) 
        else:
            self.dYwp = [0]  # coil centreline
        self.Tcoil = np.linspace(0,2*np.pi,self.nTF,endpoint=False)
        if plot:
            fig = plt.figure(figsize=(8,6))
            ax1 = fig.add_subplot(121,projection='3d')
            ax1.axis('equal')
            ax1.axis('off')
            ax2 = fig.add_subplot(122)
            ax2.axis('equal')
            ax2.axis('off')
            color = sns.color_palette('Set2',self.nTF)
            for i,tcoil in enumerate(self.Tcoil):
                for dy in self.dYwp:  # wp pattern
                    loop = self.gfl.transform(tcoil,dy)[0]  # close loop
                    loop = np.append(loop,np.reshape(loop[0,:],(1,-1)),axis=0)  
                    ax1.plot(loop[:,0],loop[:,1],loop[:,2],color=color[i])
                    ax2.plot(loop[:,0],loop[:,1],color=color[i])     
            ax1.set_xlim(-rlim,rlim)
            ax1.set_ylim(-rlim,rlim)
            ax1.set_zlim(-rlim,rlim)
            ax2.set_xlim(-rlim,rlim)
            ax2.set_ylim(-rlim,rlim)   
            pl.tight_layout()
            
    def amp_turns(self):
        rc,zc = self.eqdsk['rcentr'],self.eqdsk['zmagx']
        self.Iturn = 1
        Bo = self.point((rc,0,zc),variable='feild')
        self.Iturn = self.eqdsk['bcentr']/Bo[1]  # single coil amp-turns

    def point(self,s,variable='ripple'):  # s==3D point vector 
        B = np.zeros(2)
        n = np.array([0,1,0])
        if variable == 'ripple':
            planes = [0,np.pi/self.nTF]  # rotate (inline, ingap)
        elif variable == 'feild':
            planes = [0] 
        else:
            errtxt = '\n'
            errtxt += 'point variable error, require \'ripple\' or \'feild\'\n'
            raise ValueError(errtxt)
        for j,theta in enumerate(planes):
            sr = np.dot(s,geom.rotate(theta))
            nr = np.dot(n,geom.rotate(theta))
            Bo = np.zeros(3)
            for tcoil in self.Tcoil:
                for dy in self.dYwp:  # wp pattern
                    Bo += self.Iturn*cc.mu_o*\
                    self.gfl.B(sr,theta=tcoil,dy=dy)/self.ny
            B[j] = np.dot(nr,Bo)
        if variable == 'ripple':
            ripple = 1e2*(B[0]-B[1])/np.sum(B)
            return ripple
        else:
            return Bo
            
    def edge_ripple(self,npoints=10):
        ripple = np.zeros(npoints)
        L = np.linspace(0.2,0.8,npoints)
        for i,l in enumerate(L):
            r,z = self.plasma_interp['r'](l),self.plasma_interp['z'](l)
            ripple[i] = self.point((r,0,z),variable='ripple')
        return ripple

    def loop_ripple(self):
        self.ripple = np.zeros(self.nplasma)
        for i,plasma_point in enumerate(self.plasma_loop):
            self.ripple[i] = self.point(plasma_point,variable='ripple')
            
    def ripple_opp(self,x):
        s = np.zeros(3)
        s[0],s[2] = self.plasma_interp['r'](x),self.plasma_interp['z'](x)
        ripple = self.point(s,variable='ripple')
        return -ripple
        
    def get_ripple(self):
        self.res = minimize_scalar(self.ripple_opp,method='bounded',
                                   bounds=[0,1],options={'xatol':0.1})
        self.res['fun'] *= -1  # negate minimum (max ripple)
        return self.res['fun']
        
    def get_volume(self):
        V_TF = geom.loop_vol(self.coil_loop[:,0],self.coil_loop[:,2])
        V_pl = geom.loop_vol(self.plasma_loop[:,0],self.plasma_loop[:,2])
        ratio = V_TF/V_pl
        return {'TF':V_TF,'plasma':V_pl,'ratio':ratio}
    
    def output(self):  # output cage parameters to screen
        Vol = self.get_volume()
        print('ripple {:1.3f}'.format(self.get_ripple()))
        print('TF energy {:1.2f} GJ'.format(1e-9*self.energy()))
        print(r'TF centre-line enclosed volume {:1.0f} m3'.format(Vol['TF']))
        print(r'plasma enclosed volume {:1.0f} m3'.format(Vol['plasma']))
        print('TF/plasma volume ratio {:1.2f}'.format(Vol['ratio']))
        
    def plot_loops(self,scale=1.5,sticks=False):
        self.loop_ripple()
        ripple = self.get_ripple()
        #pl.plot(self.res['x'],self.res['fun'],'o',color=0.8*np.ones(3))
        rpl,zpl = self.plasma_loop[:,0],self.plasma_loop[:,2]
        rr,zr = geom.offset(rpl,zpl,scale*self.ripple)
        color = sns.color_palette('Set2',2)
        if sticks:  # ripple sticks
            for i in range(self.nplasma):
                pl.plot([rpl[i],rr[i]],[zpl[i],zr[i]],
                        color=color[0],lw=3)
        #pl.plot(self.plasma_loop[:,0],self.plasma_loop[:,2],'-')
        #pl.plot(self.coil_loop[:,0],self.coil_loop[:,2])
        x = self.res['x']
        pl.plot(self.plasma_interp['r'](x),self.plasma_interp['z'](x),'o',
                ms=8,color=0.5*np.ones(3))
        pl.text(self.plasma_interp['r'](x),
                self.plasma_interp['z'](x),
                '{:1.2f}%- '.format(ripple),ha='right',va='center',
                color=0.5*np.ones(3)) 
        pl.axis('equal')
        pl.axis('off')
        
    def plot_contours(self,variable='ripple',n=3e3,loop_offset=-1,**kwargs):
        if 'loop' in kwargs:
            loop_dict = kwargs['loop']
            loop = np.zeros((len(loop_dict['r']),3))
            loop[:,0] = loop_dict['r']
            loop[:,2] = loop_dict['z']
        else:
            loop = self.coil_loop
        xlim,dx = np.zeros((3,2)),np.zeros(3)
        xl = np.zeros((len(loop),3))
        xl[:,0],xl[:,2] = geom.offset(loop[:,0],loop[:,2],loop_offset)
        for i in range(3):
            xlim[i] = [np.min(xl[:,i]),np.max(xl[:,i])]
            dx[i] = xlim[i][1]-xlim[i][0]
        xlim[0][0] = self.plasma_loop[np.argmax(self.plasma_loop[:,2]),
                                      0]  # plasma top
        ar = dx[0]/dx[2]
        nz = int(np.sqrt(n/ar))
        nr = int(n/nz)    
        r = np.linspace(xlim[0][0],xlim[0][1],nr)
        z = np.linspace(xlim[2][0],xlim[2][1],nz)
        R,Z = np.meshgrid(r,z,indexing='ij')
        rpl = np.zeros((nr,nz))
        for i,r_ in enumerate(r):
            for j,z_ in enumerate(z):
                if geom.inloop(xl[:,0],xl[:,2],r_,z_):
                    if variable == 'ripple':
                        rpl[i,j] = self.point((r_,0,z_),variable=variable)
                    else:
                        rpl[i,j] = self.point((r_,0,z_),variable=variable)[1]
                else:
                    rpl[i,j] = np.NaN
    
        if variable == 'ripple':
            levels = 10**(np.linspace(np.log10(0.01),np.log10(5),7))
            levels = np.round(levels,decimals=2)
            
            geom.polyfill(self.plasma_loop[:,0],self.plasma_loop[:,2],
                  alpha=0.3,color=sns.color_palette('Set2',5)[3])
            self.get_ripple()  # get max ripple on plasma contour
            rpl_max = self.res['fun']
            iplasma = np.argmin(abs(np.log10(levels)-np.log10(rpl_max)))
            levels[iplasma] = rpl_max  # select edge contour
            CS = pl.contour(R,Z,rpl,levels=levels,colors=[0.6*np.ones(3)],lw=3)
            zc = CS.collections[iplasma]
            pl.setp(zc,color=[0.4*np.ones(3)])
            pl.clabel(CS,inline=1,fontsize='medium',fmt='%1.2f')
        else:
            pl.contour(R,Z,rpl)
            
    def energy(self):  # super-duper fast version
        nloop = int(np.floor(self.nTF/2))+1  # symetric
        theta = np.linspace(0,2*np.pi,self.nTF,endpoint=False)[:nloop]
        pair = 2*np.ones(nloop)
        pair[0] = 1
        pair[-1] = 1 if self.nTF%2 == 0 else 2
        A = np.zeros((self.gfl.npoints,3))
        for i,x in enumerate(self.gfl.loop):
            for p,t in zip(pair,theta):
                A[i,:] += p*cc.mu_o*self.gfl.A(x,theta=t)  # vector potential
        A[:,1] = 0  # zero out-of-plane values
        L = 0
        for i in range(self.gfl.npoints):
            L += np.dot(A[i],self.gfl.dL[i])
        E = 0.5*self.Iturn**2*self.nTF*L
        return E
        
    def energy_neumann(self,Jmax=7.2e7):  # slow
        nturn = 1
        self.Acs = self.Iturn/Jmax
        Theta = np.linspace(0,2*np.pi,self.nTF,endpoint=False)
        nloop = int(np.floor(self.nTF/2))+1  # symetric
        N = 2*np.ones(nloop)
        N[0] = 1
        if self.nTF%2 == 0:
            N[-1] = 1
        Theta = Theta[1:nloop]
        r = np.sqrt(self.Acs)
        neu = neumann(r=r,X=self.coil_loop)
        M = np.zeros((nloop))
        for i,(theta,n) in enumerate(zip(Theta,N)):
            neu.rotateX_(theta)
            M[i] = n*nturn**2*neu.calculate()
        E = 0.5*self.Iturn**2*self.nTF*np.sum(M)
        return E

if __name__ is '__main__':  
    
    rc = {'figure.figsize':[8,8],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
    sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
            font_scale=7/8,rc=rc)
    color = sns.color_palette('Set2')
    
    demo = DEMO()
    demo.fill_loops()
    

    
    config = 'DEMO_SN'
    
    tf = TF(config,coil_type='S',npoints=80)

    #tf.coil.set_input(inputs={'upper':0.8,'top':0.5})
   
    x = tf.coil.draw()
    tf.get_loops(x)
    tf.fill() 
    
    cage = coil_cage(nTF=18,rc=tf.rc,plasma={'config':config},
                     coil={'cl':tf.x['cl']})
    
    
    tic = time.time()
    print('ripple {:1.3f}%'.format(cage.get_ripple()))
    print('time {:1.3f}s'.format(time.time()-tic))


    print(cage.edge_ripple())
    
    #tf.coil.set_input()
    tic = time.time()   
    print('energy {:1.3f}GJ'.format(1e-9*cage.energy()))
    print('time A {:1.3f}s'.format(time.time()-tic))
    
    '''
    B = np.zeros((tf.npoints,3))
    for i,(r,z) in enumerate(zip(tf.x['cl']['r'],tf.x['cl']['z'])):
        B[i,:] = cage.Iturn*cage.point((r,0,z),variable='feild')
        
    npoints = 200
    rcl = np.linspace(np.min(tf.x['cl']['r']),np.max(tf.x['cl']['r']),npoints)
    zcl = cage.eqdsk['zmagx']*np.ones(npoints)
    Bcl = np.zeros((npoints,3))
    
    for i,(r,z) in enumerate(zip(rcl,zcl)):
        Bcl[i,:] = cage.Iturn*cage.point((r,0,z),variable='feild')
        
    pl.figure(figsize=(8,6))
    pl.plot(tf.x['cl']['r'],abs(B[:,1]))
    pl.plot(rcl,abs(Bcl[:,1]))
    pl.plot(cage.eqdsk['rcentr'],abs(cage.eqdsk['bcentr']),'o')
    sns.despine()
    '''

    
    #rp.plot_loops()


