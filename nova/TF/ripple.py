import nova.cross_coil as cc
import pylab as pl
import numpy as np
import seaborn as sns
from nova.config import Setup
from nova.streamfunction import SF
from amigo import geom
from scipy.linalg import norm
from scipy.optimize import minimize_scalar


rc = {'figure.figsize':[8,8],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
color = sns.color_palette('Set2')

#mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

class ripple(object):
    def __init__(self,nTF=18,**kwargs):
        self.nTF = nTF
        if 'plasma' in kwargs:
            self.get_seperatrix(alpha=0.95,**kwargs['plasma'])
        if 'coil' in kwargs:
            self.set_TFcoil(**kwargs['coil'])

    def get_seperatrix(self,nplasma=80,alpha=0.95,**kwargs):
        self.nplasma = nplasma
        self.plasma_loop = np.zeros((self.nplasma,3))  # initalise loop array
        if 'sf' in kwargs:  # seperatrix directly from sf object
            r,z = kwargs['sf'].get_boundary(alpha=alpha)
        elif 'config' in kwargs:
            setup = Setup(kwargs.get('config'))
            r,z = ripple.get_boundary(setup.filename,alpha=alpha)
        elif 'filename' in kwargs:  # seperatrix from eqdsk
            r,z = ripple.get_boundary(kwargs.get('filename'),alpha=alpha)
        elif 'r' in kwargs and 'z' in kwargs:  # separatrix from input
            r,z = kwargs.get('r'),kwargs.get('z')
        else:
            errtxt = '\n'
            errtxt += 'Require plasma={} input of following types:\n'
            errtxt += '1) configuration flag, {\'config\':\'SN\'} \n'
            errtxt += '2) eqdsk filename, {\'filename\':\'\'} \n'
            errtxt += '3) seperatrix profile, {\'r\':[],\'z\':[]}'
            raise ValueError(errtxt)
        r,z = geom.clock(r,z,anti=True)
        (self.plasma_loop[:,0],self.plasma_loop[:,2]) = \
        geom.rzSLine(r,z,npoints=self.nplasma)
        self.plasma_length = geom.length(self.plasma_loop[:,0],
                                         self.plasma_loop[:,2])
        rfun,zfun = geom.rzfun(r,z)
        self.plasma_interp = {'r':rfun,'z':zfun}
      
    def get_boundary(filename,alpha=0.95):
        sf = SF(filename)
        r,z = sf.get_boundary(alpha=alpha)
        return r,z
     
    def set_TFcoil(self,Rcl,Zcl,npoints=80):
        self.coil_loop = np.zeros((npoints,3))
        self.coil_loop[:,0],self.coil_loop[:,2] = \
        geom.rzSLine(Rcl,Zcl,npoints=npoints)  # coil centerline
        self.green_feild_loop = cc.GreenFeildLoop(self.coil_loop,smooth=False)

    def point_ripple(self,s):  # s==3D point vector
        B = np.zeros(2)
        n = np.array([0,1,0])
        for j,theta in enumerate([0,np.pi/self.nTF]):  # rotate (inline, ingap)
            sr = np.dot(s,geom.rotate(theta))
            nr = np.dot(n,geom.rotate(theta))
            Bo = np.zeros(3)
            for tcoil in np.linspace(0,2*np.pi,self.nTF,endpoint=False):
                Bo += cc.mu_o*self.green_feild_loop.B(sr,theta=tcoil)
            B[j] = norm(Bo)
            B[j] = np.dot(nr,Bo)
        ripple = 1e2*(B[0]-B[1])/(B[1]+B[0])
        return ripple
  
    def loop_ripple(self):
        self.ripple = np.zeros(self.nplasma)
        for i,plasma_point in enumerate(self.plasma_loop):
            self.ripple[i] = self.point_ripple(plasma_point)
            
    def ripple_opp(self,x):
        s = np.zeros(3)
        s[0],s[2] = self.plasma_interp['r'](x),self.plasma_interp['z'](x)
        ripple = self.point_ripple(s)
        return -ripple
        
    def get_ripple(self):
        self.res = minimize_scalar(self.ripple_opp,method='bounded',
                                   bounds=[0,1],options={'xatol':0.1})
        self.res['fun'] *= -1  # negate minimum (max ripple)
        return self.res['fun']
        
    def plot_loops(self,scale=1.5):
        self.loop_ripple()
        self.get_ripple()
        rpl,zpl = self.plasma_loop[:,0],self.plasma_loop[:,2]
        rr,zr = geom.offset(rpl,zpl,scale*self.ripple)
        for i in range(self.nplasma):  # ripple sticks
            pl.plot([rpl[i],rr[i]],[zpl[i],zr[i]],lw=1,color=0.5*np.ones(3))
        pl.plot(self.plasma_loop[:,0],self.plasma_loop[:,2],'-')
        pl.plot(self.coil_loop[:,0],self.coil_loop[:,2])
        x = self.res['x']
        pl.plot(self.plasma_interp['r'](x),self.plasma_interp['z'](x),'.',
                ms=10)
        pl.axis('equal')
        pl.axis('off')

    def plot_ripple(self):
        pl.figure(figsize=(8,4))
        pl.plot(self.plasma_length,self.ripple)
        pl.plot(self.res['x'],self.res['fun'],'o')

if __name__ is '__main__':  
    from nova.coils import PF,TF
    
    tf = TF(shape={'config':'DEMO_SN','coil_type':'S'})
    
    rp = ripple(plasma={'config':'DEMO_SN'},coil={'Rcl':tf.Rcl,'Zcl':tf.Zcl})  
    
    import time
    tic = time.time()
    print('ripple {:1.3f}%'.format(rp.get_ripple()))
    print('time',time.time()-tic)
    

    rp.plot_loops()

    '''
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    
    for j,theta in enumerate([0,np.pi/rp.nTF]):  # rotate (inline, ingap)
        plr = np.dot(rp.plasma_loop,geom.rotate(theta))
        ax.plot(plr[:,0],plr[:,1],plr[:,2])
    
    for tcoil in np.linspace(0,2*np.pi,rp.nTF,endpoint=False):
        cl = np.dot(rp.coil_loop,geom.rotate(tcoil))
        ax.plot(cl[:,0],cl[:,1],cl[:,2])

    ax.auto_scale_xyz([-20,20],[-20,20],[-20,20])
    '''
