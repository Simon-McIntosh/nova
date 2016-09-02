import nova.cross_coil as cc
import pylab as pl
import numpy as np
import seaborn as sns
from nova.config import Setup
from nova.streamfunction import SF
from nova.coils import TF
from amigo import geom
from amigo.addtext import linelabel
from scipy.linalg import norm
from scipy.optimize import minimize 

rc = {'figure.figsize':[8,4],'savefig.dpi':100, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
color = sns.color_palette('Set2')

mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

class ripple(object):
    def __init__(self,plasma={},coil={}):
        self.get_seperatrix(alpha=0.95,**plasma)
        self.get_coil(**coil)
        
    def get_seperatrix(self,nsep=100,alpha=0.95,**kwargs):
        self.plasma_loop = np.zeros((nsep,3))  # initalise loop array
        if 'config' in kwargs:
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
        (self.plasma_loop[:,0],self.plasma_loop[:,2]) = \
        geom.rzSLine(r,z,npoints=nsep)
        
    def get_boundary(filename,alpha=0.95):
        sf = SF(filename)
        r,z = sf.get_boundary(alpha=alpha)
        return r,z
     

    def get_coil(self,npoints=100,**kwargs):
        self.coil_loop = np.zeros((npoints,3))

        tf = TF(shape=kwargs)  # load tf coil
        self.coil_loop[:,0],self.coil_loop[:,2] = \
        geom.rzSLine(tf.Rcl,tf.Zcl,npoints=npoints)  # TFcoil centerline
        
        '''
            r,z = tf.Rmid,tf.Zmid
            self.nTF = tf.nTF
            self.Iturn = tf.Iturn
        elif 'config' in kwargs:
            tf = TF(shape={'dataname':kwargs.get('dataname')})
            print('loaded dataname')
        '''
        #
        
   
    def plot_loops(self):
        pl.plot(self.plasma_loop[:,0],self.plasma_loop[:,2])
        pl.plot(self.coil_loop[:,0],self.coil_loop[:,2])
        
    def point_ripple(self,s):  # s==3D point vector
        B = np.zeros(2)
        for j,theta in enumerate([0,np.pi/self.nTF]):  # rotate (inline, ingap)
            sr = np.dot(s,geom.rotate(theta))
            Bo = np.zeros(3)
            for tcoil in np.linspace(0,2*np.pi,self.nTF):
                loopR = np.dot(loop,geom.rotate(tcoil))
                Bo += mu_o*self.Iturn*cc.green_feild_loop(loopR,sr)
            B[j] = norm(Bo)
        ripple = (B[0]-B[1])/(B[1]+B[0])
        return ripple

rip = ripple(plasma={'config':'SN'},coil={'config':'tmp','coil_type':'S'})  
rip.plot_loops()
'''
# prepare coil loop
Nl = 100  # coil segment number
Ns = 100  # sep segment number

text = linelabel(Ndiv=10,value='1.1f',postfix='%',loc='max')

for conf in ['SXex']:  #  ,'SXex','SFp','SFm','X','SX'
    
    sf = SF(setup.filename)
    tf = TF(setup=setup)
    #tf.energy()
    


    B = np.zeros((Ns,2))

    
    
    #minimize(fun,x0,args=(),bounds=None)

    for i,s in enumerate(sep):

    ripple = 1e2*(B[:,0]-B[:,1])/(B[:,1]+B[:,0])
    pl.plot(ripple)
    text.add(conf)
    
    print(conf,'ripple {:1.1f}% E {:1.0f}GJ'.format(np.max(ripple),
                                                    1e-9*tf.Ecage))

text.plot()
sns.despine()
'''