from nova.TF.DEMOxlsx import DEMO
from nova.coils import PF,TF
from nova.loops import Profile
from nova.coil_cage import coil_cage
from nova.streamfunction import SF
from nova.config import Setup
from scipy.optimize import minimize_scalar,minimize
import numpy as np
import json
from nova.config import trim_dir
import seaborn as sns
import pylab as pl
from DEMOxlsx import DEMO

class SALOME(object):
    
    def __init__(self,config,family='S',nTF=16,obj='L'):
        self.nTF = nTF
        datadir = trim_dir('../../Data') 
        file = 'salome_input' #  config #  +'_{}{}{}'.format(family,nTF,obj)
        self.filename = datadir+'/'+file+'.json'
        self.profile = Profile(config['TF'],family=family,part='TF',
                               nTF=nTF,obj=obj)
        setup = Setup(config['eq'])
        self.sf = SF(setup.filename)
        self.tf = TF(self.profile,sf=self.sf)   
        self.pf = PF(self.sf.eqdsk)
        self.PF_support()  # calculate PF support seats
        self.cage = coil_cage(nTF=nTF,rc=self.tf.rc,
                              plasma={'config':config['eq']},
                              coil=self.tf.x['cl'])
        
    def support_arm(L,coil,TFloop):
        dl = np.sqrt((coil['r']-TFloop['r'](L))**2+
                     (coil['z']-TFloop['z'](L))**2)
        return dl
    
    def intersect(x,xc,nhat,TFloop):
        L,s = x  # unpack
        rTF,zTF = TFloop['r'](L),TFloop['z'](L)  
        rs,zs = s*nhat+xc
        err = np.sqrt((rTF-rs)**2+(zTF-zs)**2)
        return err
        
    def PF_support(self,edge=0.15,hover=0.1,argmin=60):
        self.tf.loop_interpolators(offset=-0.15)  # construct TF interpolators
        TFloop = self.tf.fun['out']
        self.PFsupport = {}
        for name in self.pf.PF_coils:
            coil = self.pf.coil[name]
            L = minimize_scalar(SALOME.support_arm,method='bounded',
                                args=(coil,TFloop),bounds=[0,1]).x 
            rTF,zTF = TFloop['r'](L),TFloop['z'](L)                    
            nhat = np.array([rTF-coil['r'],zTF-coil['z']])
            ndir = 180/np.pi*np.arctan(abs(nhat[1]/nhat[0]))  # angle, deg
            if ndir < argmin:  # limit absolute support angle
                nhat = np.array([np.sign(nhat[0]),
                                 np.tan(argmin*np.pi/180)*np.sign(nhat[1])])
            nhat /= np.linalg.norm(nhat)
            above = np.sign(np.dot(nhat,[0,1]))
            zc = coil['z']+above*(coil['dz']/2+hover)
            nodes = [[] for _ in range(4)]
            for i,sign in enumerate([-1,1]): #  inboard / outboard
                rc = coil['r']+sign*(coil['dr']/2+edge)
                nodes[i] = [rc,zc]
                xc = np.array([rc,zc])
                xo = np.array([L,0.3])
                res = minimize(SALOME.intersect,xo,method='L-BFGS-B', 
                               bounds=([0,1],[0,5]),args=(xc,nhat,TFloop)) 
                rs,zs = res.x[1]*nhat+xc
                nodes[3-i] = [rs,zs]
            self.PFsupport[name] = nodes

    def OIS(self):
        self.tf.loop_interpolators(offset=0)  # construct TF interpolators
        TFloop = self.tf.fun['cl']
        lo = 0.66
        dL = 5
        dl = dL/TFloop['L']
        print(dL,dl)
        for delta in [-1,0,1]:
            pl.plot(TFloop['r'](lo+delta*dl/2),TFloop['z'](lo+delta*dl/2),'o')

    #def CSsupport(self):
        
                                
                    
    def write(self):
        print('writing',self.filename,self.nTF)
        color = sns.color_palette('Set2',12)
        data = {'p':self.profile.loop.p,'section':self.tf.section,
                'pf':self.pf.coil,'nTF':self.nTF,'color':color,
                'PFsupport':self.PFsupport}
        with open(self.filename,'w') as f:
            json.dump(data,f,indent=4)

    def plot(self):
        self.tf.fill()
        self.pf.plot(coils=self.pf.coil,label=True,current=True)
        for name in self.PFsupport:
            nodes = np.array(self.PFsupport[name])
            pl.plot(nodes[:,0],nodes[:,1],'.-')
        
if __name__ is '__main__':
    
    config = {'TF':'SN','eq':'SN_3PF_12TF'}
    sal = SALOME(config,nTF=12)
    
    sal.OIS()
    sal.write()
    sal.plot()
    
    demo = DEMO()
    demo.fill_part('Vessel')
    demo.fill_part('Blanket')
    pl.plot(demo.limiter['L3']['r'],demo.limiter['L3']['z'])
  
    pl.plot(sal.tf.x['out']['r'],sal.tf.x['out']['z'])

    

    
    
