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
from amigo import geom

class SALOME(object):
    
    def __init__(self,config,family='S',nTF=16,obj='L'):
        self.nTF = nTF
        datadir = trim_dir('../../Data') 
        file = 'salome_input' #  config #  +'_{}{}{}'.format(family,nTF,obj)
        self.filename = datadir+'/'+file+'.json'
        self.profile = Profile(config['TF'],family=family,load=True,part='TF',
                               nTF=nTF,obj=obj)
        setup = Setup(config['eq'])
        self.sf = SF(setup.filename)
        self.tf = TF(self.profile,sf=self.sf)   
        self.pf = PF(self.sf.eqdsk)
        self.PF_support()  # calculate PF support seats
        self.CS_support()  # calculate CS support seats
        self.Gravity_support()
        self.cage = coil_cage(nTF=nTF,rc=self.tf.rc,
                              plasma={'config':config['eq']},
                              coil=self.tf.x['cl'])
        
    def write(self):
        print('writing',self.filename,self.nTF)
        color = sns.color_palette('Set2',12)
        data = {'p':self.profile.loop.p,'section':self.tf.section,
                'pf':self.pf.coil,'nTF':self.nTF,'color':color,
                'PFsupport':self.PFsupport,'CSsupport':self.CSsupport,
                'Gsupport':self.Gsupport,'OISsupport':self.OISsupport}

        with open(self.filename,'w') as f:
            json.dump(data,f,indent=4)

    def CS_support(self):
        depth = self.tf.section['winding_pack']['depth']
        side = self.tf.section['case']['side']
        width = self.tf.section['winding_pack']['width']
        inboard = self.tf.section['case']['inboard']
        nose = self.tf.section['case']['nose']
        segment = self.profile.loop.p[0]
        ro,zo = segment['p0']['r'],segment['p0']['z']
        theta = np.pi/self.nTF
        rsep = (depth/2+side)/np.tan(theta)
        rnose = ro-(width+inboard+nose)
        rwp = ro-(width+inboard)
        if rsep <= rnose:
            ynose = depth/2+side
        else:
            ynose = rnose*np.tan(theta)
        if rsep <= rwp:
            ywp = depth/2+side
        else:
            ywp = rwp*np.tan(theta)
        
        self.tf.loop_interpolators(offset=0)  # construct TF interpolators
        TFloop = self.tf.fun['out']
        L = minimize_scalar(SALOME.cs_top,method='bounded',
                            args=(rwp,TFloop),bounds=[0.5,1]).x
        ztop = float(TFloop['z'](L))  
        self.CSsupport = {'rnose':rnose,'ynose':ynose,'rwp':rwp,'ywp':ywp,
                          'ztop':ztop,'zo':zo,'dt':side}

    def cs_top(L,rwp,TFloop):
        err = abs(TFloop['r'](L)-rwp)
        return err
    
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
        
    def connect(self,coil,loop,edge=0.15,hover=0.1,argmin=60):
        L = minimize_scalar(SALOME.support_arm,method='bounded',
                                args=(coil,loop),bounds=[0,1]).x 
        rTF,zTF = loop['r'](L),loop['z'](L)                    
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
            xo = np.array([L,0.5])
            res = minimize(SALOME.intersect,xo,method='L-BFGS-B', 
                           bounds=([0,1],[0,15]),args=(xc,nhat,loop)) 
            rs,zs = res.x[1]*nhat+xc
            nodes[3-i] = [rs,zs]
        return nodes
        
    def PF_support(self):
        self.tf.loop_interpolators(offset=-0.15)  # construct TF interpolators
        TFloop = self.tf.fun['out']
        self.PFsupport = {}
        for name in self.pf.index['PF']['name']:
            coil = self.pf.coil[name]
            nodes = self.connect(coil,TFloop,edge=0.15,hover=0.1,argmin=60)
            self.PFsupport[name] = nodes

    def GS_placement(L,radius,TFloop):
        return abs(radius-TFloop['r'](L))

    def Gravity_support(self,radius=13,width=0.75):
        self.tf.loop_interpolators(offset=-0.15)  # construct TF interpolators
        TFloop = self.tf.fun['out']
        self.tf.loop_interpolators(offset=0)
        Sloop = self.tf.fun['out']
        L = minimize_scalar(SALOME.GS_placement,method='bounded',
                                args=(radius-width/2,Sloop),bounds=[0,0.5]).x
        coil = {'r':Sloop['r'](L)+width/2,'z':Sloop['z'](L)-width/2,
                'dr':width,'dz':width}
        nodes = self.connect(coil,TFloop,edge=0,hover=0,argmin=90)
        self.Gsupport = {'base':nodes}
        z = [[self.pf.coil[name]['z']-self.pf.coil[name]['dz']/2]\
             for name in self.pf.coil]
        floor = np.min(z)-1
        self.Gsupport['zbase'] = float(Sloop['z'](L))
        self.Gsupport['zfloor'] = floor
        self.Gsupport['radius'] = radius
        self.Gsupport['width'] = width

    def OIS_placment(L,TFloop,point):
        err = (point[0]-TFloop['r'](L))**2+(point[1]-TFloop['z'](L))**2
        return err

    def OIS(self,width=3.5,thickness=0.15,rmin=10):
        self.tf.loop_interpolators(offset=0)  # construct TF interpolators
        TFloop = self.tf.fun['cl']
        dl = width/TFloop['L']
        self.OISsupport = {}
        for coil in self.PFsupport:
            node = self.PFsupport[coil]
            r = np.mean([node[2][0],node[3][0]])
            z = np.mean([node[2][1],node[3][1]])
            if r>rmin and ('Coil1' in coil or 'Coil4' in coil):
                L = minimize_scalar(SALOME.OIS_placment,method='bounded',
                                args=(TFloop,(r,z)),bounds=[0,1]).x
                rcl = np.array([TFloop['r'](L-dl/2),TFloop['r'](L+dl/2)])
                zcl = np.array([TFloop['z'](L-dl/2),TFloop['z'](L+dl/2)])
                ro,zo = np.mean(rcl),np.mean(zcl)
                L = minimize_scalar(SALOME.OIS_placment,method='bounded',
                                    args=(TFloop,(ro,zo)),bounds=[0,1]).x
                dr,dz = TFloop['r'](L)-ro,TFloop['z'](L)-zo
                rcl += dr/2
                zcl += dz/2
                dt = np.array([rcl[1]-rcl[0],0,zcl[1]-zcl[0]])
                dt /= np.linalg.norm(dt)
                dn = np.cross(dt,np.array([0,1,0]))
                
                rcl = np.append(rcl+dn[0]*thickness/2,
                                rcl[::-1]-dn[0]*thickness/2)
                zcl = np.append(zcl+dn[2]*thickness/2,
                                zcl[::-1]-dn[2]*thickness/2)
                nodes = [[rcl[i],zcl[i]] for i in range(4)]
                self.OISsupport[coil] = nodes

    def plot(self):
        self.tf.fill()
        self.pf.plot(coils=self.pf.coil,label=True,current=True)
        for name in self.PFsupport:
            nodes = np.array(self.PFsupport[name])
            geom.polyfill(nodes[:,0],nodes[:,1],color=0.4*np.ones(3))
        nodes = np.array(self.Gsupport['base'])
        geom.polyfill(nodes[:,0],nodes[:,1],color=0.4*np.ones(3))
        pl.plot(self.Gsupport['radius']*np.ones(2),
                [self.Gsupport['zbase'],self.Gsupport['zfloor']],'o-',
                color=0.4*np.ones(3),lw=4)
        for name in self.OISsupport:  
            nodes = np.array(self.OISsupport[name])             
            geom.polyfill(nodes[:,0],nodes[:,1],color=0.4*np.ones(3))
        rnose = self.CSsupport['rnose']
        rwp = self.CSsupport['rwp']
        zo = self.CSsupport['zo']
        ztop = self.CSsupport['ztop']
        rCS = [rnose,rwp,rwp,rnose]
        zCS = [zo,zo,ztop,ztop]
        geom.polyfill(rCS,zCS,color=0.4*np.ones(3))

        
if __name__ is '__main__':
    
    nPF,nTF = 3,16
    config = {'TF':'SN','eq':'SN_{:d}PF_{:d}TF'.format(nPF,nTF)}
    sal = SALOME(config,nTF=nTF)
    
    sal.OIS()
    sal.write()
    sal.plot()
    
    demo = DEMO()
    demo.fill_part('Vessel')
    demo.fill_part('Blanket')
    demo.plot_ports()
  
    rp,zp = demo.port['P4']['right']['r'],demo.port['P4']['right']['z']


                                    
    pl.plot(rp,zp,'k',lw=3)
    xc = [rp[0],zp[0]]
    nhat = np.array([rp[1]-rp[0],zp[1]-zp[0]])
    sal.tf.loop_interpolators(offset=0)  # construct TF interpolators
    loop = sal.tf.fun['out']

    Lo = minimize_scalar(SALOME.OIS_placment,method='bounded',
                         args=(loop,(rp[0],zp[0])),bounds=[0,1]).x
    xo = [Lo,0]             
    L = minimize(SALOME.intersect,xo,method='L-BFGS-B', 
                 bounds=([0.1,0.9],[0,15]),args=(xc,nhat,loop)).x
    pl.plot(loop['r'](L[0]),loop['z'](L[0]),'o')
    
    pl.plot(loop['r'](0.4),loop['z'](0.4),'d')
    pl.plot(loop['r'](0.5),loop['z'](0.5),'d')
    pl.plot(loop['r'](0.6),loop['z'](0.6),'d')
    pl.plot(loop['r'](0.7),loop['z'](0.7),'d')
    pl.plot(loop['r'](0.8),loop['z'](0.8),'d')
    


    
    
