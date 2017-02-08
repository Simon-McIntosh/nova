Yfrom nova.DEMOxlsx import DEMO
from nova.coils import PF,TF
from nova.loops import Profile
from nova.coil_cage import coil_cage
from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.config import Setup
from scipy.optimize import minimize_scalar,minimize
import numpy as np
import json
from amigo.IO import trim_dir
import seaborn as sns
import pylab as pl
from amigo import geom
from amigo.ANSYS import table
from nova.force import force_feild
import sys

class OCC(object):
    
    def __init__(self,config,family='S',nTF=16,obj='L'):
        self.nTF = nTF
        self.config = config
        datadir = trim_dir('../../Data') 
        file = 'occ_input' #  config #  +'_{}{}{}'.format(family,nTF,obj)
        self.filename = datadir+'/'+file+'.json'
        self.profile = Profile(config['TF'],family=family,load=True,part='TF',
                               nTF=nTF,obj=obj,npoints=250)
        setup = Setup(config['eq'])
        self.sf = SF(setup.filename)
        self.tf = TF(self.profile,sf=self.sf)   
        self.pf = PF(self.sf.eqdsk)
        self.PF_support()  # calculate PF support seats
        self.CS_support()  # calculate CS support seats
        self.Gravity_support()
        self.cage = coil_cage(nTF=nTF,rc=self.tf.rc,ny=3,
                              plasma={'config':config['eq']},
                              coil=self.tf.x['cl'])
        self.eq = EQ(self.sf,self.pf,dCoil=0.5,sigma=0,
                     boundary=self.sf.get_sep(expand=0.4),n=2e3) 
        self.eq.plasma()
        self.ff = force_feild(self.pf.index,self.pf.coil,
                              self.eq.coil,self.eq.plasma_coil)
        
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
        L = minimize_scalar(OCC.cs_top,method='bounded',
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
        L = minimize_scalar(OCC.support_arm,method='bounded',
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
            res = minimize(OCC.intersect,xo,method='L-BFGS-B', 
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
        L = minimize_scalar(OCC.GS_placement,method='bounded',
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
        
    def draw_OIS(self,L,width,thickness,TFloop):
        dl = width/TFloop['L']
        rcl = np.array([TFloop['r'](L-dl/2),TFloop['r'](L+dl/2)])
        zcl = np.array([TFloop['z'](L-dl/2),TFloop['z'](L+dl/2)])
        ro,zo = np.mean(rcl),np.mean(zcl)
        L = minimize_scalar(OCC.OIS_placment,method='bounded',
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
        return nodes

    def OIS(self,width=3.5,thickness=0.15,rmin=10):
        self.tf.loop_interpolators(offset=0)  # construct TF interpolators
        TFloop = self.tf.fun['cl']
        self.OISsupport = {}
        for i,(L,width) in enumerate(zip([0.4,0.64],[4.5,2.5])):
            nodes = self.draw_OIS(L,width,thickness,TFloop)                    
            self.OISsupport['OIS{:d}'.format(i)] = nodes

    def plot(self):
        self.tf.fill()
        self.pf.plot(coils=self.pf.coil,label=True,current=True)
        self.pf.plot(coils=self.eq.coil,plasma=True)
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
        self.ff.plot()
        
    def ansys(self,plot=False,nl=250,nr=5,ny=5):
        datadir = trim_dir('../../Data/')
        filename = datadir+'TFload_{:d}'.format(self.nTF)
        ans = table(filename)
        ans.f.write('! loading tables for {:d}TF coil concept\n'.format(self.nTF))
        ans.f.write('! loop length parameterized from 0-1\n')
        ans.f.write('! loop starts at the inboard midplane\n')
        ans.f.write('! loop progresses in the anti-clockwise direction\n')
        ans.f.write('! tables defined with cylindrical coordinate system\n')
        ans.f.write('! body applied to nodes of winding-pack in cartisean system x,y,z\n')
        ans.f.write('! winding-pack must be labled as named-selection \'wp\'\n')
        ans.f.write('\nlocal,11,1,{:1.9f},0,{:1.9f},0,90,0'\
                    .format(self.sf.mo[0],self.sf.mo[1]))
        ans.f.write('  ! define local cylindrical coordinate system\n')
        ans.f.write('csys,0  ! restore to cartesian\n')
        
        ans.f.write('\n! per-TF PF coil forces (Fr,Fz) [N]\n')
        ans.f.write('! order as numbered in plots\n')
        F = self.ff.get_force()['F']
        ans.load('F_coil',1e6*F/self.nTF)
        ans.write_array()
        
        apdlstr = '''
        nPF = 4
        *do,i,1,nPF
          F,c%i-1%,Fz,1e6*F_coil(i,2)
        *enddo
        F,cs,Fz,1e6*F_coil(i+1,2)
        '''
        

        ans.f.write('\n/nopr  ! suppress large table output\n')
        
        self.tf.loop_interpolators(offset=0,full=True)  # construct TF interpolators
        TFloop = self.tf.fun['cl']
        
        ngrid = {'nr':20,'nt':150} #  coordinate interpolation grid
        ndata = {'nl':nl,'nr':nr,'ny':ny} #  coordinate interpolation grid
        l = np.linspace(0,1,250)  # calculate grid extent
        xin,zin = self.tf.fun['in']['r'](l),self.tf.fun['in']['z'](l)
        rin = np.sqrt((xin-self.sf.mo[0])**2+(zin-self.sf.mo[1])**2)
        rmin = np.min(rin)  # minimum radius
        xout,zout = self.tf.fun['out']['r'](l),self.tf.fun['out']['z'](l)
        rout = np.sqrt((xout-self.sf.mo[0])**2,(zout-self.sf.mo[1])**2)
        rmax = np.max(rout)  # maximum radius

        radius = np.linspace(rmin,rmax,ngrid['nr'])
        theta = np.linspace(-np.pi,np.pi,ngrid['nt'])
        l_map = np.zeros((ngrid['nr'],ngrid['nt']))
        dr_map = np.zeros((ngrid['nr'],ngrid['nt']))
        for i in range(ngrid['nr']):
            for j in range(ngrid['nt']):
                x = self.sf.mo[0]+radius[i]*np.cos(theta[j])
                z = self.sf.mo[1]+radius[i]*np.sin(theta[j])
                L = minimize_scalar(OCC.OIS_placment,method='bounded',
                                        args=(TFloop,(x,z)),bounds=[0,1]).x
                xl,zl = TFloop['r'](L),TFloop['z'](L)
                l_map[i,j] = L
                dr_map[i,j] = np.sqrt((x-xl)**2+(z-zl)**2)*\
                np.sign(np.dot([x-xl,z-zl],[x-self.sf.mo[0],z-self.sf.mo[1]]))
                
        width = self.tf.section['winding_pack']['width']
        depth = self.tf.section['winding_pack']['depth']
        cross_section = width*depth
        l_data = np.linspace(0,1,ndata['nl'])
        if ndata['nr']>1:
            dr_data = np.linspace(-width/2,width/2,ndata['nr'])
        else:
            dr_data = np.array([0])
        if ndata['ny']>1:
            dy_data = np.linspace(-depth/2,depth/2,ndata['ny'])
        else:
            dy_data = np.array([0])
        Fbody = {}
        for var in ['x','y','z']:
            Fbody[var] = np.zeros((ndata['nl'],ndata['nr'],ndata['ny']))
        
        self.tf.loop_interpolators(offset=0,full=True)  # centreline
        Jturn = self.cage.Iturn/cross_section  # current density magnitude
        for i,l in enumerate(l_data): 
            iter_str = '\rcalculating TF body force:'
            iter_str += 'segment {:d} of {:d}'.format(i,ndata['nl'])
            sys.stdout.write(iter_str)
            sys.stdout.flush()
            xo = self.tf.fun['cl']['r'](l)
            zo = self.tf.fun['cl']['z'](l)
            dxo = self.tf.fun['cl']['dr'](l)
            dzo = self.tf.fun['cl']['dz'](l)
            that = np.array([dxo,0,dzo])
            that /= np.linalg.norm(that)
            J = Jturn*that  # current density vector
            nhat = np.array([that[2],0,-that[0]])
            for j,dr in enumerate(dr_data):
                for k,dy in enumerate(dy_data):
                    point = np.array([xo,dy,zo])+dr*nhat
                    Fb = self.ff.topple(point,J,self.cage,self.eq.Bpoint,
                                        method='BS')  # body force
                    for m,var in enumerate(['x','y','z']):  # store
                        Fbody[var][i,j,k] = Fb[m]
                    if plot:
                        Fvec = 1e-8*Fb#/np.linalg.norm(Fb)
                        #Fvec = that
                        pl.arrow(point[0],point[2],Fvec[0],Fvec[2],
                                 head_width=0.15,head_length=0.3)       
        print('\n',np.sum(Fbody['x'][:-1,:,:])*1e-9,
              np.sum(Fbody['y'][:-1,:,:])*1e-9,
              np.sum(Fbody['z'][:-1,:,:])*1e-9)

        ans.f.write('\n! parametric coil length, fn(theta)\n')
        ans.load('l_map',l_map,[radius,theta])
        ans.write(['radus','theta'])
        ans.f.write('\n! parametric coil offset, fn(theta)\n')
        ans.load('dr_map',dr_map,[radius,theta])
        ans.write(['radus','theta'])
        for var in ['x','y','z']:
            ans.f.write('\n! winding-pack body force, Fbody_{} [N/m3]\n'.format(var))
            ans.load('Fbody_{}'.format(var),Fbody[var],
                     [l_data,dr_data,dy_data])
            ans.write(['l_map','dr_map','offset'])
        ans.f.write('/gopr  ! enable output\n')    
            
        apdlstr = '''
pi = 4*atan(1)
csys,11  ! switch to cylindrical coordinate system
esel,s,elem,,wp  ! select winding pack (requires named selection 'wp')
*get,nel,elem,0,count
*vget,el_sel,elem,,esel  ! selection mask
*vget,el_id,elem,,elist  ! winding pack selection array
*vget,el_vol,elem,,geom  ! element volume
*vget,el_radius,elem,,cent,x  ! element radius
*vget,el_theta,elem,,cent,y  ! element theta
*vget,el_offset,elem,,cent,z  ! element axial offset
*voper,el_theta,el_theta,mult,pi/180  ! convert to radians
csys,0  ! return coordinate system

! compress selections
*dim,el_v,array,nel
*dim,el_r,array,nel
*dim,el_t,array,nel
*dim,el_o,array,nel
*dim,el_l,array,nel
*dim,el_dr,array,nel

*vmask,el_sel
*vfun,el_v,comp,el_vol  ! volume
*vmask,el_sel
*vfun,el_r,comp,el_radius  ! radius
*vmask,el_sel
*vfun,el_t,comp,el_theta  ! theta
*vmask,el_sel
*vfun,el_o,comp,el_offset  ! offset
*vitrp,el_l,l_map,el_r,el_t  ! interpolate l_map table
*vitrp,el_dr,dr_map,el_r,el_t !  interpolate dr_map table

xyz = 'x','y','z'  ! axes
fcum,add  ! accumulate nodal forces
*do,i,1,nel  ! apply forces to loads
  esel,s,elem,,el_id(i)
  nsle  ! select nodes attached to element
  nsel,r,node,,wp  ! ensure all nodes from winding pack
  *get,nnd,node,0,count  ! count nodes
  *do,j,1,3  ! Fx,Fy,Fz - all nodes attached to element
      F,all,F%xyz(j)%,Fbody_%xyz(j)%(el_l(i),el_dr(i),el_o(i))*el_v(i)/nnd
  *enddo
*enddo
allsel  
        '''
        ans.f.write(apdlstr)
        ans.close()

        
if __name__ is '__main__':
    
    #nPF,nTF = 3,13
    #config = {'TF':'SN','eq':'SN_{:d}PF_{:d}TF'.format(nPF,nTF)}

    nTF,nPF,nCS = 13,4,1
    #config = {'TF':'NTP','eq':'SN'}

    config = {'TF':'dtt','eq':'SN'}
    config['TF'] = '{}{}{:d}'.format(config['eq'],config['TF'],nTF)
    config['eq'] = 'SNdtt{:d}_{:d}PF_{:d}CS'.format(nTF,nPF,nCS)

    occ = OCC(config,nTF=nTF)
    
    occ.OIS()
    occ.write()
    occ.ansys(plot=True,nl=50,nr=1,ny=1)
    occ.plot()
    
    #pl.savefig('../Figs/CoilForces.png')


    
    
