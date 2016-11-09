import nova.cross_coil as cc
import pylab as pl
import numpy as np
import matplotlib

class force_feild(object):
    
    def __init__(self,index,pf_coil,eq_coil,eq_plasma_coil,Iscale=1e6,
                 multi_filament=True,plot=False,**kwargs):
        self.index = index
        self.pf_coil = pf_coil
        self.eq_coil = eq_coil
        self.eq_plasma_coil = eq_plasma_coil
        self.Iscale = Iscale  # current units (MA)
        self.nC = len(self.pf_coil)
        self.active_coils = kwargs.get('active_coils',list(self.pf_coil.keys()))
        self.passive_coils = kwargs.get('passive_coils',['Plasma'])
        self.set_force_feild(state='both',multi_filament=multi_filament)
        self.set_current()
        if plot:
            self.plot()
        
    def check(self):
        if not self.force_feild_active:
            errtxt = 'no force feild\n'
            errtxt += 'set_force_feild\n'
            raise ValueError(errtxt)
        if self.index['PF']['n'] == 0:
            errtxt = 'PF_coils empty\n'
            raise ValueError(errtxt)
        if self.index['CS']['n'] == 0:
            errtxt = 'CS_coils empty\n'
            raise ValueError(errtxt)
   
    def set_force_feild(self,state='both',multi_filament=False):
        # [I]T([Fa][I]+[Fp]) = F
        self.force_feild_active = True
        if state == 'both' or state == 'active':
            self.set_active_force_feild(multi_filament=multi_filament)
        if state == 'both' or state == 'passive':
            self.set_passive_force_feild()

    def set_active_force_feild(self,multi_filament=False):  
        self.Fa = np.zeros((self.nC,self.nC,2))  # active
        for i,sink in enumerate(self.active_coils):
            for j,source in enumerate(self.active_coils):
                rG = cc.Gtorque(self.eq_coil,self.pf_coil,
                                source,sink,multi_filament)*self.Iscale**2
                self.Fa[i,j,0] = 2*np.pi*cc.mu_o*rG[1]  #  cross product
                self.Fa[i,j,1] = -2*np.pi*cc.mu_o*rG[0]
        
    def set_passive_force_feild(self):
        self.Fp = np.zeros((self.nC,2))  # passive
        for i,sink in enumerate(self.active_coils):
            rB = cc.Btorque(self.eq_coil,self.eq_plasma_coil,
                            self.passive_coils,sink)*self.Iscale
            self.Fp[i,0] = 2*np.pi*cc.mu_o*rB[1]  #  cross product
            self.Fp[i,1] = -2*np.pi*cc.mu_o*rB[0] 

    def set_force(self,I):  # evaluate coil force and force jacobian 
        self.F = np.zeros((self.nC,2))
        dF = np.zeros((self.nC,self.nC,2))
        Im = np.dot(I.reshape(-1,1),np.ones((1,self.nC)))  # current matrix
        for i in range(2):  # coil force (bundle of eq elements)
            self.F[:,i] = 1e-6*(I*(np.dot(self.Fa[:,:,i],I)+self.Fp[:,i]))  # MN
            dF[:,:,i] = Im*self.Fa[:,:,i]       
            diag = np.dot(self.Fa[:,:,i],I)+\
            I*np.diag(self.Fa[:,:,i])+self.Fp[:,i]
            np.fill_diagonal(dF[:,:,i],diag)
        dF *= 1e-6  # force jacobian MN/MA
        return self.F,dF
            
    def get_force(self):
        self.check()
        F,dF = self.set_force(self.I)
        Fcoil = {'PF':{'r':0,'z':0},'CS':{'sep':0,'zsum':0},'F':F,'dF':dF}    
        Fcoil['PF']['r'] = np.max(abs(F[self.index['PF']['index'],0]))
        Fcoil['PF']['z'] = np.max(abs(F[self.index['PF']['index'],1]))
        FzCS = F[self.index['CS']['index'],1] 
        if self.index['CS']['n'] > 1:
            Fsep = np.zeros(self.index['CS']['n']-1)  # seperation force
            for j in range(self.index['CS']['n']-1):  # evaluate each gap
                Fsep[j] = np.sum(FzCS[j+1:])-np.sum(FzCS[:j+1])
            Fcoil['CS']['sep'] = np.max(Fsep)
        Fcoil['CS']['zsum'] = np.sum(FzCS)
        return Fcoil
        
    def set_current(self):  # set eq current vector from pf
        self.I = np.zeros(self.nC)
        for i,name in enumerate(self.active_coils):
            Nfilament = self.eq_coil[name+'_0']['Nf']
            self.I[i] = self.pf_coil[name]['I']/Nfilament  # store current
        self.I /= self.Iscale  # convert current unit (MA)
        
    def plot(self,scale=3):
        fs = matplotlib.rcParams['legend.fontsize']
        if not hasattr(self,'F'):
            self.set_force(self.I)
        Fmax = np.max(np.linalg.norm(self.F,axis=1))
        for i,name in enumerate(self.pf_coil):
            r,z = self.pf_coil[name]['r'],self.pf_coil[name]['z']
            F = self.F[i]
            pl.arrow(r,z,scale*F[0]/Fmax,0,  # Fr
                     linewidth=2,head_width=0.4,head_length=0.6,
                     ec=0.4*np.ones(3),fc=0.7*np.ones(3))
            pl.arrow(r,z,0,scale*F[1]/Fmax,  # Fz
                     linewidth=2,head_width=0.4,head_length=0.6,
                     ec=0.4*np.ones(3),fc=0.7*np.ones(3))
            if name in self.index['PF']['name']:
                zarrow = scale*F[1]/Fmax
                if abs(zarrow) < self.pf_coil[name]['dz']:
                    zarrow = np.sign(zarrow)*self.pf_coil[name]['dz']
                if zarrow > 0:
                    va = 'bottom'
                else:
                    va = 'top' 
                pl.text(r,z+zarrow,'{:1.2f}MN'.format(F[1]),
                        ha='center',va=va,fontsize=1.1*fs,color=0.5*np.ones(3))

if __name__ is '__main__':  # test functions
    from nova.config import Setup
    from nova.streamfunction import SF
    from nova.elliptic import EQ
    from nova.coils import PF
    from nova.radial_build import RB
    
    setup = Setup('SN_3PF_18TF')
    
    sf = SF(setup.filename)
    pf = PF(sf.eqdsk)
    rb = RB(setup,sf)
    eq = EQ(sf,pf,dCoil=0.25,sigma=0,boundary=rb.get_fw(expand=0.25),n=1e3)
    eq.get_plasma_coil()
    #eq.gen_opp()
    
    ff = force_feild(pf.index,pf.coil,eq.coil,eq.plasma_coil)
    Fcoil = ff.get_force()
    
    ff.plot()
    pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 
    pf.plot(coils=eq.coil,label=False,plasma=True,current=False) 
    pl.axis('equal')
    pl.axis('off')
    print(Fcoil['F'][:,1])
    '''
    print('writing',self.filename,self.nTF)

    data = {'p':self.profile.loop.p,'section':self.tf.section,
            'pf':self.pf.coil,'nTF':self.nTF,'color':color,
            'PFsupport':self.PFsupport,'CSsupport':self.CSsupport}
    with open(self.filename,'w') as f:
        json.dump(data,f,indent=4)
    '''
    