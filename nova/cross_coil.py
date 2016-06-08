import numpy as np
from scipy.special import ellipk, ellipe

mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]
 
def green(R,Z,Rc,Zc):
    r = np.array((R-Rc)**2+(Z-Zc)**2)
    m = 4*R*Rc/((R+Rc)**2+(Z-Zc)**2)
    g = np.array((Rc*R)**0.5*((2*m**-0.5-m**0.5)*\
    ellipk(m)-2*m**-0.5*ellipe(m))/(2*np.pi))
    g[r==0] = 0
    return g
    
def green_feild(R,Z,Rc,Zc):
    feild = np.zeros(2)
    a = np.sqrt((R+Rc)**2+(Z-Zc)**2)
    m = 4*R*Rc/a**2
    I1 = 4/a*ellipk(m)
    I2 = 4/a**3*ellipe(m)/(1-m)
    A = (Z-Zc)**2+R**2+Rc**2
    B = -2*R*Rc
    feild[0] = Rc/(4*np.pi)*(Z-Zc)/B*(I1-A*I2)
    feild[1] = Rc/(4*np.pi)*((Rc+R*A/B)*I2-R/B*I1)
    return feild
    
def green_feild_loop(coil, point): 
    nArc,B = 100,[]

    theta,dtheta = np.linspace(0,2*np.pi,nArc,endpoint=False,retstep=True)  # angle
    c = np.transpose(np.array([coil['r']*np.cos(theta),
                               coil['r']*np.sin(theta),
                               np.array([coil['z']]*len(theta))
                               ]))  # position
    dL = coil['r']*np.transpose(np.array([-np.sin(theta),
                                                np.cos(theta),
                                                np.array([0]*len(theta))]))*dtheta  # segment
    p = np.array(point)*np.ones((len(theta),3))  # point
    r = p-c  # vectors
    
    r_mag = np.transpose(np.sum(r*r,axis=1)**0.5*np.ones((3,len(theta))))
    '''
    core = 1-np.exp(-r_mag**3/coil['rc']**3)  # magnet core
    core = 1
    for i,mag in enumerate(r_mag):
        if mag[0] < 1e-16:
            r_mag[i] = 3*[1e-16]
    '''
    #r_hat = r/r_mag
    B = coil['I']*mu_o/(4*np.pi)*np.sum(np.cross(dL,r)/r_mag**3,axis=0)  # Bfeild 
    return B
    
def Gtorque(eq_coil,sf_coil,source,sink,multi_filament):  # source-sink 
    if multi_filament:
        coil = eq_coil
        Nbundle = 1
        Nsource = coil[source+'_0']['Nf']
        Nsink = coil[sink+'_0']['Nf']
        name_source = lambda i:source+'_{:1.0f}'.format(i)
        name_sink = lambda j:sink+'_{:1.0f}'.format(j)
    else:  # single-filament
        coil = sf_coil
        Nbundle = eq_coil[source+'_0']['Nf']*eq_coil[sink+'_0']['Nf']
        Nsource,Nsink = 1,1
        name_source = lambda i:source
        name_sink = lambda j:sink
    feild = np.zeros(2)
    for i in range(Nsource):
        source_strand = name_source(i)  # source
        ri = coil[source_strand]['r']
        zi = coil[source_strand]['z']
        for j in range(Nsink):
            sink_strand = name_sink(j)  # sink
            r = coil[sink_strand]['r']
            z = coil[sink_strand]['z']
            rc = coil[sink_strand]['rc']
            r_mag = np.sqrt((r-ri)**2+(z-zi)**2)
            if r_mag > rc:  # outside coil core
                dfeild = green_feild(r,z,ri,zi)
            else:  # inside coil core
                dfeild,B = np.zeros(2), np.zeros(2)
                dz = (rc**2-(r-ri)**2)**0.5  # Br
                for i,zc in enumerate([zi-dz, zi+dz]):
                    B[i] = green_feild(r,z,ri,zc)[0]
                dfeild[0] = sum(B)/2 + (z-zi)*(B[1]-B[0])/(2*dz)
                dr = (rc**2-(z-zi)**2)**0.5  # Bz
                for i,rc in enumerate([ri-dr, ri+dr]):
                    B[i] = green_feild(r,z,rc,zi)[1]
                dfeild[1] = sum(B)/2 + (r-ri)*(B[1]-B[0])/(2*dr) 
            feild += Nbundle*r*dfeild  # feild couple, rG
    return feild
    
def Btorque(eq,passive_coils,sink):
    Csink = eq.coil
    Nsink = Csink[sink+'_0']['Nf']
    feild = np.zeros(2)
    for source in passive_coils:
        if source == 'Plasma':
            Csource = eq.plasma_coil
            Nsource = len(Csource)
        else:
            Csource = eq.coil
            Nsource = Csource[source+'_0']['Nf']
        for i in range(Nsource):
            source_strand = source+'_{:1.0f}'.format(i)
            ri = Csource[source_strand]['r']  # source
            zi = Csource[source_strand]['z']
            Ii = Csource[source_strand]['I']
            for j in range(Nsink):
                sink_strand = sink+'_{:1.0f}'.format(j)
                r = Csink[sink_strand]['r']  # sink
                z = Csink[sink_strand]['z']
                feild += Ii*r*green_feild(r,z,ri,zi)
    return feild
    
def Gfeild(coil,plasma_coil,point):
    feild = np.zeros(2)
    for coil_set in [coil,plasma_coil]:
        feild += Btorque(coil_set,point)
    return feild
    
def tangent(R,Z,norm=True):
    dR,dZ = np.diff(R),np.diff(Z)
    R,Z = (R[:-1]+R[1:])/2,(Z[:-1]+Z[1:])/2
    if norm:
        mag = np.sqrt(dR**2+dZ**2)
    else:
        mag = np.ones(len(R))
    index = mag>0
    dR,dZ,mag = dR[index],dZ[index],mag[index]  # clear duplicates
    R,Z = R[index],Z[index]
    return dR/mag,dZ/mag,R,Z
    
def normal(R,Z,norm=True):
    tR,tZ,R,Z = tangent(R,Z,norm=norm)
    t = np.zeros((len(R),3))
    t[:,0],t[:,1] = tR,tZ
    n = np.cross(t, [0,0,1])
    nR,nZ = n[:,0],n[:,1]
    return nR,nZ,R,Z

'''    
def Fcoil(coil,plasma_coil,name):
    point = (coil[name]['r'],coil[name]['z'])
    B = Bfeild(coil,plasma_coil,point)
    Fr = 2*np.pi*coil[name]['r']*B[1]*coil[name]['I']
    Fz = -2*np.pi*coil[name]['r']*B[0]*coil[name]['I']
    return Fr,Fz
'''
    
'''    
def tolist(var):
    if type(var) is not list:
        var = [var]
    return var
    
def Pcalc(r,z,ri,zi,I):
    m = 4*r*ri/((r+ri)**2+(z-zi)**2)
    psi = mu_o*I/(2*np.pi)*(ri*r)**0.5*\
        ((2*m**-0.5-m**0.5)*K(m)-2*m**-0.5*E(m))
    return psi

def Bfield(coil,r,ri,z,zi):
    feild = np.zeros(2)
    a = ((r+ri)**2+(z-zi)**2)**0.5
    m = 4*r*ri/a**2
    I1 = 4/a*K(m)
    I2 = 4/a**3*E(m)/(1-m)
    A = (z-zi)**2+r**2+ri**2
    B = -2*r*ri
    feild[0] = mu_o*coil['I']/(4*np.pi)*ri*(z-zi)/B*(I1-A*I2)
    feild[1] = mu_o*coil['I']/(4*np.pi)*ri*((ri+r*A/B)*I2-r/B*I1) 
    return feild
    

    

    
def Pcoil(coil,point):
    psi = 0
    for name in coil.keys():
        ri,zi = (coil[name]['r'], coil[name]['z'])
        r,z = point
        I = coil[name]['I']
        psi += Pcalc(r,z,ri,zi,I)
    return psi

def Bcoil_a(coil, point):

    nArc, B, Clist = 20, [], {} 
    for key in coil.keys():
        Clist[key] = tolist(coil[key])
        
    for cr, cz, I, rc in zip(Clist['r'],Clist['z'],Clist['I'],Clist['rc']):
        psi, dpsi = np.linspace(0, 2*np.pi, nArc, endpoint=False, retstep=True)  # angle
        c = np.transpose(np.array([cr*np.cos(psi),cr*np.sin(psi),np.array([cz]*len(psi))]))  # position
        dL = cr*np.transpose(np.array([-np.sin(psi),np.cos(psi),np.array([0]*len(psi))]))*dpsi  # segment
        p = np.array(point)*np.ones((len(psi),3))  # point
        r = p-c  # vectors
        r_mag = np.transpose(np.sum(r*r,axis=1)**0.5*np.ones((3,len(psi))))
        core = 1-np.exp(-r_mag**3/rc**3)  # magnet core
        for i,mag in enumerate(r_mag):
            if mag[0] < 1e-8:
                r_mag[i] = 3*[1e-8]
        r_hat = r/r_mag
        B.append(I*mu_o/(4*np.pi)*np.sum(core*np.cross(dL,r_hat)/r_mag**2, axis=0))  # Bfeild 
    return np.sum(B, axis=0)
    
    from xalglib import incompleteellipticintegralk as iK  # first kind
    from xalglib import incompleteellipticintegrale as iE  # second kind
        
    def elliptic(kind, r, rc, r_mag, m):
        if r_mag>rc:
            if 'K' in kind:
                intergral = K(m)    
            else:
                intergral = E(m)
        else:
            dtheta = rc/r*(1-r_mag/rc)/2
            if 'K' in kind:
                intergral = iK(np.pi/2-dtheta, m)    
            else:
                intergral = iE(np.pi/2-dtheta, m)
        return intergral
  
  def green_loop(coil, point): 
    nArc, B = 1000, []
    for name in coil.keys():
        #print(name)
        psi, dpsi = np.linspace(0, 2*np.pi, nArc, endpoint=False, retstep=True)  # angle
        c = np.transpose(np.array([coil[name]['r']*np.cos(psi),
                                   coil[name]['r']*np.sin(psi),
                                   np.array([coil[name]['z']]*len(psi))
                                   ]))  # position
        dL = coil[name]['r']*np.transpose(np.array([-np.sin(psi),
                                                    np.cos(psi),
                                                    np.array([0]*len(psi))]))*dpsi  # segment
        p = np.array(point)*np.ones((len(psi),3))  # point
        r = p-c  # vectors
        r_mag = np.transpose(np.sum(r*r,axis=1)**0.5*np.ones((3,len(psi))))
        core = 1-np.exp(-r_mag**3/coil[name]['rc']**3)  # magnet core

        for i,mag in enumerate(r_mag):
            #print(mag)
            if mag[0] < 1e-16:
                r_mag[i] = 3*[1e-16]
        #print('')
        r_hat = r/r_mag
        B.append(coil[name]['I']*mu_o/(4*np.pi)*
                 np.sum(core*np.cross(dL,r_hat)/r_mag**2, 
                 axis=0))  # Bfeild 
    return np.sum(B, axis=0)
    
def Bcoil_c(coil, point):
    feild = np.zeros(2)
    for name in coil.keys():
        ri,zi = (coil[name]['r'], coil[name]['z'])
        r,z = point
        r_mag = ((r-ri)**2+(z-zi)**2)**0.5

        if r_mag > 1e-6:
            if r_mag > coil[name]['rc']:
                dfeild = Bfield(coil[name], r, ri, z, zi)
            else:
                r_theta = np.arctan2(z-zi,r-ri)
                B = np.zeros((2,2))
                for i,theta in enumerate([0,np.pi]):
                    rc = ri+coil[name]['rc']*np.cos(theta+r_theta)
                    zc = zi+coil[name]['rc']*np.sin(theta+r_theta)
                    B[:,i] = Bfield(coil[name], rc, ri, zc, zi)
                dfeild = B[:,0] + (B[:,0]-B[:,1])/2*(r_mag/coil[name]['rc']-1)

        if r_mag > coil[name]['rc']:
            dfeild = Bfield(coil[name], r, ri, z, zi)
        else:
            if r_mag > 1e-3:
                r_theta = np.arctan2(z-zi,r-ri)
            else:
                r_theta = 0 
            B = np.zeros((2,2))
            for i,theta in enumerate([0,np.pi]):
                rc = ri+coil[name]['rc']*np.cos(theta+r_theta)
                zc = zi+coil[name]['rc']*np.sin(theta+r_theta)
                B[:,i] = Bfield(coil[name], rc, ri, zc, zi)
            dfeild = B[:,0] + (B[:,0]-B[:,1])/2*(r_mag/coil[name]['rc']-1)
         
        feild += dfeild
    return feild
'''

