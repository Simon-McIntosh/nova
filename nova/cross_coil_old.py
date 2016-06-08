import numpy as np
from xalglib import ellipticintegralk as K  # first kind
from xalglib import ellipticintegrale as E  # second kind

mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

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
    
def Bcoil(coil,point):
    feild = np.zeros(2)
    for name in coil.keys():
        ri,zi = (coil[name]['r'],coil[name]['z'])
        r,z = point
        r_mag = ((r-ri)**2+(z-zi)**2)**0.5

        if r_mag > coil[name]['rc']:
            dfeild = Bfield(coil[name],r,ri,z,zi)
        else:
            #print('incoil')
            dfeild,B = np.zeros(2), np.zeros(2)
            dz = (coil[name]['rc']**2-(r-ri)**2)**0.5  # Br
            for i,zc in enumerate([zi-dz, zi+dz]):
                B[i] = Bfield(coil[name], r, ri, zc, zi)[0]
            dfeild[0] = sum(B)/2 + (z-zi)*(B[1]-B[0])/(2*dz)
            dr = (coil[name]['rc']**2-(z-zi)**2)**0.5  # Bz
            for i,rc in enumerate([ri-dr, ri+dr]):
                B[i] = Bfield(coil[name], rc, ri, z, zi)[1]
            dfeild[1] = sum(B)/2 + (r-ri)*(B[1]-B[0])/(2*dr)
        feild += dfeild
    return feild
    
def Fcoil(coil,name):
    B = Bcoil(coil,[coil[name]['r'],coil[name]['z']])
    Fr = -2*np.pi*coil[name]['r']*B[1]*coil[name]['I']
    Fz = 2*np.pi*coil[name]['r']*B[0]*coil[name]['I']
    return Fr,Fz
    
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

def Bcoil_b(coil, point):
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


