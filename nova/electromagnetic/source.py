from nova.electromagnetic.coilset import CoilSet
from scipy.special import ellipk, ellipe
import numpy as np


class Vectors(object):
    
    mu_o = 4 * np.pi * 1e-7  # magnetic constant [Vs/Am]
    
    def __init__(self, *args):
        'store source and target points'
        self.load_geometry(*args[:4])  # source / target
        
        #if len(args) == 6:
        #    self.dx, self.dz = args[4:6]
            
    def load_geometry(self, rs, zs, r, z):
        self.rs = rs  # source radius
        self.zs = zs  # soruce height
        self.r = r  # target radius
        self.z = z  # target height
        self._calculate_coefficents()
        
    def _calculate_coefficents(self):
        self.gamma = self.zs - self.z
        self.a2 = self.gamma**2 + (self.r + self.rs)**2
        self.a = np.sqrt(self.a2)
        self.k2 = 4 * self.r * self.rs / self.a2  # modulus
        self.ck2 = 1 - self.k2  # complementary modulus
        self._K, self._E = None, None  # initialize elliptic integrals
        
    @property
    def K(self):  
        'complete elliptic integral of the first kind'
        if self._K is None:
            self._K = ellipk(self.k2)
        return self._K
    
    @property
    def E(self):
        'complete elliptic integral of the second kind'
        if self._E is None:
            self._E = ellipe(self.k2)
        return self._E
        

class Filament(Vectors):
    'complete circular filaments'
    def __init__(self, *args):
        super().__init__(*args)

    def flux(self):
        'vector potential in phi'
        Aphi = 1 / (2 * np.pi) * self.a / self.r * ((1 - self.k2 / 2) * self.K - self.E)
        Apsi = 2 * np.pi * self.mu_o * self.r * Aphi  # scalar potential
        return Apsi
    


class Rectangular(Vectors):
    
    def __init__(self, *args):
        super().__init__(*args)
        
        

if __name__ == '__main__':
    
    x, z = 0.5, 0
    dl, dt = 0.5, 0.1
    
    cs = CoilSet()
    cs.add_coil(x, z, dl, dt, dCoil=-1, Nt=12, cross_section='rectangle',
                turn_section='rectangle', turn_fraction=1, subcoil=True)
    cs.plot(subcoil=True)
    #cs.plot()
    


    
    '''
    cs.Ic = 1
    cs.grid.generate_grid(expand=1, n=1e3)
    #cs.grid.plot_flux(lw=3)
    
 
    xo = np.exp(1/dx * ((x + dx/2) * np.log(x + dx/2) - 
                        (x - dx/2) * np.log(x - dx/2) - dx))
    xo = np.sqrt(np.mean(cs.subcoil.x**2))
    
    #xo = np.sqrt(cs.coil.x**2 + cs.coil.dx**2 / 12)  # square
    xo = np.sqrt(cs.coil.x**2 + cs.coil.dx**2 / 16)  # circle
    xo = np.sqrt(cs.coil.x**2 + cs.coil.dx**2 / 8)  # circle  # skin
    
    print(xo, gmd(cs.subcoil.x, cs.subcoil.Nt), np.sqrt(np.mean(cs.subcoil.x**2)))
    
    cs_ = CoilSet(**cs.coilset)
    cs_.coil.x = xo
    #cs_.coil.xm = xo
    cs_.meshcoil(dCoil=0)
    #cs_.plot()
    
    
    #cs_.Ic = cs.Ic
    
    cs_.grid.solve_interaction()
    
    
    f = Filament(cs_.grid.source_m['x'], cs_.grid.source_m['z'],
                 cs_.grid.target_m['x'], cs_.grid.target_m['z'])
    
    cs_.grid.flux = f.flux()
    
    #cs_.grid.plot_flux(color='C0', levels=cs.grid.levels)
    '''
    