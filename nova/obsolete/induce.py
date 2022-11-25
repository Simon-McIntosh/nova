from nova.frame.coilset import CoilSet
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
        'vector and scalar potential'
        Aphi = 1 / (2*np.pi) * self.a/self.r * \
            ((1 - self.k2/2) * self.K - self.E)  # 
        psi = 2 * np.pi * self.mu_o * self.r * Aphi  # scalar potential
        return psi
    

class Rectangle(Vectors):
    
    def __init__(self, *args):
        super().__init__(*args)
        
    def flux(self):
        'calculate flux for rectangular coil section'
        return np.zeros(len(self.r))
        
        
class BiotSavart:
    
    def __init__(self, *args):
        # assemble source and target
    
        # calculate seperation between source and target 
        # (low order / high order)
        
        # split source high order source coils dependant on type
        # (rectangle, polygon, shell)
        a=1
    #def flux
    
    
        

if __name__ == '__main__':
    
    x, z = 0.3, 0
    dl, dt = 0.5, 0.7
    
    cs = CoilSet()
    cs.add_coil(x, z, dl, dt, dCoil=-1, Nt=150, cross_section='skin',
                turn_section='rectangle', turn_fraction=1, Ic=40e3)
    
    #cs.add_coil(x, z, dl, dt, dCoil=-1, Nt=5, 
    #            cross_section='square', turn_section='skin',
    #            skin_fraction=0.7)
    #cs.plot(subcoil=True)
    cs.plot()
    
    cs.grid.generate_grid(expand=0, n=4e3)
    #cs.grid.plot_grid()
    cs.grid.plot_flux()
    
    f = Filament(cs.grid.source_m['x'], cs.grid.source_m['z'],
                 cs.grid.target_m['x'], cs.grid.target_m['z'])
    #cs.grid.flux = cs.grid.save_matrix(f.flux())[0]
    
    #cs.grid.flux = f.flux()
    cs.grid.plot_flux(color='C0', levels=cs.grid.levels)

    