from dataclasses import dataclass, field

from nova.electromagnetic.biotset import BiotSet


@dataclass
class BiotFilament(BiotSet):
    """Extend BiotSet, compute interaction for complete circular filaments."""

    name = 'filament'  # applicable cross section type

    psi:
    radial_field:
    vertical_field:
        
    def __post_init__(self):
        super().__post_init__()
        self.source_radius = self.source('rms')
        #source_height
        #target_radius
        #target_height    
    
        #self.initialize_filaments(source, target)
        #self.offset_filaments(source)
        #self.calculate_coefficients()

    def initialize_filaments(self, source, target):
        self.rs, self.zs = source._rms_, source._z_  # source
        self.r, self.z = target._x_, target._z_  # target

    def offset_filaments(self, source, n_fold=0, n_merge=1,
                         rms_offset=True):
        """
        Offset source and target filaments.

        Parameters
        ----------
        source : BiotFrame
            Source filament biotframe.
        n_fold : float, optional
            Number of e-foling lenghts within filament. The default is 1.
        n_merge : float, optional
            Merge radius, multiple of filament widths. The default is 1.25.
        rms_offset : bool, optional
            Maintain rms offset for filament pairs. The default is False.

        Returns
        -------
        None.

        """
        # extract interaction
        df = np.max([source._dx_, source._dz_], axis=0) / 2
        dL = np.array([(self.r-self.rs), (self.z-self.zs)])
        dL_mag = np.linalg.norm(dL, axis=0)
        # select filaments within merge radius
        idx = np.where(dL_mag <= df*n_merge)[0]
        # reduce
        dL_mag = dL_mag[idx]
        dL = dL[:, idx]
        df = df[idx]
        ro = source._dx_[idx]*source._cs_factor_[idx]/2  # self seperation
        # interacton orientation
        index = np.isclose(dL_mag, 0)
        dL_norm = np.zeros((2, len(index)))
        dL_norm[0, index] = 1  # radial offset
        dL_norm[:, ~index] = dL[:, ~index] / dL_mag[~index]
        if n_fold == 0:
            factor = (1 - dL_mag / (df*n_merge))  # linear blending
        else:
            factor = np.exp(-n_fold*(dL_mag/df)**2)  # exponential blending
        dr = factor*ro*dL_norm[0, :]  # radial offset
        dz = factor*ro*dL_norm[1, :]  # vertical offset
        if rms_offset:
            drms = -(self.r[idx]+self.rs[idx])/4 + np.sqrt(
                (self.r[idx]+self.rs[idx])**2 -
                8*dr*(self.r[idx] - self.rs[idx] + 2*dr))/4
            self.rs[idx] += drms
            self.r[idx] += drms
        # offset source filaments
        self.rs[idx] -= dr/2
        self.zs[idx] -= dz/2
        # offset target filaments
        self.r[idx] += dr/2
        self.z[idx] += dz/2

    def _offset_filaments(self):
        'offset source and target points'
        # point seperation
        dL = np.array([(self.r-self.rs), (self.z-self.zs)])
        dL_mag = np.linalg.norm(dL, axis=0)
        dr = self.dl/2  # filament characteristic radius
        ro = dr*self.cross_section_factor  # self seperation

        # zero-seperation
        index = np.isclose(dL_mag, 0)
        dL_norm = np.zeros((2, self.nI))
        dL_norm[0, index] = 1  # radial offset
        dL_norm[:, ~index] = dL[:, ~index] / dL_mag[~index]
        # initalize offsets
        dr, dz = np.zeros(self.nI), np.zeros(self.nI)

        # mutual offset
        nx = dL[0] / self.drs
        nz = dL[1] / self.dzs
        mutual_index = np.where((nx <= 5) & (nz <= 5))  # mutual index
        mutual_factor = self.gmr.evaluate(nx[mutual_index], nz[mutual_index])
        dr[mutual_index] = (mutual_factor-1) * dL[0, mutual_index]
        dz[mutual_index] = (mutual_factor-1) * dL[1, mutual_index]

        # self inductance index
        self_index = np.where(dL_mag <= ro)  # seperation < dl/2
        # self_dr = self.dl[self_index]/2  # filament characteristic radius
        # self_ro = self_dr*self.cross_section_factor[self_index]  # seperation
        self_ro = ro[self_index]
        self_factor = 1 - dL_mag[self_index]/self_ro
        dr[self_index] = self_factor*self_ro*dL_norm[0, self_index]  # radial
        dz[self_index] = self_factor*self_ro*dL_norm[1, self_index]  # vertical

        # rms offset
        drms = -(self.r+self.rs)/4 + np.sqrt((self.r+self.rs)**2 -
                                             8*dr*(self.r - self.rs + 2*dr))/4
        self.rs += drms
        self.r += drms
        # offset source filaments
        self.rs -= dr/2
        self.zs -= dz/2
        # offset target filaments
        self.r += dr/2
        self.z += dz/2

    def calculate_coefficients(self):
        self.b = self.rs + self.r
        self.gamma = self.zs - self.z
        self.a2 = self.gamma**2 + (self.r + self.rs)**2
        self.a = np.sqrt(self.a2)
        self.k2 = 4 * self.r * self.rs / self.a2
        self.ck2 = 1 - self.k2  # complementary modulus
        self.K = ellipk(self.k2)  # first complete elliptic integral
        self.E = ellipe(self.k2)  # second complete elliptic integral

    def scalar_potential(self):
        'vector and scalar potential'
        Aphi = 1 / (2*np.pi) * self.a/self.r * \
            ((1 - self.k2/2) * self.K - self.E)  # Wb/Amp-turn-turn
        psi = 2 * np.pi * mu_o * self.r * Aphi  # scalar potential
        return psi

    def radial_field(self):
        Br = mu_o / (2*np.pi) * self.gamma * (
            self.K - (2-self.k2) / (2*self.ck2) * self.E) / (self.a*self.r)
        return Br  # T / Amp-turn-turn

    def vertical_field(self):  # T / Amp-turn-turn
        Bz = mu_o / (2*np.pi) * (self.r*self.K - \
            (2*self.r - self.b*self.k2) /
            (2*self.ck2) * self.E) / (self.a*self.r)
        return Bz  # T / Amp-turn-turn


if __name__ == '__main__':
    
    source = {'x': [3, 3.4, 3.6], 'z': [3.1, 3, 3.3],
          'dl': 0.3, 'dt': 0.3, 'section': 'hex'}
    biotfilament = BiotFilament(source, source)
