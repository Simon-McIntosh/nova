from read_dina import dina
from amigo.IO import readtxt
import numpy as np
from amigo.pyplot import plt
from nova.streamfunction import SF
import matplotlib.animation as manimation
from amigo.time import clock
from nep.VS.VSgeom import VS
from nep.DINA.read_plasma import read_plasma


class read_psi:

    def __init__(self, database_folder='disruptions'):
        self.dina = dina(database_folder)

    def read_file(self, folder):
        self.folder = folder
        self.filename = self.dina.locate_file('psi_data', folder=folder)
        self.name = self.filename.split('\\')[-2]
        self.read_header()
        self.initalise_arrays()
        self.read_scalars()
        self.vs = VS()  # load VS coil object
        self.pl = read_plasma(self.dina.database_folder)
        self.pl.read_file(self.folder)  # load time trace

    def read_header(self):
        # read grid - calculate file size
        with readtxt(self.filename) as self.rt:
            index = self.rt.readline(True)
            self.nfw, self.nzero, _, self.nsep, self.nflux = index
            self.fw = np.zeros((self.nfw, 2))

            self.rt.skipnumber(2*self.nzero)  # skip zeros
            self.fw[:, 0] = 1e-2*self.rt.readarray(self.nfw)
            self.fw[:, 1] = 1e-2*self.rt.readarray(self.nfw)
            self.nline_to_scalars = self.rt.nline  # number of lines to scalars
            self.rt.readarray(8)  # block parameters
            self.npsi = self.rt.skipblock(ncol=6)  # scalar potential
            self.nx = self.rt.skipblock(ncol=6)  # x grid dimension
            self.nz = self.rt.skipblock(ncol=6)  # z grid dimension
            self.nline_to_sep = self.rt.nline  # number of lines to seperatrix

            if self.nx * self.nz != self.npsi:  # check psi grid dimension
                txt = '\nmissmatch between psi data and xz grid\n'
                txt += 'nx*nz {}, npsi {}'.format(self.nx*self.nz, self.npsi)
                raise ValueError(txt)

            self.rt.skipnumber(2*self.nsep)  # skip seperatrix
            self.rt.skipnumber(2*self.nflux)  # skip flux functions
            self.nline_index = self.rt.nline  # line number for timestamp

    def initalise_arrays(self):
        with open(self.filename) as f:  # count total line number
            for self.nline, l in enumerate(f):
                pass
            self.nline += 1
        if self.nline % self.nline_index != 0:
            txt = 'file length is not an interger multiple'
            txt += 'of block length'
            raise ValueError(txt)

        self.nt = int(self.nline/self.nline_index)  # number of timestamps
        self.scalar_dtype = [('dx', float), ('dz', float), ('pmag', float),
                             ('pbound', float), ('ps', float),
                             ('xo', float), ('zo', float), ('t', float)]
        self.scalars = np.zeros(self.nt, dtype=self.scalar_dtype)
        self.psi = np.zeros((self.nx, self.nz))
        self.sep = np.zeros((self.nsep, 2))
        self.xmp = np.zeros(self.nflux)  # x coordinate of midplane
        self.Jphi = np.zeros(self.nflux)  # toroidal current density

        self.x = np.ones(self.nx)  # read x-z grid
        self.z = np.ones(self.nz)
        with readtxt(self.filename) as self.rt:
            self.rt.skiplines(self.nline_to_scalars+2)
            self.rt.skipblock(ncol=6)  # scalar potential
            self.x = 1e-2*self.rt.readarray(self.nx)  # cm - m
            self.z = 1e-2*self.rt.readarray(self.nz)  # cm - m
        self.read_single_array(0)  # read first time index
        self.update_sf()

    def read_scalars(self):
        with readtxt(self.filename) as self.rt:
            for time_index in range(self.nt):
                self.rt.skiplines(self.nline_to_scalars)
                scalar_data = self.rt.readarray(8)
                for key, value in zip(self.scalars.dtype.names, scalar_data):
                    if key in ['dx', 'dz', 'xo', 'zo']:
                        value *= 1e-2  # cm to meters
                    if key in ['pmag', 'pbound', 'ps']:
                        value *= 1e-5  # kGcm2 to Tm2
                    if key == 't':
                        value *= 1e-3  # ms to s
                    self.scalars[time_index][key] = value
                self.rt.skiplines(self.nline_index -
                                  self.nline_to_scalars - 2)

    def read_array(self):
        self.time_index = int(self.rt.nline / self.nline_index)
        self.rt.skiplines(self.nline_to_scalars+2)
        self.psi = self.rt.readarray(self.npsi).reshape((self.nz, self.nx))
        self.psi = 1e-5*self.psi.T  # kGcm2/rad to Webber/rad
        self.x = 1e-2*self.rt.readarray(self.nx)  # cm - m
        self.z = 1e-2*self.rt.readarray(self.nz)  # cm - m
        for i in range(2):
            self.sep[:, i] = 1e-2*self.rt.readarray(self.nsep)  # cm - m
        self.xmp = 1e-2*self.rt.readarray(self.nflux)  # cm - m
        self.Jphi = 1e4*self.rt.readarray(self.nflux)  # Acm-2 - Am-2

    def read_single_array(self, time_index=0):
        if time_index > self.nt-1:
            txt = '\nrequested index {:d} '.format(time_index)
            txt += 'greater than file length {:d}'.format(self.nt-1)
            raise ValueError(txt)
        nff = time_index*self.nline_index  # fast-forward
        with readtxt(self.filename) as self.rt:
            if nff > 0:  # skip data blocks
                self.rt.skiplines(nff)
            self.read_array()

    def update_sf(self):
        # prepare eqdsk dict for nova.streamfunction
        eqdsk = {
            'name': 'DINA_'+self.name,
            # Number of horizontal and vertical points
            'nx': self.nx, 'nz': self.nz,  # Location of the grid-points
            'x': self.x, 'z': self.z,  # size of the domain in meters
            'xdim': self.scalars[self.time_index]['dx'],
            'zdim': self.scalars[self.time_index]['dz'],
            # Reference vacuum toroidal field (m, T)
            'xcentr': np.array([]), 'bcentr': np.array([]),
            'xgrid1': self.x[0],  # x of left side of domain
            'zmid': (self.z[-1] + self.z[0]) / 2,  # z mid-domain
            # x-z location of magnetic axis
            'xmagx': self.scalars[self.time_index]['xo'],
            'zmagx': self.scalars[self.time_index]['zo'],
            # Poloidal flux at the axis (Weber / rad)
            'simagx': self.scalars[self.time_index]['pmag'],
            # Poloidal flux at plasma boundary (Weber / rad)
            'sibdry': self.scalars[self.time_index]['pbound'],
            'cpasma': np.array([]),
            'psi': self.psi,    # Poloidal flux in Weber/rad on grid points
            'fpol': np.zeros(self.nflux),  # Poloidal current function
            # 'FF'(psi) in (mT)^2/(Weber/rad) on uniform flux grid'
            'ffprim': np.zeros(self.nflux),
            # "P'(psi) in (nt/m2)/(Weber/rad) on uniform flux grid"
            'pprime': np.zeros(self.nflux),
            # Plasma pressure in nt/m^2 on uniform flux grid
            'pressure': np.zeros(self.nflux),
            'qpsi': np.array([]),  # q values on uniform flux grid
            'pnorm': np.array([]),  # uniform flux grid
            # Plasma boundary
            'nbdry': self.nsep,
            'xbdry': self.sep[:, 0], 'zbdry': self.sep[:, 1],
            'nlim': self.nfw, 'xlim': self.fw[:, 0], 'zlim': self.fw[:, 1],
            'ncoil': 0, 'xc': np.array([]), 'zc': np.array([]),  # coils
            'dxc': np.array([]), 'dzc': np.array([]), 'Ic': np.array([])}
        if not hasattr(self, 'sf'):  # create
            self.sf = SF('DINA_'+self.name, eqdsk=eqdsk)
        else:
            self.sf.update_plasma(eqdsk)  # update
        return

    def get_force_array(self):
        # magnetic field vector at points fn(t)
        B = np.zeros((self.nt, 3, self.vs.nP))
        F = np.zeros((self.nt, 3, self.vs.nP))
        Fo = np.zeros((self.nt, 3, self.vs.nP))
        tick = clock(self.nt)
        with readtxt(self.filename) as self.rt:
            for i in range(self.nt):
                self.read_array()  # read psi matp
                self.update_sf()  # update eqdsk
                B[i], F[i], Fo[i] = self.get_force()
                tick.tock()
        return B, F, Fo

    def get_field(self, points):
        nP = len(points)
        B = np.zeros((3, nP))
        for iP, point in enumerate(points):
            B[::2, iP] = self.sf.Bpoint(point)
        return B

    def get_force(self, plot=False, scale=1e-2):
        B = self.get_field(self.vs.points)
        Ivs = np.zeros((3, self.vs.nP))
        Ivs[1, :] = 4*self.pl.Ivs_o[self.time_index]  # Amp-turns
        F = np.zeros((3, self.vs.nP))
        for i, coil in enumerate(self.vs.geom):
            vs_sign = self.vs.geom[coil]['sign']
            F[:, i] = np.cross(vs_sign * Ivs[:, i], B[:, i], axis=0)
        if plot:
            for i, point in enumerate(self.vs.points):
                plt.arrow(point[0], point[1],
                          scale * F[0, i], scale * F[2, i],
                          width=0.05, head_length=0.15, color='C3')
        return B, F, Fo

    def plot(self, time_index=0, levels=None):
        if time_index != 0:
            self.read_single_array(time_index)
            self.update_sf()

        self.sf.contour(boundary=False)
        self.vs.plot()
        self.get_force(True)
        plt.plot(self.fw[:, 0], self.fw[:, 1])
        plt.plot(self.sep[:, 0], self.sep[:, 1], linewidth=2.5, color='gray')
        plt.axis('off')
        plt.axis('equal')

    def movie(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 10))
        moviename = '../Movies/{}'.format(self.name+'_psi')
        moviename += '.mp4'
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=20, bitrate=-1, codec='libx264',
                              extra_args=['-pix_fmt', 'yuv420p'])
        tick = clock(self.nt)
        plt.axis('off')
        plt.axis('equal')
        levels = self.sf.contour(boundary=False)  # fix contour levels
        xlim = np.array([self.x[0], self.x[-1]])
        zlim = np.array([self.z[0], self.z[-1]])
        with writer.saving(fig, moviename, 72):
            with readtxt(self.filename) as self.rt:
                for i in range(self.nt):
                    plt.clf()
                    self.read_array()
                    self.update_sf()
                    self.sf.contour(levels=levels, boundary=False)
                    self.vs.plot()
                    self.get_force(True)
                    plt.plot(self.fw[:, 0], self.fw[:, 1])
                    plt.plot(self.sep[:, 0], self.sep[:, 1],
                             linewidth=2.5, color='gray')
                    plt.plot(xlim[0], zlim[0], '.', alpha=0)
                    plt.plot(xlim[1], zlim[1], '.', alpha=0)
                    writer.grab_frame()
                    tick.tock()


if __name__ == '__main__':

    psi = read_psi('disruptions')
    psi.read_file(3)

    # psi.movie()
    # plt.figure(figsize=(7, 10))
    psi.plot(30)


