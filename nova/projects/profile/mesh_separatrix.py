import numpy as np
import pygmsh
from nep.DINA.read_eqdsk import read_eqdsk
from nova.streamfunction import SF
from amigo.pyplot import plt
from rdp import rdp
import meshio
import fipy as fp
from matplotlib import tri
from nova.cross_coil import mu_o
from scipy.interpolate import interp2d
from amigo import geom, mesh
from shapely.geometry.polygon import LinearRing, Polygon
from scipy.interpolate import splprep, splev


#eqdsk = read_eqdsk(file='burn').eqdsk
#sf = SF(eqdsk=eqdsk)
#xbdry, zbdry = sf.get_boundary(alpha=1, locate='zmin')


class loop:

    def __init__(self, x, z, interp='rdp', close=True, **kwargs):
        self.close = close
        self.set_interpolator(interp, **kwargs)
        self.load(x, z, **kwargs)

    def set_interpolator(self, name, **kwargs):
        if name == 'linear':
            self.interpolate = self.linear
            self.n = kwargs.get('n', 50)  # point number
        elif name == 'rdp':
            self.interpolate = self.rdp
            self.f = kwargs.get('f', 0.0005)  # epsilon = f * loop_length

    def load(self, x, z, **kwargs):
        x, z = self.orient(x, z)
        close = kwargs.get('close', self.close)
        if close:
            x, z = self.close_loop(x, z)
        self.ring = LinearRing(np.c_[x, z])  # mutable loop
        if 'shrink' in kwargs:
            self.shrink(kwargs['shrink'])
        else:
            self.interpolate()

    @staticmethod
    def smooth(x, z, s=3):
        tck, u = splprep(np.c_[x, z].T, u=None, s=s, per=1)
        u_ = np.linspace(u.min(), u.max(), len(x))
        x, z = splev(u_, tck, der=0)
        return x, z

    def shrink(self, fraction):
        bounds = self.ring.bounds
        dx = bounds[2] - bounds[0]
        dz = bounds[3] - bounds[1]
        delta = fraction * np.min([dx, dz]) / 2
        offset_ring = self.ring.parallel_offset(delta, side='left')
        x, z = self.smooth(*offset_ring.xy, s=1)
        x, z = self.orient(x, z)
        self.load(x, z)
        self.interpolate()

    @staticmethod
    def orient(x, z):
        # counter clockwise starting at zmin
        geom.order(x, z, anti=True)
        iloc = np.argmin(z)
        x = np.append(x[iloc:], x[:iloc+1])
        z = np.append(z[iloc:], z[:iloc+1])
        return x, z

    def close_loop(self, x, z, eps=1e-6):
        if np.sqrt((x[0]-x[-1])**2 + (z[0]-z[-1])**2) > eps:
            x_ = np.mean([x[0], x[-1]])
            z_ = np.mean([z[0], z[-1]])
            x, z = np.append(x, x_), np.append(z, z_)
        return x, z

    def linear(self, n=None):
        # linear interpolation with n points
        if n is not None:
            self.n = n  # update point number
        x, z = self.get_points().T
        x, z = geom.xzInterp(x, z, npoints=self.n)
        self.ring = LinearRing(np.c_[x, z])

    def rdp(self, f=None):
        # simplify loop using the Ramer-Douglas-Peucker algorithm
        if f is not None:
            self.f = f
        epsilon = self.f * self.ring.length
        x, z = self.get_points().T
        x, z = rdp(np.c_[x, z], epsilon).T
        self.ring = LinearRing(np.c_[x, z])

    def plot(self, *args, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        for segment in ['inner', 'outer']:
            x, z = self.get_points(segment).T
            ax.plot(x, z, *args, **kwargs)
        ax.axis('equal')

    def get_points(self, segment='loop', dim=2):
        x, z = self.ring.xy
        if segment != 'loop':
            imax = np.argmax(z)
            if segment == 'inner':
                x, z = x[imax:], z[imax:]
            elif segment == 'outer':
                x, z = x[:imax+1], z[:imax+1]
            else:
                errtxt = 'segment {segment} not in [loop, inner, outer]'
                raise IndexError(errtxt)
        points = np.c_[x, z]
        if dim == 3:
            points = np.append(points, np.zeros((len(x), 1)), axis=1)
        return points


class gridgen:
    # generate grids for fipy simulations

    def __init__(self, x, z, nr, nc, shrink=0.5):
        self.geom = pygmsh.built_in.Geometry()
        self.geom.add_raw_code('Mesh.Algorithm=8;')
        self.set_loops(x, z, shrink=shrink)
        self.set_points(lcar=0.1)
        self.set_edge(nr, nc)

    def set_loops(self, x, z, shrink=0.5, interp='rdp'):
        self.loops = {}
        self.loops['sep'] = loop(x, z, interp=interp)
        self.loops['core'] = loop(x, z, interp=interp, shrink=shrink)

    def set_points(self, lcar=None):
        self.points = {}
        for loop in self.loops:
            points = self.loops[loop].get_points(dim=3)
            self.points[loop] = \
                [self.geom.add_point(p, lcar=lcar) for p in points[:-1]]
            self.points[loop].append(self.points[loop][0])

    def set_edge(self, nr, nc):
        lines = []
        lines.append(self.geom.add_line(self.points['sep'][-1],
                                        self.points['core'][-1]))
        lines.append(self.geom.add_spline(self.points['core'][::-1]))
        lines.append(self.geom.add_line(self.points['core'][0],
                                        self.points['sep'][0]))
        lines.append(self.geom.add_spline(self.points['sep']))
        #ll = self.geom.add_line_loop(lines)
        #surface = self.geom.add_plane_surface(ll)

        ll = self.geom.add_line_loop([lines[-1]])
        surface = self.geom.add_plane_surface(ll)

        self.geom.add_physical_surface(surface, label='plasma')
        self.geom.add_physical_line(lines, label='separatrix')
        #self.geom.add_physical_line(lines[1], 'core')

        '''
        self.geom.set_transfinite_lines([lines[0], lines[2]], nr)
        self.geom.set_transfinite_lines([lines[1], lines[3]], nc)
        self.geom.set_transfinite_surface(surface)
        self.geom.add_raw_code(f'Recombine Surface {{{surface.id}}};')

        self.geom.add_physical_surface(surface, 'plasma')
        self.geom.add_physical_line(lines[-1], 'separatrix')
        self.geom.add_physical_line(lines[1], 'core')
        '''

    def generate(self, verbose=False):
        self.mesh = pygmsh.generate_mesh(self.geom, dim=3, verbose=verbose,
                                         mesh_file_type='msh')

    def plot(self):
        for loop in self.loops:
            self.loops[loop].plot('-', color='gray')
        mesh.plot(self.mesh[0], self.mesh[1])




gg = gridgen(xbdry, zbdry, nr=10, nc=60, shrink=0.8)

gg.generate(verbose=True)

ax = plt.subplots(1, 1, figsize=(6, 8))[1]
plt.axis('off')
gg.plot()


'''
class fixed_boundary:

    def __init__(self, npoints=80):
        self.npoints = npoints

    def add_boundary(self, x, y):
        self.x

    def core(self, )

'''


'''
#triangulation = tri.Triangulation(points[:, 0], points[:, 1],
#                                  cells['triangle'])

meshio.write_points_cells("plasma.msh", *gg.mesh,
                          file_format="gmsh2-ascii")

mesh = fp.Gmsh2D("plasma.msh")  # read mesh from file


triangulation_face = tri.Triangulation(*mesh.faceCenters)

psi = fp.CellVariable(name=r'$\psi$', mesh=mesh, value=0.0)
psi_norm = fp.CellVariable(name=r'$\psi_{norm}$', mesh=mesh, value=0.0)
psi_norm_f = fp.FaceVariable(name='$\psi_{norm}$', mesh=mesh)
#psi.constrain(0.0, where=mesh.physicalFaces['separatrix'])
psi.constrain(0.0, where=mesh.exteriorFaces)


source = fp.CellVariable(name='$source$', mesh=mesh, value=0.5)
r = fp.CellVariable(name='$rcell$', mesh=mesh)
r.setValue(mesh.cellCenters[0])
r2 = fp.CellVariable(name='$rcell$', mesh=mesh)
r2.setValue(r**2)
conv = fp.FaceVariable(name='$conv$', mesh=mesh, rank=1)
conv.setValue([mesh.faceCenters[0]**-1, np.zeros(mesh.numberOfFaces)])

eq = fp.DiffusionTerm()
eq -= fp.ConvectionTerm(coeff=conv)
eq -= fp.ImplicitSourceTerm(r**-2)
eq -= source


def set_psi_norm(psi):
    psi_minmax = [psi.value.min(), psi.value.max()]
    if psi_minmax[1] > psi_minmax[0]:
        xpsi, mpsi = psi_minmax
    else:
        xpsi, mpsi = psi_minmax[::-1]
    return (psi.value-mpsi) / (xpsi - mpsi)


def solve():
    residual = 1
    psi.setValue(0)
    source.setValue(0.5)
    while residual > 1e-5:
        residual = eq.sweep(var=psi)
        psi_norm.setValue(set_psi_norm(psi))
        dPpsi = sf.dPpsi(psi_norm)
        dFFpsi = sf.dFFpsi(psi_norm)
        source.setValue(mu_o*r2*dPpsi+0.5*dFFpsi)
        print(residual)

solve()
viewer = fp.Viewer(vars=psi_norm, title='',
                   colorbar='vertical', cmap=plt.cm.inferno, datamin=0, datamax=1)
ax = viewer.axes
ax.axis('equal')
viewer.cmap = plt.cm.inferno
#viewer.plot()
ax.axis('off')

#ax = plt.subplots(1, 1)[1]
#ax.axis('equal')
#tri.CubicTriInterpolator(triangulation, psi_norm.value)
psi_norm_f.setValue(psi_norm.faceValue)
#psi_norm_f.setValue(psi_norm.harmonicFaceValue)

psi_norm_f.setValue(1, where=mesh.exteriorFaces)



#ax.triplot(triangulation_face)
ax.tricontour(triangulation_face, psi_norm_f, colors='gray',
              levels=np.linspace(0, 1, 21), linewidths=0.75, alpha=0.5)





#quadplot(points[:, 0], points[:, 1], cells['quad'],
#         facecolors=[], edgecolors='gray')
'''