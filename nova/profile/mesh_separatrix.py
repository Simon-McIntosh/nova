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
from amigo import geom


eqdsk = read_eqdsk(file='burn').eqdsk
sf = SF(eqdsk=eqdsk)

xbdry, zbdry = sf.get_boundary(alpha=1, locate='zmin')
xbdry, zbdry = geom.xzInterp(xbdry, zbdry, npoints=120)

boundary = np.array([xbdry[:-1], zbdry[:-1], np.zeros(len(xbdry[:-1]))]).T
#boundary = rdp(boundary, 0.01)

geo = pygmsh.built_in.Geometry()
poly = geo.add_polygon(boundary, make_surface=True)
geo.add_raw_code(f'Recombine Surface {{{poly.surface.id}}};')

bl = geo.add_boundary_layer(thickness=0.5)


geo.add_raw_code('Mesh.Algorithm=8;')
geo.add_raw_code('Mesh.Smoothing = 5;')




#geo.set_transfinite_surface

points, cells = pygmsh.generate_mesh(geo, dim=2, mesh_file_type='vtk',
                                     verbose=False)[:2]

triangulation = tri.Triangulation(points[:, 0], points[:, 1],
                                  cells['triangle'])


meshio.write_points_cells("plasma.msh", points, cells,
                          file_format="gmsh2-ascii")

mesh = fp.Gmsh2D("plasma.msh")  # read mesh from file
triangulation_face = tri.Triangulation(*mesh.faceCenters)

psi = fp.CellVariable(name=r'$\psi$', mesh=mesh, value=0.0)
psi_norm = fp.CellVariable(name=r'$\psi_{norm}$', mesh=mesh, value=0.0)
psi_norm_f = fp.FaceVariable(name='$\psi_{norm}$', mesh=mesh)
psi.constrain(1.0, where=mesh.exteriorFaces)

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
        #print(residual)

solve()
viewer = fp.Viewer(vars=psi_norm, title='',
                   colorbar=None, cmap=plt.cm.inferno, datamin=0, datamax=1)
ax = viewer.axes
ax.axis('equal')
viewer.cmap = plt.cm.inferno
#viewer.plot()
ax.axis('off')

#ax = plt.subplots(1, 1)[1]
#ax.axis('equal')
#tri.CubicTriInterpolator(triangulation, psi_norm.value)
psi_norm_f.setValue(psi_norm.faceValue)
psi_norm_f.setValue(1, where=mesh.exteriorFaces)

'''
x2d, z2d, = geom.grid(5e4, [mesh.x.value.min(), mesh.x.value.max(),
                            mesh.y.value.min(), mesh.y.value.max()])[:2]
psi_interp = tri.CubicTriInterpolator(triangulation_face, psi_norm_f,
                                      kind='min_E')
ax.contour(x2d, z2d, psi_interp(x2d, z2d), levels=np.linspace(0, 0.95, 21),
           linewidths=0.75, colors='gray')
'''
tri.TriMesh(triangulation)
ax.tricontour(triangulation_face, psi_norm_f, colors='gray',
              levels=np.linspace(0, 1, 21), linewidths=0.75, alpha=0.5)


