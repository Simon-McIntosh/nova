import numpy as np
import scipy.spatial.transform
from nova.frame.coilset import CoilSet


def arc(radius, theta, angles=(0, 0, 0), number=500):
    """Return arc in the x-y plane."""
    theta = np.linspace(0, theta, number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=-1
    )
    rotation = scipy.spatial.transform.Rotation.from_euler("xyz", angles)
    return rotation.apply(points)


def coil(points, minimum_arc_nodes=3, **kwargs):
    """Return grid contour levels."""
    coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay", "Az"])
    coilset.winding.insert(
        points,
        {"c": (0, 0, 0.25)},
        minimum_arc_nodes=minimum_arc_nodes,
        Ic=1e3,
    )
    coilset.grid.solve(1e3, limit=[0.1, 4, -2, 2])
    if minimum_arc_nodes > 3:
        coilset.plot()

    # vtk(coilset)

    coilset.grid.axes.streamplot(
        coilset.grid.data.x.data,
        coilset.grid.data.z.data,
        coilset.grid.bx_.T,
        coilset.grid.bz_.T,
    )

    return coilset
    # return coilset.grid.plot("bx", **kwargs)


def vtk(coilset):
    import pyvista as pv
    import vedo

    points = np.stack(
        [
            coilset.grid.data.x2d,
            np.zeros_like(coilset.grid.data.x2d),
            coilset.grid.data.z2d,
        ],
        axis=-1,
    ).reshape(-1, 3)

    mesh = pv.PolyData(points).delaunay_2d()
    contours = mesh.contour(isosurfaces=71, scalars=coilset.grid.bx.reshape(-1))

    coilset.frame.vtkplot()
    vedo.Mesh(contours, c="black").show(new=False)


rng = np.random.default_rng(19)
angles = np.pi * (2 * rng.random(3) - 1)
angles = (0, 0, 0)

theta = rng.random(1)[0] * 2 * np.pi
theta = 2 * np.pi
# assert np.allclose(angles, [1.91638989, 1.93484905, 0.09629334])
# angles -= 1e-1
# angles = np.array([1.91638989, 1.93484905, 0.09629334])
points = arc(5.3, theta, angles)
coilset = coil(points, 501, colors="C0")

grid = coilset.grid

_ = grid.ay
psi = grid.data.x2d.data * grid.ay_
coilset.grid.axes.contour(grid.data.x, grid.data.z, psi.T, levels=51)


rbf = scipy.interpolate.RectBivariateSpline(grid.data.x, grid.data.z, psi)
u_xx = rbf.ev(grid.data.x2d, grid.data.z2d, dx=2)
u_zz = rbf.ev(grid.data.x2d, grid.data.z2d, dy=2)

bx = scipy.interpolate.RectBivariateSpline(grid.data.x, grid.data.z, grid.bx_)
bx_z = bx.ev(grid.data.x2d, grid.data.z2d, dy=1)
bz = scipy.interpolate.RectBivariateSpline(grid.data.x, grid.data.z, grid.bz_)
bz_x = bx.ev(grid.data.x2d, grid.data.z2d, dx=1)

grid.axes.contour(grid.data.x, grid.data.z, (u_xx + u_xx).T, levels=51)
grid.axes.contour(grid.data.x, grid.data.z, (bx_z - bz_x).T, levels=51)

# coilset = coil(points, 3, colors="C1", linestyles="--")


"""
def plot(minimum_arc_nodes=3, attr="bx", number=500, **kwargs):
    theta, dtheta = np.linspace(0, 2 * np.pi, number, retstep=True)
    radius = 5.3
    Ic = 1e6
    # points = np.stack(
    #    [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=-1
    # )

    points = np.stack(
        [radius * np.cos(theta), 1.5 * np.ones_like(theta), radius * np.sin(theta)],
        axis=-1,
    )

    field_attrs = ["Bx", "By", "Bz", "Ax", "Ay", "Az"]
    if minimum_arc_nodes == 0:
        field_attrs.append("Psi")
    coilset = CoilSet(field_attrs=field_attrs)
    if minimum_arc_nodes == 0:
        coilset.coil.insert(radius, 0, 0.25, 0.25, section="c", Ic=Ic)
    else:
        coilset.winding.insert(
            points, {"c": (0, 0, 0.25)}, Ic=Ic, minimum_arc_nodes=minimum_arc_nodes
        )

    coilset.grid.solve(1e3, 0.5)
    coilset.plot()
    return coilset.grid.plot(attr, **kwargs)


# levels = plot(3, colors="C1")
# print(levels := plot(0, colors="C0"))
print(plot(3, colors="C1", linestyles="-"))
print(plot(201, colors="C3", linestyles="--"))
# coilset.frame.vtkplot()
"""
