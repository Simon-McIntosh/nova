# -*- coding: utf-8 -*-
import pygmsh

# from helpers import compute_volume


def test():
    geom = pygmsh.built_in.Geometry()

    poly = geom.add_polygon(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        lcar=0.1,
    )

    geom.add_boundary_layer(
        edges_list=[poly.line_loop.lines[0]],
        hfar=0.1,
        hwall_n=0.01,
        ratio=1.1,
        thickness=0.2,
        anisomax=100.0,
    )

    geom.add_boundary_layer(
        nodes_list=[poly.line_loop.lines[1].points[1]],
        hfar=0.1,
        hwall_n=0.01,
        ratio=1.1,
        thickness=0.2,
        anisomax=100.0,
    )

    # geom.add_background_field([field0, field1])

    # geom.add_raw_code(f'Recombine Surface {{{poly.surface.id}}};')
    geom.add_raw_code("Mesh.Algorithm=6;")

    points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    # assert abs(compute_volume(points, cells) - ref) < 1.0e-2 * ref
    return points, cells


if __name__ == "__main__":
    points, cells = test()
    triangulation = tri.Triangulation(points[:, 0], points[:, 1], cells["triangle"])

    plt.triplot(triangulation)

    def quatplot(x, y, quatrangles, ax=None, **kwargs):
        if not ax:
            ax = plt.gca()
        xy = np.c_[x, y]
        verts = xy[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()

    quatplot(points[:, 0], points[:, 1], cells)
