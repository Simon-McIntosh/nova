from amigo.pyplot import plt
import matplotlib
import numpy as np


def plot(points, cells, ax=None, **kwargs):
    # https://stackoverflow.com/questions/49640311/
    # matplotlib-unstructered-quadrilaterals-instead-of-triangles
    if ax is None:
        ax = plt.gca()
    xy = np.c_[points[:, 0], points[:, 1]]
    for element in ["triangle", "quad"]:
        if element in cells:
            index = cells[element]
            verts = xy[index]
            pc = matplotlib.collections.PolyCollection(
                verts, facecolors=None, edgecolors="gray"
            )
            ax.add_collection(pc)
