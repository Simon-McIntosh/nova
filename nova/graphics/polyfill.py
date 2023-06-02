"""Methods for visulizing 2d and 3d polygons."""
import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d


def polyfill(x, z, color=0.5 * np.ones(3), alpha=1):
    """Plot 2d polygon."""
    verts = np.array([x, z])
    verts = [np.swapaxes(verts, 0, 1)]
    coll = PolyCollection(verts, edgecolors="none", color=color, alpha=alpha)
    ax = plt.gca()
    ax.add_collection(coll)
    ax.autoscale_view()


def polyfill3D(x, y, z, ax=None, color=0.5 * np.ones(3), alpha=1, lw=2):
    """Plot 3d polygon."""
    if ax is None:
        ax = Axes3D(plt.figure())
    verts = np.array([x, y, z])
    verts = [np.swapaxes(verts, 0, 1)]
    coll = art3d.Poly3DCollection(verts, edgecolors="none", color=color, alpha=alpha)
    coll.set_edgecolor("k")
    coll.set_linewidth(lw)
    ax.add_collection3d(coll)
    ax.plot(x, y, z, ".", alpha=0)
    ax.autoscale_view()
    ax.autoscale()
