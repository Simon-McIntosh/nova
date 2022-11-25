import colorsys
import functools
import operator

from descartes import PolygonPatch
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.colors as mc
import numpy as np
import pandas as pd
import shapely.geometry

from nova.utilities.IO import human_format
from nova.plot import plt


class CoilPlot:


    def plot(self, subcoil=False, plasma=True, plasma_boundary=True,
             label='active', current='A',
             field=True, zeroturn=False, feedback=False, ax=None):
        if ax is None:
            plt.set_aspect(0.9)
            ax = plt.gca()
        if subcoil:
            self.plot_coil(self.subcoil, zeroturn=zeroturn,
                           feedback=feedback, ax=ax)
        else:
            self.plot_coil(self.coil, zeroturn=zeroturn,
                           feedback=feedback, ax=ax)
        ax.axis('equal')
        ax.axis('off')
        plt.tight_layout()
        if plasma and self.coil.nP > 0:
            self.label_plasma(ax)
        if plasma_boundary and hasattr(self, 'plasma_boundary'):
            plt.plot(*self.plasma_boundary.boundary.xy, 'C0')
        if label or current or field:
            self.label_coil(ax, label, current, field)



    def label_plasma(self, ax, fs=None):
        if fs is None:
            fs = matplotlib.rcParams['legend.fontsize']
        x = self.coil.x[self.coil.plasma]
        z = self.coil.z[self.coil.plasma] + self.coil.dz[self.coil.plasma]/10
        ax.text(x, z, f' {1e-6*self.Ip:1.1f}MA', fontsize='medium',
                ha='center', va='center', color=0.95 * np.ones(3),
                zorder=10)

    def label_gaps(self, ax=None):
        coil_index = []
        for end in ['L', 'U']:
            position = range(1, 4) if end == 'U' else range(3, 0, -1)
            for i in position:
                coil_index.append(f'CS{i}{end}')
        gap_index = ['LDP'] + coil_index + ['LDP']
        if ax is None:
            ax = plt.gca()
        for i, name in enumerate(coil_index):
            x, z, dx, dz = self.coil.loc[name, ['x', 'z', 'dx', 'dz']]
            drs = 2/3*dx
            ax.text(x + drs, z, f'Coil {i}',
                    ha='left', va='center', color=0.2 * np.ones(3))
        xo, zo = self.coil.loc[coil_index[0], ['x', 'z']]
        z1 = self.coil.loc[coil_index[-1], 'z']
        dzo = (z1-zo) / (len(coil_index) - 1)
        z = zo - dzo/2
        for i in range(7):
            ax.text(x - drs, z, f'{gap_index[i]}-{gap_index[i+1]}',
                    ha='right', va='center', color='C3')
            ax.text(x + drs, z, f'Gap {i}',
                    ha='left', va='center', color='C3')
            z += dzo
