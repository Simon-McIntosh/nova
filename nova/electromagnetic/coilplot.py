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
from nova.utilities.pyplot import plt


class CoilPlot:

    @staticmethod
    def patch_coil(coil, overwrite=False, patchwork_factor=0.15, **kwargs):
        # call on-demand
        part_color = {'VS3': 'C0', 'VS3j': 'gray', 'CS': 'C0', 'PF': 'C0',
                      'trs': 'C3', 'dir': 'C3',
                      'vv': 'C3', 'vvin': 'C3', 'vvout': 'C3',
                      'bb': 'C7',
                      'plasma': 'C4', 'Plasma': 'C4',
                      'cryo': 'C5'}
        color = kwargs.get('part_color', part_color)
        zorder = kwargs.get('zorder', {'VS3': 1, 'VS3j': 0, 'CS': 3, 'PF': 2})
        alpha = {'plasma': 0.75}
        if 'coil' not in coil:
            patchwork_factor = 0
        patch = [[] for __ in range(coil.nC)]
        for i, (x, z, dx, dz, cross_section,
                current_patch, polygon, part) in enumerate(
                coil.loc[:, ['x', 'z', 'dx', 'dz', 'cross_section', 'patch',
                             'polygon', 'part']].values):
            if overwrite or np.array(pd.isnull(current_patch)).any():
                if isinstance(polygon, dict):
                    polygon = shapely.geometry.shape(polygon)
                if isinstance(polygon, shapely.geometry.Polygon):
                    patch[i] = [PolygonPatch(polygon)]
                else:
                    patch[i] = []
            else:
                patch[i] = [current_patch]
            for j in range(len(patch[i])):
                patch[i][j].set_edgecolor('darkgrey')
                patch[i][j].set_linewidth(0.25)
                patch[i][j].set_antialiased(True)
                patch[i][j].set_facecolor(color.get(part, 'C9'))
                patch[i][j].set_zorder = zorder.get(part, 0)
                patch[i][j].set_alpha(alpha.get(part, 1))
                if patchwork_factor != 0:
                    CoilPlot.patchwork(patch[i][j], patchwork_factor)
        coil.loc[:, 'patch'] = np.asarray(patch, object)

    @staticmethod
    def patchwork(patch, factor):
        'alternate facecolor lightness by +- factor'
        factor *= 1 - 2 * np.random.rand(1)[0]
        c = patch.get_facecolor()
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        c = colorsys.hls_to_rgb(
                c[0], max(0, min(1, (1 + factor) * c[1])), c[2])
        patch.set_facecolor(c)


    def plot_coil(self, coil, alpha=1, ax=None, zeroturn=False,
                  feedback=False, **kwargs):
        if ax is None:
            ax = plt.gca()
        if not coil.empty:
            if pd.isnull(coil.loc[:, 'patch']).any() or len(kwargs) > 0:
                CoilPlot.patch_coil(coil, **kwargs)  # patch on-demand
            index = np.full(coil.nC, True)
            if not zeroturn:  # exclude zeroturn filaments (Nt == 0)
                index &= (coil.Nt != 0)
            if not feedback:  # exclude stabilization coils
                index &= ~coil.feedback
            patch = coil.loc[index, 'patch']
            # form list of lists
            patch = [p if pd.api.types.is_list_like(p)
                     else [p] for p in patch]
            if len(patch) > 0:
                # flatten
                patch = functools.reduce(operator.concat, patch)
                # sort
                patch = np.array(patch)[np.argsort([p.zorder for p in patch])]
                pc = PatchCollection(patch, match_original=True)
                ax.add_collection(pc)

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

    def label_coil(self, ax, label='coil', current='A', field=True,
                   coil=None, fs='medium', Nmax=20):
        if coil is None:
            coil = self.coil
        if label == 'all' or label == 'full':  # all coils
            parts = coil.part
        elif label == 'status':  # based on coil.update_status
            parts = coil.part[coil._current_index[coil._mpc_referance]]
            parts = parts
        elif label == 'active':  # active == True
            parts = coil.part[coil.active & ~coil.plasma & ~coil.feedback]
        elif label == 'passive':  # active == False
            parts = coil.part[~coil.active & ~coil.plasma & ~coil.feedback]
        elif label == 'coil':  # plasma == False
            parts = coil.part[~coil.plasma & ~coil.feedback]
        elif label == 'plasma':  # plasma == True
            parts = coil.part[coil.plasma & ~coil.feedback]
        elif label == 'free':  # optimize == True
            parts = coil.part[coil.optimize & ~coil.plasma & ~coil.feedback]
        elif label == 'fix':  # optimize == False
            parts = coil.part[~coil.optimize & ~coil.plasma & ~coil.feedback]
        else:
            if not pd.api.types.is_list_like(label):
                label = [label]
            parts = self.coil.part
            parts = [_part for _part in label if _part in parts]
        parts = parts.unique()
        parts = list(parts)
        N = {p: sum(coil.part == p) for p in parts}
        # check for presence of field instance
        field = False if 'field' not in self.biot_instances else field
        # referance vertical length scale
        dz_ref = np.diff(ax.get_ylim())[0] / 100
        nz = np.sum(np.array([parts is not False, current is not None,
                              field is not False]))
        if nz == 1:
            dz_ref = 0
        ztext = {name: 0 for name, value
                 in zip(['label', 'current', 'field'],
                        [label, current, field]) if value}
        for name, dz in zip(ztext, nz*dz_ref * np.linspace(1, -1, nz)):
            ztext[name] = dz
        for name, part in zip(coil.index, coil.part):
            if part in parts and N[part] < Nmax:
                x, z = coil.loc[name, 'x'], coil.loc[name, 'z']
                dx = coil.loc[name, 'dx']
                drs = 2/3*dx * np.array([-1, 1])
                if coil.part[name] == 'CS':
                    drs_index = 0
                    ha = 'right'
                else:
                    drs_index = 1
                    ha = 'left'
                # label coil
                ax.text(x + drs[drs_index], z + ztext['label'],
                        name, fontsize=fs, ha=ha, va='center',
                        color=0.2 * np.ones(3))
                if current:
                    if current == 'Ic' or current == 'A':  # line current
                        unit = 'A'
                        Ilabel = coil.loc[name, 'Ic']
                    elif current == 'It' or current == 'AT':  # turn current
                        unit = 'At'
                        Ilabel = coil.loc[name, 'It']
                    else:
                        raise IndexError(f'current {current} not in '
                                         '[Ic, A, It, AT]')
                    txt = f'{human_format(Ilabel, precision=1)}{unit}'
                    ax.text(x + drs[drs_index], z + ztext['current'], txt,
                            fontsize=fs, ha=ha, va='center',
                            color=0.2 * np.ones(3))
                if field:
                    self.update_field()
                    Blabel = coil.loc[name, 'B']
                    txt = f'{human_format(Blabel, precision=4)}T'
                    ax.text(x + drs[drs_index], z + ztext['field'], txt,
                            fontsize=fs, ha=ha, va='center',
                            color=0.2 * np.ones(3))

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
