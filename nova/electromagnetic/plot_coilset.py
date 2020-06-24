from amigo.pyplot import plt
import numpy as np
import collections
from amigo import geom
import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib import patches
from scipy.interpolate import griddata


class plot_coilset:

    def __init__(self, coilset=None):
        self.initalize_keys()
        self.initalize_collection()
        self.link(coilset)

    def initalize_keys(self):
        self.color_key = \
            ['CS', 'skip', 'VS3', 'vv_DINA', 'plasma',
             'PF', 'skip', 'bb_DINA', 'trs', 'skip',
             'skip', 'skip', 'skip', 'vv', 'skip']
        # set defaults
        self.properties = {'var': None, 'cmap': plt.cm.RdBu, 'clim': None}

    def initalize_collection(self, *args):
        self.patch = {'patch': []}

    def link(self, coilset):
        if coilset:
            self.coilset = coilset

    def update_properties(self, **kwargs):
        for key in kwargs:
            if key in self.properties:
                self.properties[key] = kwargs.get(key)

    @staticmethod
    def get_coil_label(name):
        sname = name.split('_')
        if len(sname) == 1:
            label = name
        elif len(sname) == 2:
            label = sname[0]
        else:
            label = '_'.join(sname[:-1])
        return label

    def label_coil(self, coil, label, current, fs=12):
        for name in coil.index:
            x, z = coil.loc[name, 'x'], coil.loc[name, 'z']
            dx, dz = coil.loc[name]['dx'], coil.loc[name, 'dz']
            if 'index' in self.coilset:
                if name in self.coilset['index']['CS']['name']:
                    drs = -2.5 / 3 * dx
                    ha = 'right'
                else:
                    drs = 2.5 / 3 * dx
                    ha = 'left'
            else:
                drs = 2.5 / 3 * dx
                ha = 'left'

            if label and current:
                zshift = max([dz / 10, 0.5])
            else:
                zshift = 0
            print_label = ('CS' in name or 'PF' in name)
            if label and print_label:
                plt.text(x + drs, z + zshift, name, fontsize=fs,
                         ha=ha, va='center', color=0.2 * np.ones(3))
            if current and print_label:
                if current == 'A' and 'Nt' in coil[name]:  # amps
                    Nt = coil[name]['Nt']
                    linecurrent = coil[name]['It'] / Nt
                    txt = '{:1.1f}kA'.format(linecurrent * 1e-3)
                else:  # amp turns
                    if abs(coil.loc[name, 'It']) < 0.1e6:  # display as kA.t
                        txt = '{:1.1f}kAT'.format(coil.loc[name, 'It'] * 1e-3)
                    else:  # MA.t
                        txt = '{:1.1f}MAT'.format(coil.loc[name, 'It'] * 1e-6)
                plt.text(x + drs, z - zshift, txt,
                         fontsize=fs, ha=ha, va='center',
                         color=0.2 * np.ones(3))

    def get_coil(self, subcoil):
        if subcoil:
            coil = self.coilset['subcoil']
        else:
            coil = self.coilset['coil']
        return coil

    def plot(self, subcoil=True, label=False, plasma=False,
             current=False, alpha=1, ax=None, patch=True):
        if ax is None:
            ax = plt.gca()
        coil = self.get_coil(subcoil)
        self.plot_coil(coil, label=label, current=current, alpha=alpha,
                       ax=ax, patch=patch)
        if plasma:
            self.plot_coil(self.coilset['plasma'], alpha=alpha,
                           coil_color='C4', ax=ax)
        ax.axis('equal')
        ax.axis('off')

    def get_single(self, name):
        coil = self.get_coils(True)
        single_coil = collections.OrderedDict(
                [(n, coil[n]) for n in self.coilset['index'][name]['name']])
        return single_coil

    def plot_single(self, name, subcoil=False, label=False, plasma=False,
                    current=False, alpha=1, ax=None, plot=True, **kwargs):
        coil = self.get_single(name)
        self.plot_coil(coil, label=label, current=current,
                       alpha=alpha, ax=ax, plot=plot, **kwargs)

    def contour_single(self, name, variable, n=1e3, ax=None):
        if ax is None:
            ax = plt.gca()
        coil = self.get_single(name)
        x = [coil[name]['x'] for name in coil]
        z = [coil[name]['z'] for name in coil]
        values = [coil[name][variable] for name in coil]
        x2d, z2d, xg, zg = geom.grid(n, [np.min(x), np.max(x),
                                     np.min(z), np.max(z)])[:4]
        v2d = griddata((x, z), values, (x2d, z2d), method='linear')
        levels = np.linspace(np.min(values), np.max(values), 11)
        CS = ax.contour(x2d, z2d, v2d, levels=levels, colors='gray',
                        linewidths=1.75)
        ax.clabel(CS, levels[1::2], inline=1, fmt='%1.1f', fontsize=14)

    def plot_coil(self, coil, label=False, current=False,
                  alpha=1, plot=True, ax=None, patch=True, **kwargs):
        self.update_properties(**kwargs)
        coil_color = kwargs.get('coil_color', None)
        kwargs.pop('coil_color', None)
        for name in coil.index:
            if coil.loc[name, 'cross_section'] == 'square':
                shape = 'rectangle'
            else:
                shape = 'circle'
            if coil_color is None:
                color = self.get_coil_color(name)
            else:
                color = coil_color
            if 'VS3' in name or 'VS_' in name:
                zorder = 10
            else:
                zorder = None
            if 'plasma' in name.lower() or 'PL' in name:
                alpha_ = 0.5
            else:
                alpha_ = alpha
            self.patch_coil(coil.loc[name], alpha=alpha_,
                            coil_color=color, shape=shape, zorder=zorder,
                            **kwargs)
        if plot:
            self.plot_patch(ax=ax, **kwargs)
            if label or current:
                fs = matplotlib.rcParams['legend.fontsize']
                self.label_coil(coil, label, current, fs=fs)

    def get_coil_color(self, name, coil_color=None):
        label = self.get_coil_label(name)
        if coil_color is None:
            coil_color = 'C3'  # default
            for group in self.coilset['index']:
                if label in self.coilset['index'][group]['name']\
                        and group in self.color_key:
                    ic = self.color_key.index(group)
                    icolor = ic % 5
                    coil_color = 'C{}'.format(icolor)
                    break
        return coil_color

    def patch_coil(self, coil, alpha=1, shape='rectangle', **kwargs):
        edge_color = kwargs.get('edge_color', 'k')
        coil_color = kwargs.get('coil_color', 'C0')
        zorder = kwargs.get('zorder', None)
        x, z, dx, dz = coil['x'], coil['z'], coil['dx'], coil['dz']
        if 'It' in coil:
            Ic = coil['It']
        else:
            Ic = coil['If']
        # coil['Jc'] = 1e-6 * Ic / (dx*dz)
        if shape == 'circle':
            patch = patches.Circle((x, z), dx/2, facecolor=coil_color,
                                   alpha=alpha, edgecolor=edge_color,
                                   zorder=zorder)
        elif shape == 'rectangle':
            patch = patches.Rectangle((x-dx/2, z-dz/2), dx, dz,
                                      facecolor=coil_color, alpha=alpha,
                                      edgecolor=edge_color,
                                      zorder=zorder)
        else:
            raise TypeError('shape not in [''circle'', ''rectangle'']')
        self.append_patch(patch, **coil)

    def append_patch(self, patch, **kwargs):
        self.patch['patch'].append(patch)
        for var in kwargs:
            try:
                self.patch[var].append(kwargs[var])
            except KeyError:
                self.patch[var] = [kwargs[var]]

    def sort_patch(self):
        zorder = [patch.get_zorder() for patch in self.patch['patch']]
        index = np.argsort(zorder)
        for name in self.patch:
            self.patch[name] = [x for __, x in
                                sorted(zip(index, self.patch[name]))]

    def plot_patch(self, reset=False, ax=None, **kwargs):
        self.update_properties(**kwargs)
        var = self.properties['var']  # plot variable
        match_original = True if var is None else False
        clim = self.properties['clim']  # plot variable
        if ax is None:
            ax = plt.gca()
        if var is None:
            for patch in self.patch['patch']:
                ax.add_patch(patch)
        else:
            # self.sort_patch()  # zorder not used within PatchCollection
            pc = PatchCollection(self.patch['patch'],
                                 match_original=match_original,
                                 cmap=self.properties['cmap'],
                                 zorder=range(len(self.patch['patch'])))
            im = ax.add_collection(pc)
            if var in self.patch:
                var_array = np.array(self.patch[var])
            else:  # var is np.array len == len(patch)
                npatch = len(self.patch['patch'])
                if isinstance(var, np.ndarray) and len(var) == npatch:
                    var_array = var
                else:
                    raise ValueError('color array incompatable with patches')
            if clim is not None:
                if clim == 'symetric':
                    var_max = np.max(abs(var_array))
                    clim = [-var_max, var_max]
                elif clim == 'tight':
                    clim = [np.min(var_array), np.max(var_array)]
                pc.set_clim(vmin=clim[0], vmax=clim[1])
            pc.set_array(var_array)
            plt.sca(ax)
            cb = plt.colorbar(im, orientation='vertical', aspect=50,
                              shrink=0.8, fraction=0.1)
            if var == 'It':
                cb.set_label('$I_c$ kA')
            elif var == 'Jc':
                cb.set_label('$J_c$ MAm$^{-2}$')

        ax.axis('equal')
        ax.axis('off')
        if reset:  # reset patch collection
            self.initalize_collection()
