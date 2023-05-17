from amigo.pyplot import plt
import scipy as sp
import numpy as np
from scipy.special import binom as bn
from scipy.special import iv as besl
from amigo import geom
import collections
import seaborn as sns
import pandas as pd
import pickle
from nova.config import nova_path
from amigo.geom import String


def unwind(p, angle=2.5, plot=False):
    # re-samlpe based on minimum turning angle and min/max segment lengths
    string = String(np.array([p['x'], p['z']]).T,
                    angle=angle, verbose=False)
    if plot:
        string.plot()
    return {'x': string.points[:, 0], 'z': string.points[:, 1]}


def add_value(Xo, i, name, value, lb, ub, clip=True):
    if clip:
        if value < lb:
            value = lb
        if value > ub:
            value = ub
    Xo['name'][i] = name
    Xo['value'][i] = value
    Xo['lb'][i] = lb
    Xo['ub'][i] = ub


def normalize_variables(Xo):
    X = (Xo['value'] - Xo['lb']) / (Xo['ub'] - Xo['lb'])
    return X


def denormalize_variables(x, Xo):
    Xo['value'] = x * (Xo['ub'] - Xo['lb']) + Xo['lb']
    return Xo['value']


def plot_variables(Xo, eps=1e-2, fmt='1.2f', scale=1, postfix=''):
    xo = normalize_variables(Xo)
    Xo['norm'] = xo
    data = pd.DataFrame(Xo)
    data.reset_index(level=0, inplace=True)
    plt.figure(figsize=plt.figaspect(0.5))
    sns.set_color_codes("muted")
    sns.barplot(x='norm', y='name', data=data, color="b")
    sns.despine(bottom=True)
    plt.ylabel('')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    patch = ax.patches
    # values = [xo[var]['value'] for var in xo]
    # xnorms = [xo[var]['xnorm'] for var in xo]
    for p, value, norm, var in zip(patch, Xo['value'], Xo['norm'], Xo['name']):
        x = p.get_width()
        if norm < 0:
            x = 0
        y = p.get_y() + p.get_height() / 2
        size = 'small'
        if norm < eps or norm > 1 - eps:
            size = 'large'
        text = ' {:{fmt}}'.format(scale * value, fmt=fmt)
        text += postfix + ' '
        # if var not in oppvar:
        #     text += '*'
        if value < 0.5:
            ha = 'left'
            color = 0.25 * np.ones(3)
        else:
            ha = 'right'
            color = 0.75 * np.ones(3)
        ax.text(x, y, text, ha=ha, va='center',
                size=size, color=color)
    plt.plot(0.5 * np.ones(2), np.sort(ax.get_ylim()), '--',
             color=0.5 * np.ones(3), zorder=0, lw=1)
    plt.plot(np.ones(2), np.sort(ax.get_ylim()), '-', color=0.25 * np.ones(3),
             zorder=0, lw=1.5)
    xlim = ax.get_xlim()
    xmin, xmax = np.min([0, xlim[0]]), np.max([1, xlim[1]])
    plt.xlim([xmin, xmax])


def check_var(var, xo):
    if var == 'l':
        var = 'l0s'
    if var not in xo:
        var = next((v for v in xo if var in v))  # match sub-string
    return var


def set_oppvar(xo, oppvar):  # set optimization bounds and normalize
    nopp = len(oppvar)
    xnorm, bnorm = np.zeros(nopp), np.zeros((nopp, 2))
    for i, var in enumerate(oppvar):
        var = check_var(var, xo)
        xnorm[i] = (xo[var]['value'] - xo[var]['lb']) / (xo[var]['ub'] -
                                                         xo[var]['lb'])
        bnorm[i, :] = [0, 1]
    return xnorm, bnorm


def get_oppvar(xo, oppvar, xnorm):
    x = np.copy(xnorm)
    for i, var in enumerate(oppvar):
        var = check_var(var, xo)
        x[i] = x[i] * (xo[var]['ub'] - xo[var]['lb']) + xo[var]['lb']
    return x


def get_value(xo):
    x = np.zeros(len(xo))
    for i, name in enumerate(xo):
        x[i] = xo[name]['value']
    return x


def remove_oppvar(oppvar, var):
    if var in oppvar:
        oppvar.remove(var)


def plot_oppvar(xo, oppvar, eps=1e-2, fmt='1.2f', scale=1, postfix=''):
    xnorm, bnorm = set_oppvar(xo, oppvar)
    for var in xo:
        xo[var]['xnorm'] = (xo[var]['value'] - xo[var]['lb']) /\
                           (xo[var]['ub'] - xo[var]['lb'])
    data = pd.DataFrame(xo).T
    data.reset_index(level=0, inplace=True)
    plt.figure(figsize=plt.figaspect(1))
    sns.set_color_codes("muted")
    sns.barplot(x='xnorm', y='index', data=data, color="b")
    sns.despine(bottom=True)
    plt.ylabel('')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    patch = ax.patches
    values = [xo[var]['value'] for var in xo]
    xnorms = [xo[var]['xnorm'] for var in xo]
    for p, value, xnorm, var in zip(patch, values, xnorms, xo):
        x = p.get_width()
        if xnorm < 0:
            x = 0
        y = p.get_y() + p.get_height() / 2
        size = 'small'
        if xnorm < eps or xnorm > 1 - eps:
            size = 'large'
        text = ' {:{fmt}}'.format(scale * value, fmt=fmt)
        text += postfix + ' '
        if var not in oppvar:
            text += '*'
        if xnorm < 0.1:
            ha = 'left'
            color = 0.25 * np.ones(3)
        else:
            ha = 'right'
            color = 0.75 * np.ones(3)
        ax.text(x, y, text, ha=ha, va='center',
                size=size, color=color)
    plt.plot(0.5 * np.ones(2), np.sort(ax.get_ylim()), '--',
             color=0.5 * np.ones(3), zorder=0, lw=1)
    plt.plot(np.ones(2), np.sort(ax.get_ylim()), '-', color=0.25 * np.ones(3),
             zorder=0, lw=1.5)
    xlim = ax.get_xlim()
    xmin, xmax = np.min([0, xlim[0]]), np.max([1, xlim[1]])
    plt.xlim([xmin, xmax])


def get_input(oppvar=[], **kwargs):
    if 'x' in kwargs:
        inputs = {}
        X = kwargs.get('x')
        try:
            for var, x in zip(oppvar, X):
                inputs[var] = x
        except ValueError:
            errtxt = '\n'
            errtxt += 'Require self.variables'
            raise ValueError(errtxt)
    elif 'inputs' in kwargs:
        inputs = kwargs.get('inputs')
    else:
        inputs = {}
    return inputs


def close_loop(p, npoints):
    for var in ['x', 'z']:
        p[var] = np.append(p[var], p[var][0])
    p['x'], p['z'] = geom.xzSLine(p['x'], p['z'], npoints=npoints)
    return p


def set_limit(po, limits=True):
    if limits:
        if po['value'] < po['lb']:
            po['value'] = po['lb']
        if po['value'] > po['ub']:
            po['value'] = po['ub']
    return po


class Aloop(object):  # tripple arc loop
    def __init__(self, npoints=200, limits=True):
        self.npoints = npoints
        self.initalise_parameters()
        self.name = 'Aloop'
        self.limits = limits

    def initalise_parameters(self):
        self.xo = collections.OrderedDict()
        self.xo['xo'] = {'value': 4.486, 'lb': 3, 'ub': 5}  # x origin
        self.xo['zo'] = {'value': 0, 'lb': -1, 'ub': 1}  # z origin
        self.xo['sl'] = {'value': 6.428,
                         'lb': 0.5, 'ub': 10}  # straight length
        self.xo['f1'] = {'value': 2, 'lb': 0.1, 'ub': 10}  # rs == f1*z small
        self.xo['f2'] = {'value': 5, 'lb': 0.1, 'ub': 10}  # rm == f2*rs mid
        self.xo['a1'] = {'value': 45, 'lb': 5,
                         'ub': 65}  # small arc angle, deg
        self.xo['a2'] = {'value': 75, 'lb': 5,
                         'ub': 110}  # middle arc angle, deg
        self.oppvar = list(self.xo.keys())
        # for rmvar in ['a1','a2']:  # remove arc angles from oppvar
        #    self.oppvar.remove(rmvar)
        # self.oppvar.remove('a2')

    def set_input(self, **kwargs):
        inputs = get_input(self.oppvar, **kwargs)
        for key in inputs:
            if key in self.xo:
                try:  # dict
                    for k in inputs[key]:
                        self.xo[key][k] = inputs[key][k]
                except TypeError:  # single value
                    self.xo[key]['value'] = inputs[key]
                self.xo[key] = set_limit(self.xo[key], limits=self.limits)

    def get_xo(self):
        values = []
        for n in ['xo', 'zo', 'sl', 'f1', 'f2', 'a1', 'a2']:
            values.append(self.xo[n]['value'])
        return values

    def draw(self, **kwargs):
        self.npoints = kwargs.get('npoints', self.npoints)
        self.set_input(**kwargs)
        self.segments = {'x': [], 'z': []}
        xo, zo, sl, f1, f2, a1, a2 = self.get_xo()
        a1 *= np.pi / 180  # convert to radians
        a2 *= np.pi / 180
        asum = a1 + a2
        # straight section
        x = xo * np.ones(2)
        z = np.array([zo, zo + sl])
        self.cs = {'x': x[-1], 'z': z[-1]}  # top of nose (cs seat)
        self.segments['x'].append(x)
        self.segments['z'].append(z)
        # small arc
        theta = np.linspace(0, a1, round(0.5 * self.npoints * a1 / np.pi))
        rx, zx = x[-1], z[-1]
        x = np.append(x, x[-1] + f1 * (1 - np.cos(theta)))
        z = np.append(z, z[-1] + f1 * np.sin(theta))
        self.segments['x'].append(rx + f1 * (1 - np.cos(theta)))
        self.segments['z'].append(zx + f1 * np.sin(theta))
        # mid arc
        theta = np.linspace(
            theta[-1], asum, round(0.5 * self.npoints * a2 / np.pi))
        rx, zx = x[-1], z[-1]
        x = np.append(x, x[-1] + f2 * (np.cos(a1) - np.cos(theta)))
        z = np.append(z, z[-1] + f2 * (np.sin(theta) - np.sin(a1)))
        self.segments['x'].append(rx + f2 * (np.cos(a1) - np.cos(theta)))
        self.segments['z'].append(zx + f2 * (np.sin(theta) - np.sin(a1)))
        # large arc
        rl = (z[-1] - zo) / np.sin(np.pi - asum)
        theta = np.linspace(theta[-1], np.pi, 60)
        rx, zx = x[-1], z[-1]
        x = np.append(x, x[-1] + rl *
                      (np.cos(np.pi - theta) - np.cos(np.pi - asum)))
        z = np.append(z, z[-1] - rl * (np.sin(asum) - np.sin(np.pi - theta)))
        self.segments['x'].append(
            rx + rl * (np.cos(np.pi - theta) - np.cos(np.pi - asum)))
        self.segments['z'].append(
            zx - rl * (np.sin(asum) - np.sin(np.pi - theta)))
        x = np.append(x, x[::-1])[::-1]
        z = np.append(z, -z[::-1] + 2 * zo)[::-1]
        x, z = geom.rzSLine(x, z, self.npoints)  # distribute points
        x = {'x': x, 'z': z}
        x = close_loop(x, self.npoints)
        x['x'], x['z'] = geom.clock(x['x'], x['z'])
        self.xmin = x['x'].min()
        return x

    def plot(self, inputs={}):
        x = self.draw(inputs=inputs)
        plt.plot(x['x'], x['z'], '-', ms=8, color=0.4 * np.ones(3))
        plt.axis('equal')
        plt.axis('off')


class Dloop(object):  # Princton D
    def __init__(self, npoints=100, limits=True):
        self.npoints = npoints
        self.initalise_radii()
        self.name = 'Dloop'
        self.limits = limits

    def initalise_radii(self):
        self.xo = collections.OrderedDict()
        self.xo['x1'] = {'value': 4.486, 'lb': 3, 'ub': 5}  # inner radius
        self.xo['x2'] = {'value': 15.708, 'lb': 10, 'ub': 20}  # outer radius
        self.xo['dz'] = {'value': 0, 'lb': -10, 'ub': 10}  # vertical offset
        self.oppvar = list(self.xo.keys())

    def set_input(self, **kwargs):
        inputs = get_input(self.oppvar, **kwargs)
        for key in inputs:
            if key in self.xo:
                try:  # dict
                    for k in inputs[key]:
                        self.xo[key][k] = inputs[key][k]
                except TypeError:  # single value - object is not iterable
                    self.xo[key]['value'] = inputs[key]
                self.xo[key] = set_limit(self.xo[key], limits=self.limits)

    def get_xo(self):
        values = []
        for n in ['x1', 'x2', 'dz']:
            values.append(self.xo[n]['value'])
        return values

    def draw(self, **kwargs):
        self.npoints = kwargs.get('npoints', self.npoints)
        self.set_input(**kwargs)
        self.segments = {'x': [], 'z': []}
        x1, x2, dz = self.get_xo()
        xo = np.sqrt(x1 * x2)
        k = 0.5 * np.log(x2 / x1)
        theta = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 2 * self.npoints)
        x, z = np.zeros(2 * self.npoints), np.zeros(2 * self.npoints)
        s = np.zeros(2 * self.npoints, dtype='complex128')
        n = 0
        while True:  # sum convergent series
            n += 1
            ds = 1j / n * (np.exp(-1j * n * theta) - 1) *\
                (1 + np.exp(1j * n * (theta + np.pi))) *\
                np.exp(1j * n * np.pi / 2) * \
                (besl(n - 1, k) + besl(n + 1, k)) / 2
            s += ds
            if np.max(abs(ds)) < 1e-14:
                break
        z = abs(xo * k * (besl(1, k) * theta + s))
        x = xo * np.exp(k * np.sin(theta))
        z -= np.mean(z)
        x, z = geom.space(x, z, self.npoints)
        z += dz  # vertical shift
        self.segments['x'].append([x[-1], x[0]])
        self.segments['z'].append([z[-1], z[0]])
        self.segments['x'].append(x)
        self.segments['z'].append(z)
        self.cs = {'x': x[-1], 'z': z[-1]}  # top of nose (cs seat)
        x = {'x': x, 'z': z}
        x = close_loop(x, self.npoints)
        x['x'], x['z'] = geom.clock(x['x'], x['z'])
        return x

    def plot(self, inputs={}):
        x = self.draw(inputs=inputs)
        plt.plot(x['x'], x['z'], '-', ms=8, color=0.4 * np.ones(3))
        # for x, z in zip(self.segments['x'], self.segments['z']):
        #    plt.plot(x, z, lw=3)
        plt.axis('equal')
        plt.axis('off')


class Sloop(object):  # polybezier
    def __init__(self, npoints=200, symetric=False, tension='full',
                 limits=True):
        self.name = 'Sloop'
        self.symetric = symetric
        self.tension = tension
        self.limits = limits
        self.npoints = npoints
        self.initalise_nodes()
        self.set_symetric()
        self.set_tension()

    def initalise_nodes(self):
        self.xo = collections.OrderedDict()
        self.xo['x1'] = {'value': 4.486, 'lb': 3, 'ub': 8}  # inner radius
        self.xo['x2'] = {'value': 15.708, 'lb': 5, 'ub': 25}  # outer radius
        self.xo['z2'] = {'value': 0, 'lb': -0.9,
                         'ub': 0.9}  # outer node vertical shift
        self.xo['height'] = {'value': 17.367,
                             'lb': 0.1, 'ub': 50}  # full loop height
        self.xo['top'] = {'value': 0.5, 'lb': 0.05,
                          'ub': 1}  # horizontal shift
        self.xo['upper'] = {'value': 0.7, 'lb': 0.2, 'ub': 1}  # vertical shift
        self.set_lower()  # lower loop parameters (bottom,lower)
        self.xo['dz'] = {'value': 0, 'lb': -5, 'ub': 5}  # vertical offset
        # fraction outboard straight
        self.xo['flat'] = {'value': 0, 'lb': 0, 'ub': 0.8}
        self.xo['tilt'] = {'value': 0, 'lb': -
                           45, 'ub': 45}  # outboard angle [deg]
        self.oppvar = list(self.xo.keys())
        self.lkeyo = ['l0s', 'l0e', 'l1s', 'l1e', 'l2s', 'l2e', 'l3s', 'l3e']
        self.set_l({'value': 0.8, 'lb': 0.45, 'ub': 1.8})  # 1/tesion

    def remove_oppvar(self, name):
        try:
            self.oppvar.remove(name)
        except ValueError:
            pass

    def reset_oppvar(self, symetric):
        self.initalise_nodes()
        self.oppvar = list(self.xo.keys())
        self.symetric = symetric
        self.set_symetric()
        self.set_tension()

    def adjust_xo(self, name, **kwargs):  # value,lb,ub
        for var in kwargs:
            if name == 'l':
                for lkey in self.lkeyo:
                    self.xo[lkey][var] = kwargs[var]
            else:
                self.xo[name][var] = kwargs[var]
        self.set_symetric()
        self.set_tension()

    def check_tension_length(self, tension):
        tension = tension.lower()
        options = collections.OrderedDict()
        options['full'] = 8
        options['half'] = 4
        options['dual'] = 2
        options['single'] = 1
        if tension in options:
            self.nl = options[tension]
        else:
            errtxt = '\n'
            errtxt += 'Select Sloop tension length multiple from:\n'
            for option in options:
                errtxt += '\'{}\''.format(option)
                errtxt += ' (nl={:1.0f})\n'.format(options[option])
            raise ValueError(errtxt)

    def set_lower(self):
        for u, l in zip(['top', 'upper'], ['bottom', 'lower']):
            self.xo[l] = {}
            for key in self.xo[u]:
                self.xo[l][key] = self.xo[u][key]

    def enforce_symetric(self):
        self.symetric = True
        self.set_symetric()
        self.set_tension()

    def set_symetric(self):
        if self.symetric:  # set lower equal to upper
            self.xo['tilt']['value'] = 0
            self.xo['z2']['value'] = self.xo['dz']['value']
            if 'z2' in self.oppvar:
                self.oppvar.remove('z2')
            for u, l in zip(['top', 'upper'], ['bottom', 'lower']):
                self.xo[l] = self.xo[u]
                if l in self.oppvar:  # remove lower from oppvar
                    self.oppvar.remove(l)
            if 'tilt' in self.oppvar:
                self.oppvar.remove('tilt')

    def set_tension(self):
        tension = self.tension.lower()
        if self.symetric:
            if tension == 'full':
                self.tension = 'half'
            elif tension == 'half':
                self.tension = 'dual'
            else:
                self.tension = tension
        else:
            self.tension = tension

        self.check_tension_length(self.tension)
        if self.tension == 'single':
            self.lkey = ['l', 'l', 'l', 'l', 'l', 'l', 'l', 'l']
        elif self.tension == 'dual':
            self.lkey = ['l0', 'l0', 'l1', 'l1', 'l1', 'l1', 'l0', 'l0']
        elif self.tension == 'half':
            if self.symetric:
                self.lkey = ['l0s', 'l0e', 'l1s',
                             'l1e', 'l1e', 'l1s', 'l0e', 'l0s']
            else:
                self.lkey = ['l0', 'l0', 'l1', 'l1', 'l2', 'l2', 'l3', 'l3']
        elif self.tension == 'full':
            self.lkey = ['l0s', 'l0e', 'l1s',
                         'l1e', 'l2s', 'l2e', 'l3s', 'l3e']
        oppvar = np.copy(self.oppvar)  # remove all length keys
        for var in oppvar:
            if var[0] == 'l' and var != 'lower':
                self.oppvar.remove(var)
        for oppkey in np.unique(self.lkey):  # re-populate
            self.oppvar.append(oppkey)
        for i, (lkey, lkeyo) in enumerate(zip(self.lkey, self.lkeyo)):
            if len(lkey) == 1:
                lkey += '0s'
            elif len(lkey) == 2:
                lkey += 's'
            self.xo[lkeyo] = self.xo[lkey].copy()

    def get_xo(self):
        values = []
        for var in ['x1', 'x2', 'z2', 'height', 'top',
                    'bottom', 'upper', 'lower', 'dz', 'flat', 'tilt']:
            if var not in self.xo:
                if var == 'bottom':
                    var = 'top'
                if var == 'lower':
                    var = 'upper'
            values.append(self.xo[var]['value'])
        return values

    def set_l(self, l):
        for lkey in self.lkeyo:
            self.xo[lkey] = l.copy()

    def get_l(self):
        ls, le = [], []  # start,end
        for i in range(4):
            ls.append(self.xo['l{:1.0f}s'.format(i)]['value'])
            le.append(self.xo['l{:1.0f}e'.format(i)]['value'])
        return ls, le

    def basis(self, t, v):
        n = 3  # spline order
        return bn(n, v) * t**v * (1 - t)**(n - v)

    def midpoints(p):  # convert polar to cartesian
        x = p['x'] + p['l'] * np.cos(p['t'])
        z = p['z'] + p['l'] * np.sin(p['t'])
        return x, z

    def control(p0, p3):  # add control points (length and theta or midpoint)
        p1, p2 = {}, {}
        xm, ym = np.mean([p0['x'], p3['x']]), np.mean([p0['z'], p3['z']])
        dl = sp.linalg.norm([p3['x'] - p0['x'], p3['z'] - p0['z']])
        for p, pm in zip([p0, p3], [p1, p2]):
            if 'l' not in p:  # add midpoint length
                p['l'] = dl / 2
            else:
                p['l'] *= dl / 2
            if 't' not in p:  # add midpoint angle
                p['t'] = np.arctan2(ym - p['y'], xm - p['x'])
            pm['x'], pm['z'] = Sloop.midpoints(p)
        return p1, p2, dl

    def append_keys(self, key):
        if key[0] == 'l' and key != 'lower':
            length = len(key)
            if length == 1:
                key = self.lkeyo  # control all nodes
            elif length == 2:
                key = [key + 's', key + 'e']
            else:
                key = [key]
        else:
            key = [key]
        return key

    def set_input(self, **kwargs):
        inputs = get_input(self.xo.keys(), **kwargs)  # (self.oppvar, **kwargs)
        for key in inputs:
            keyo = self.append_keys(key)
            for keyo_ in keyo:
                if keyo_ in self.xo:
                    try:  # dict
                        for k in inputs[key]:
                            self.xo[keyo_][k] = inputs[key][k]
                    except TypeError:  # single value - object is not iterable
                        self.xo[keyo_]['value'] = inputs[key]
                    self.xo[keyo_] = set_limit(self.xo[keyo_],
                                               limits=self.limits)
        self.set_symetric()
        self.set_tension()

    def verticies(self):
        x1, x2, z2, height, top, bottom,\
            upper, lower, dz, ds, alpha_s = self.get_xo()
        x, z, theta = np.zeros(6), np.zeros(6), np.zeros(6)
        alpha_s *= np.pi / 180
        ds_z = ds * height / 2 * np.cos(alpha_s)
        ds_r = ds * height / 2 * np.sin(alpha_s)
        x[0], z[0], theta[0] = x1, upper * \
            height / 2, np.pi / 2  # upper sholder
        x[1], z[1], theta[1] = x1 + top * (x2 - x1), height / 2, 0  # top
        x[2], z[2], theta[2] = x2 + ds_r, z2 * height / \
            2 + ds_z, -np.pi / 2 - alpha_s  # outer, upper
        x[3], z[3], theta[3] = x2 - ds_r, z2 * height / \
            2 - ds_z, -np.pi / 2 - alpha_s  # outer, lower
        x[4], z[4], theta[4] = x1 + bottom * \
            (x2 - x1), -height / 2, -np.pi  # bottom
        x[5], z[5], theta[5] = x1, -lower * \
            height / 2, np.pi / 2  # lower sholder
        z += dz  # vertical loop offset
        return x, z, theta

    def linear_loop_length(self, x, z):
        self.L = 0
        for i in range(len(x) - 1):
            self.L += sp.linalg.norm([x[i + 1] - x[i], z[i + 1] - z[i]])

    def segment(self, p, dl):
        n = int(np.ceil(self.npoints * dl / self.L))  # segment point number
        t = np.linspace(0, 1, n)
        curve = {'x': np.zeros(n), 'z': np.zeros(n)}
        for i, pi in enumerate(p):
            for var in ['x', 'z']:
                curve[var] += self.basis(t, i) * pi[var]
        return curve

    def polybezier(self, x, z, theta):
        p = {'x': np.array([]), 'z': np.array([])}
        self.po = []
        self.linear_loop_length(x, z)
        ls, le = self.get_l()
        for i, j, k in zip(range(len(x) - 1), [0, 1, 3, 4], [1, 2, 4, 5]):
            p0 = {'x': x[j], 'z': z[j], 't': theta[j], 'l': ls[i]}
            p3 = {'x': x[k], 'z': z[k], 't': theta[k] - np.pi, 'l': le[i]}
            p1, p2, dl = Sloop.control(p0, p3)
            curve = self.segment([p0, p1, p2, p3], dl)
            self.po.append({'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3})
            for var in ['x', 'z']:
                p[var] = np.append(p[var], curve[var][:-1])
        for var in ['x', 'z']:
            p[var] = np.append(p[var], curve[var][-1])
            p[var] = p[var][::-1]
        return p

    def draw(self, **kwargs):
        self.npoints = kwargs.get('npoints', self.npoints)
        self.set_input(**kwargs)
        x, z, theta = self.verticies()
        self.cs = {'x': x[0], 'z': z[0]}  # top of nose (cs seat)
        p = self.polybezier(x, z, theta)
        p = close_loop(p, self.npoints)
        p['x'], p['z'] = geom.clock(p['x'], p['z'])
        self.xmin = p['x'].min()
        return p

    def plot(self, inputs={}, ms=8):
        p = self.draw(inputs=inputs)
        x, z, theta = self.verticies()
        c1, c2 = 0.75 * np.ones(3), 0.4 * np.ones(3)
        plt.plot(x, z, 's', color=c1, ms=2 * ms, zorder=10)
        plt.plot(x, z, 's', color=c2, ms=ms, zorder=10)
        plt.plot(p['x'], p['z'], '-', color=c2, ms=ms)
        for po in self.po:
            plt.plot([po['p0']['x'], po['p1']['x']],
                     [po['p0']['z'], po['p1']['z']], color=c1, ms=ms, zorder=5)
            plt.plot(po['p1']['x'], po['p1']['z'], 'o',
                     color=c1, ms=2 * ms, zorder=6)
            plt.plot(po['p1']['x'], po['p1']['z'], 'o',
                     color=c2, ms=ms, zorder=7)
            plt.plot([po['p3']['x'], po['p2']['x']],
                     [po['p3']['z'], po['p2']['z']], color=c1, ms=ms, zorder=5)
            plt.plot(po['p2']['x'], po['p2']['z'], 'o',
                     color=c1, ms=2 * ms, zorder=6)
            plt.plot(po['p2']['x'], po['p2']['z'], 'o',
                     color=c2, ms=ms, zorder=7)
        plt.axis('equal')
        plt.axis('off')


class Profile(object):

    def __init__(self, name, family='S', part='TF', npoints=200,
                 symetric=False, read_write=True, **kwargs):
        self.name = name
        self.part = part
        self.read_write = read_write
        data_dir = nova_path('Data')
        self.dataname = data_dir + self.name + '_{}.pkl'.format(part)
        self.nTF = kwargs.get('nTF', 'unset')
        self.obj = kwargs.get('obj', 'L')
        self.update(npoints=npoints, symetric=symetric, family=family)

    def update(self, **kwargs):
        # family: A==arc, D==Princton-D, S==spline
        for key in ['npoints', 'symetric', 'family', 'nTF', 'L']:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        self.initalise_loop()  # initalize loop object
        self.read_loop_dict()

    def initalise_loop(self):
        if self.family == 'A':  # tripple arc (5-7 parameter)
            self.loop = Aloop(npoints=self.npoints)
        elif self.family == 'D':  # princton D (3 parameter)
            self.loop = Dloop(npoints=self.npoints)
        elif self.family == 'S':  # polybezier (8-16 parameter)
            self.loop = Sloop(npoints=self.npoints, symetric=self.symetric)
        else:
            errtxt = '\n'
            errtxt += 'loop type \'' + self.family + '\'\n'
            errtxt += 'select from [A,D,S]\n'
            raise ValueError(errtxt)

    def read_loop_dict(self):
        if self.read_write:  # atempt to read loop from file
            try:
                with open(self.dataname, 'rb') as input:
                    self.loop_dict = pickle.load(input)
            except FileNotFoundError:
                print('file ' + self.dataname + ' not found')
                print('initializing new loop_dict')
                self.loop_dict = {}
            self.frame_data()
            try:
                self.load(nTF=self.nTF, obj=self.obj)
            except KeyError:
                wstr = 'loop parameters '
                wstr += 'nTF:\'{}\', obj:\'{}\''.format(self.nTF, self.obj)
                wstr += ' not avalible\n'
                print(wstr)
                self.avalible_data()
        else:
            self.loop_dict = {}
        self.frame_data()

    def frame_data(self):
        self.data_frame = {}
        for family in self.loop_dict:
            data = {}
            for nTF in self.loop_dict[family]:
                data[nTF] = {}
                for obj in self.loop_dict[family][nTF]:
                    data[nTF][obj] = True
            self.data_frame[family] = pd.DataFrame(data)

    def load(self, nTF='unset', obj='L'):
        data = self.loop_dict.get(self.family, {})
        if nTF in self.loop_dict[self.family]:
            data = data.get(nTF, {})
        if obj in data:
            loop_dict = data[obj]
            for key in loop_dict:
                if hasattr(self.loop, key):
                    setattr(self.loop, key, loop_dict[key])
        else:
            errtxt = '\n'
            errtxt += 'data not found:\n'
            errtxt += 'loop type {}, nTF {}, obj {}\n'.\
                format(self.family, nTF, obj)
            errtxt += self.avalible_data(verbose=False)
            raise ValueError(errtxt)

    def avalible_data(self, verbose=True):
        if len(self.loop_dict) == 0:
            datatxt = 'no data avalible'
        else:
            datatxt = '\n{}: data avalible [obj,nTF]'.format(self.name)
            for family in self.data_frame:
                datatxt += '\n\nloop type {}:\n{}'.\
                    format(family, self.data_frame[family].fillna(''))
        if verbose:
            print(datatxt)
        else:
            return datatxt

    def clear_data(self):
        with open(self.dataname, 'wb') as output:
            self.loop_dict = {}
            pickle.dump(self.loop_dict, output, -1)

    def write(self):  # write xo and oppvar to file
        if self.family in self.loop_dict:
            if self.nTF not in self.loop_dict[self.family]:
                self.loop_dict[self.family][self.nTF] = {self.obj: []}
        else:
            self.loop_dict[self.family] = {self.nTF: {self.obj: []}}
        cdict = {}
        for key in ['xo', 'oppvar', 'family', 'symetric', 'tension', 'limits']:
            if hasattr(self.loop, key):
                cdict[key] = getattr(self.loop, key)
        self.loop_dict[self.family][self.nTF][self.obj] = cdict
        if self.read_write:  # write loop to file
            with open(self.dataname, 'wb') as output:
                pickle.dump(self.loop_dict, output, -1)
        self.frame_data()


if __name__ is '__main__':  # plot loop classes
    # loop = Aloop()
    # x = loop.plot()
    # loop = Sloop(limits=False, symetric=False, tension='single')
    # loop.set_tension('full')
    # x = loop.plot({'l2':1.5})
    # loop.draw()
    profile = Profile('demo', family='D', load=True,
                      part='TF', nTF=16, obj='L', npoints=500)

    # profile.update(family='D')

    profile.loop.plot()
