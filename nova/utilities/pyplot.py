from os import path

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from mpl_toolkits import axes_grid1
from matplotlib.transforms import Bbox
from matplotlib.colors import to_rgb
from matplotlib.patches import Arc, RegularPolygon
from matplotlib.path import Path


def ring_coding(ob):
    '''
    taken from: https://sgillies.net/2010/04/06/
    painting-punctured-polygons-with-matplotlib.html
    '''
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    return codes


def pathify(polygon):
    '''
    taken from: https://sgillies.net/2010/04/06/
    painting-punctured-polygons-with-matplotlib.html
    '''
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.
    vertices = np.concatenate(
                    [np.asarray(polygon.exterior)]
                    + [np.asarray(r) for r in polygon.interiors])
    codes = np.concatenate(
                [ring_coding(polygon.exterior)]
                + [ring_coding(r) for r in polygon.interiors])
    return Path(vertices, codes)


class rail_patch:

    def __init__(self, x, y):
        extents = np.array([np.min(x), np.min(y), np.max(x), np.max(y)])
        self.Bbox = Bbox.from_extents(*extents)

    def get_window_extent(self, *args):
        return self.Bbox


def set_xticks(n, ax=None, ntick=20, whitespace=False):
    if n > ntick:
        space = int(n / ntick) + 1
    else:
        space = 1
    ticklabels = ['{:1d}'.format(i) if i % space == 0 else ''
                  for i in range(n)]
    if whitespace:
        X = np.arange(n)
    else:
        X = np.arange(0, n, space)
        ticklabels = ticklabels[::space]
    if ax is not None:
        ax.set_xticks(X)
        ax.set_xticklabels(ticklabels)
    else:
        plt.xticks(X, ticklabels)


def add_colorbar(im, aspect=20, pad_fraction=1.5, **kwargs):
    ''' add a vertical color bar to an image plot '''
    ax = kwargs.get('ax', plt.gca())
    divider = axes_grid1.make_axes_locatable(ax)
    width = axes_grid1.axes_size.AxesY(ax, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes('right', size=width, pad=pad)
    plt.sca(current_ax)
    return plt.colorbar(im, cax=cax, **kwargs)


def arrow_arc(x, y, r, angle=0, gap=0.4*np.pi, ax=None, color='C0', **kwargs):
    if ax is None:
        ax = plt.gca()
    reverse = False if r > 0 else True
    radius = abs(r)
    width = height = radius  # arc radius
    linewidth = 2.5*radius
    if reverse:
        rotation = gap-angle
        theta = np.array([180/np.pi*gap, 0])
        flip = np.pi
    else:
        rotation = 2*np.pi-gap+angle
        theta = np.array([0, 180/np.pi*(2*np.pi-gap)])
        flip = 0
    arc = Arc((x, y), width, height, 180/np.pi*angle,
              theta1=theta[0], theta2=theta[1],
              capstyle='round', linewidth=linewidth, color=color, **kwargs)
    ax.add_patch(arc)
    x1 = x + radius/2*np.cos(rotation)
    y1 = y + radius/2*np.sin(rotation)
    ax.add_patch(RegularPolygon((x1, y1), 3, radius/6,
                                flip+rotation, color=color, **kwargs))


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class plstyle:

    def __init__(self, context='notebook'):
        self.reset()  # reset style
        #self.style_dir = path.join(root_dir, 'style_sheets')
        self.context = context
        self.sns_contex(context)
        # mpl.rcParams['figure.dpi'] = 120
        # mpl.rcParams['figure.figsize'] = np.array([4, 3]) / 0.394
        # mpl.rcParams['font.size'] = 18
        # self.use('SOFE2')

    def font_scale(self, scale):
        sns.set_context(self.context, scale)

    def sns_contex(self, context):
        '''
        set sns context style sheet
        '''
        try:
            plt.style.use('seaborn-{}'.format(context))
        except OSError:
            errtxt = 'context \'{}\''.format(context)
            errtxt += 'not avalible as seabone style\n'
            errtxt += 'specify plot context as:\n'
            errtxt += '\'paper\', \'notebook\', \'talk\' or \'poster\''
            raise OSError(errtxt)

    def use(self, style_sheet):
        if style_sheet in plt.style.available:
            plt.use(style_sheet)
        else:
            plt.style.use(path.join(self.style_dir, style_sheet + '.mplstyle'))

    def set_aspect(self, aspect):
        width = mpl.rcParams['figure.figsize'][0]
        height = aspect * width
        mpl.rcParams['figure.figsize'] = [width, height]

    def avalible(self):
        print(plt.style.available)

    def reset(self):
        # mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use('default')


class multimap(object):
    # generate single colorbar from multiple mappables

    def __init__(self):
        self.mappable = []
        self.clim = []

    def add(self, mappable):
        self.mappable.append(mappable)
        self.clim.append(mappable.get_clim())

    def colorbar(self, *args, clim=None, **kwargs):
        if clim is None:
            vmin, vmax = self.clim[0]
            for clim in self.clim[1:]:
                vmin = np.min([vmin, clim[0]])
                vmax = np.max([vmax, clim[1]])
        else:
            vmin, vmax = clim
        self.index = np.argmax(np.diff(self.clim)[0], axis=0)
        referance = self.mappable[self.index]
        cmap = kwargs.get('cmap', referance.get_cmap())
        vmin = kwargs.get('vmin', vmin)
        vmax = kwargs.get('vmax', vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        sm._A = []
        for mappable in self.mappable:
            mappable.set_cmap(cmap)
            mappable.set_clim((vmin, vmax))
        kwargs.pop('vmin', None)
        kwargs.pop('vmax', None)
        cb = plt.colorbar(sm, *args, **kwargs)
        return cb


def detick(ax):  # remove xticks from upper subplots
    for i in range(len(ax) - 1):
        plt.setp(ax[i].get_xticklabels(), visible=False)
        ax[i].set_xlabel('')

def insert_yticks(ticks, ax=None, N=8):
    if ax is None:
        ax = plt.gca()
    ylim = ax.get_ylim()
    dy = ylim[-1] - ylim[0]
    yticks = ax.get_yticks()
    if not hasattr(ticks, '__iter__'):
        ticks = [ticks]
    for tick in ticks:
        yticks = yticks[abs(yticks-tick) > dy / (2*N)]
    yticks = np.append(yticks, ticks)
    ax.set_yticks(np.sort(yticks))


mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

plt.despine = sns.despine  # add despine function to plt object
plt.set_context = sns.set_context
plt.plstyle = plstyle()
plt.detick = detick
plt.add_colorbar = add_colorbar
plt.set_xticks = set_xticks
plt.insert_yticks = insert_yticks
plt.set_aspect = plt.plstyle.set_aspect

Cn = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.Cn = [to_rgb(from_hex) for from_hex in Cn]  # rgb color index

plt.set_context('notebook')
plt.set_context('talk')
plt.set_context('poster')

if __name__ == '__main__':

    from numpy.random import rand

    cmap = plt.cm.viridis
    mmap = multimap()
    fig, ax = plt.subplots(3, 3)
    cax = fig.add_axes([0.2, 0, 0.6, 0.04])
    fig.text(0.5, 0.95, 'Multiple images', ha='center')

    for i, a in enumerate(ax.flatten()):
        data = ((1 + i)/10.0)*rand(10, 20)*1e-6
        im = a.imshow(data, cmap=cmap)
        mmap.add(im)
    for i in range(2):
        for a in ax[i, :]:
            a.set_xticks([])

    cb = mmap.colorbar(cax, orientation='horizontal')
    cb.set_label('colors')
