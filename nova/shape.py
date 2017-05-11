import pylab as pl
import numpy as np
from amigo import geom
import seaborn as sns
from scipy.optimize import fmin_slsqp
import time
from nova.DEMOxlsx import DEMO
from nova.loops import set_oppvar, get_oppvar, plot_oppvar, Profile
from nova.config import select
import matplotlib.animation as manimation
from itertools import cycle
from amigo.time import clock


class Shape(object):

    def __init__(self, profile, **kwargs):
        self.color = cycle(sns.color_palette('Set2', 10))
        self.profile = profile
        self.obj = kwargs.get('objective', profile.obj)
        self.loop = self.profile.loop
        self.bound = {}  # initalise bounds
        self.bindex = {'internal': [0], 'interior': [0], 'external': [0]}
        for side in ['internal', 'interior', 'external']:
            self.bound[side] = {'r': [], 'z': []}
            if side in kwargs:
                self.add_bound(kwargs[side], side)
        # define optimisation functions
        self.objective = self.geometric_objective
        self.constraints = self.geometric_constraints
        self.args = ()

    def add_bound(self, x, side):
        for var in ['r', 'z']:
            self.bound[side][var] = np.append(self.bound[side][var], x[var])
        self.bindex[side].append(len(self.bound[side]['r']))

    def add_interior(self, r_gap=0.001):  # offset minimum internal radius
        argmin = np.argmin(self.bound['internal']['r'])
        self.add_bound({'r': self.bound['internal']['r'][argmin] - r_gap,
                        'z': self.bound['internal']['z'][argmin]},
                       'interior')

    def clear_bound(self):
        for side in self.bound:
            for var in ['r', 'z']:
                self.bound[side][var] = np.array([])

    def plot_bounds(self):
        for side, marker in zip(['internal', 'interior', 'external'],
                                ['.-', 'd', 's']):
            index = self.bindex[side]
            for i in range(len(index) - 1):
                pl.plot(self.bound[side]['r'][index[i]:index[i + 1]],
                        self.bound[side]['z'][index[i]:index[i + 1]],
                        marker, markersize=6, color=next(self.color))

    def geometric_objective(self, xnorm, *args):
        xo = get_oppvar(self.loop.xo, self.loop.oppvar, xnorm)  # de-normalize
        if hasattr(self, 'xo'):
            self.xo = np.vstack([self.xo, xo])
        else:
            self.xo = xo
        x = self.loop.draw(x=xo)
        if self.obj == 'L':  # loop length
            objF = geom.length(x['r'], x['z'], norm=False)[-1]
        elif self.obj == 'V':  # loop volume (torus)
            objF = geom.loop_vol(x['r'], x['z'])
        else:
            errtxt = 'objective {} '.format(self.profile.obj)
            errtxt += 'not defined within gemetric_objective function'
            raise ValueError(errtxt)
        return objF

    def dot_diffrence(self, x, side):
        Rloop, Zloop = x['r'], x['z']  # inside coil loop
        switch = 1 if side is 'internal' else -1
        nRloop, nZloop, Rloop, Zloop = geom.normal(Rloop, Zloop)
        R, Z = self.bound[side]['r'], self.bound[side]['z']
        dot = np.zeros(len(R))
        for j, (r, z) in enumerate(zip(R, Z)):
            i = np.argmin((r - Rloop)**2 + (z - Zloop)**2)
            dr = [Rloop[i] - r, Zloop[i] - z]
            dn = [nRloop[i], nZloop[i]]
            dot[j] = switch * np.dot(dr, dn)
        return dot

    def geometric_constraints(self, xnorm, *args):
        xo = get_oppvar(self.loop.xo, self.loop.oppvar, xnorm)  # de-normalize
        x = self.loop.draw(x=xo)
        constraint = np.array([])
        for side in ['internal', 'interior']:
            constraint = np.append(constraint, self.dot_diffrence(x, side))
        return constraint

    def update(self, xnorm, *args):
        # empty function - overloaded externaly
        return 0

    def minimise(self, verbose=False, acc=0.002):
        tic = time.time()
        iprint = 1 if verbose else -1
        xnorm, bnorm = set_oppvar(self.loop.xo, self.loop.oppvar)  # normalize
        xnorm = fmin_slsqp(self.objective, xnorm, f_ieqcons=self.constraints,
                           bounds=bnorm, acc=acc, iprint=iprint,
                           args=self.args)
        xo = get_oppvar(self.loop.xo, self.loop.oppvar, xnorm)  # de-normalize
        self.loop.set_input(x=xo)  # inner loop
        self.profile.write()  # store loop
        self.update(xnorm, *self.args)  # update loop in calling function
        if verbose:
            self.toc(tic)

    def toc(self, tic):
        print('optimisation time {:1.1f}s'.format(time.time() - tic))
        print('noppvar {:1.0f}'.format(len(self.loop.oppvar)))

    def movie(self, filename):
        fig, ax = pl.subplots(1, 2, figsize=(12, 8))
        demo = DEMO()
        moviename = '../Movies/{}'.format(filename)
        moviename += '.mp4'
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=20, bitrate=5000, codec='libx264',
                              extra_args=['-pix_fmt', 'yuv420p'])
        timer = clock(len(self.xo))
        with writer.saving(fig, moviename, 100):
            for i, xo in enumerate(self.xo):
                self.frame(ax, demo, xo=xo)
                writer.grab_frame()
                timer.ticktoc()

    def frames(self, filename):
        fig, ax = pl.subplots(1, 2, figsize=(12, 8))
        demo = DEMO()
        figname = '../Figs/{}'.format(filename)
        self.frame(ax, demo, xo=self.xo[0])
        pl.savefig(figname + '_s.png')
        self.frame(ax, demo, xo=self.xo[-1])
        pl.savefig(figname + '_e.png')

    def frame(self, ax, demo, **kwargs):
        xo = kwargs.get('xo', self.xo[-1])
        pl.sca(ax[0])
        # pl.cla()
        pl.plot([3, 18], [-10, 10], 'ko', alpha=0)
        demo.fill_part('Blanket')
        demo.fill_part('Vessel')

        self.loop.set_input(x=xo)
        # self.plot_bounds()
        # self.update()
        # self.tf.fill()
        geom.polyfill(self.cage.plasma_loop[:, 0],
                      self.cage.plasma_loop[:, 2],
                      alpha=0.3, color=sns.color_palette('Set2', 5)[3])
        # self.cage.plot_loops(sticks=False)
        if len(ax) > 1:
            pl.sca(ax[1])
            pl.cla()
            plot_oppvar(shp.loop.xo, shp.loop.oppvar)

    '''
    def toc(self, tic):
    print('optimisation time {:1.1f}s'.format(time.time() - tic))
    print('noppvar {:1.0f}'.format(len(self.loop.oppvar)))
    if self.profile.nTF is not 'unset':
        self.cage.output()
    '''





    '''
    def fit(self, xnorm, *args):
        xo = get_oppvar(self.loop.xo, self.loop.oppvar, xnorm)  # de-normalize
        if hasattr(self, 'xo'):
            self.xo = np.vstack([self.xo, xo])
        else:
            self.xo = xo
        x = self.loop.draw(x=xo)
        if self.profile.obj is 'L':  # loop length
            objF = geom.length(x['r'], x['z'], norm=False)[-1]
        elif self.obj is 'E':  # stored energy
            x = self.tf.get_loops(x=x)
            self.cage.set_TFcoil(x['cl'])
            objF = 1e-9 * self.cage.energy()
        else:  # loop volume (torus)
            objF = geom.loop_vol(x['r'], x['z'])
        return objF
    '''


if __name__ is '__main__':

    nTF = 16
    family = 'S'
    ripple = False

    config = {'TF': 'demo', 'eq': 'SN'}
    config, setup = select(config, nTF=nTF)

    demo = DEMO()
    profile = Profile(config['TF'], family=family, part='TF', nTF=nTF)
    shp = Shape(profile, obj='L')

    shp.add_vessel(demo.parts['Vessel']['out'])
    shp.minimise(verbose=True)
    shp.plot_bounds()

    '''

    profile = Profile(config['TF'], family=family,
                      part='TF', nTF=nTF)  # ,load=False
    shp = Shape(profile, nTF=nTF, obj='L', eqconf=config['eq_base'], ny=1)
    shp.add_vessel(demo.parts['Vessel']['out'])
    shp.minimise(ripple=ripple, verbose=False)
    cage = shp.cage

    # shp.update()
    # shp.tf.fill()
    # shp.loop.plot({'flat':0.3,'tilt':13})
    # shp.loop.plot()
    # demo.fill_part('TF_Coil',alpha=0.8)
    # shp.cage.plot_contours(variable='ripple',n=2e3,loop=demo.fw)
    # shp.cage.pattern(plot=True)
    # plot_oppvar(shp.loop.xo,shp.loop.oppvar)


    #x_in = demo.parts['TF_Coil']['in']
    #tf = TF(x_in=x_in,nTF=nTF)
    #x = tf.get_loops(x_in)
    #cage = coil_cage(nTF=18,rc=tf.rc,plasma={'config':config['eq']},ny=3)
    #cage.set_TFcoil(x['cl'],smooth=True)


    Vol = cage.get_volume()
    print('')
    print('nTF {:1.0f}'.format(nTF))
    print('ripple {:1.3f}'.format(cage.get_ripple()))
    print('energy {:1.2f} GJ'.format(1e-9 * cage.energy()))
    print(r'TF volume {:1.0f} m3'.format(Vol['TF']))
    print(r'plasma volume {:1.0f} m3'.format(Vol['plasma']))
    print('ratio {:1.2f}'.format(Vol['ratio']))

    fig, ax = pl.subplots(1, 1, figsize=(8, 10))
    pl.plot([3, 18], [-10, 10], 'ko', alpha=0)
    demo.fill_part('Blanket')
    demo.fill_part('Vessel')

    # demo.fill_part('TF_Coil',color=0.75*np.ones(3))

    shp.tf.fill()

    # cage.plot_contours(variable='ripple',n=2e3,loop=demo.fw)  # 2e5
    pl.axis('off')

    pl.savefig('../Figs/ripple_referance')


    filename = '{}_{}_{}'.format(config['TF'],family,ripple)
    shp.movie(filename)
    #shp.frames(filename)
    #pl.savefig('../Figs/TFloop_{}.png'.format(family))
    '''
