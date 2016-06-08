import pylab as pl
import numpy as np
import matplotlib as mpl
from itertools import cycle
from eqConfig import Config
from radial_build import RB
from streamfunction import SF
from elliptic import EQ
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from addtext import linelabel
from shelf import PKL

import seaborn as sns
Color = cycle(sns.color_palette('Set2'))
'''
rc = {'figure.figsize':[6*12/16,6],'savefig.dpi':200, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':0.75}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

'''
pl.figure()
pl.axis('equal')
pl.axis('off')



pkl = PKL('moveSX')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])



inv.swing_fix(60)
inv.solve() 

inv.update_coils(plot=True)
sf.plot_coils(Color,coils=sf.coil,label=False,plasma=False,current=True) 
sf.plot_coils(Color,coils=eq.coil,label=False,plasma=False) 
 
eq.run()
sf.contour(Plasma=True)  # ,levels=np.linspace(-60,60,40)
eq.plasma()
eq.plotb()
inv.plot_fix()
'''
from matplotlib.animation import FuncAnimation

class UpdateDist(object):
    def __init__(self, ax, prob=0.5):
        self.success = 0
        self.prob = prob
        self.line, = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 200)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 15)
        self.ax.grid(True)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.ax.axvline(prob, linestyle='--', color='black')

    def init(self):
        self.success = 0
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            return self.init()

        # Choose success based on exceed a threshold with a uniform pick
        if np.random.rand(1,) < self.prob:
            self.success += 1
        y = ss.beta.pdf(self.x, self.success + 1, (i - self.success) + 1)
        self.line.set_data(self.x, y)
        return self.line,

fig, ax = pl.subplots()
ud = UpdateDist(ax, prob=0.7)
anim = FuncAnimation(fig, sf.contour, frames=np.arange(100), init_func=ud.init,
                     interval=100, blit=True)
pl.show()

'''


