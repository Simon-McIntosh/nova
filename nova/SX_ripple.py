import pylab as pl
import numpy as np
from matplotlib import animation, gridspec
from shelf import PKL
import seaborn as sns
rc = {'figure.figsize':[6,6*7.5/24],'savefig.dpi':150, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.75}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=0.5,rc=rc)
color = sns.color_palette('Set2')
from amigo.addtext import linelabel

from itertools import cycle
Color = cycle(sns.color_palette('Set2'))
from radial_build import RB
#from eqConfig import Config
#from nova.config import Setup


#conf = Config('SXex')
#setup = Setup('SXex')

pkl = PKL('moveSX')  #'moveSX_dev2'
sf,rb = pkl.fetch(['sf','rb'])
