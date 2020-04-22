import numpy as np
from nova.force import force_field
from nep.coil_geom import PFgeom, VSgeom
from nep.DINA.read_tor import read_tor
from collections import OrderedDict
import nova.cross_coil as cc
from nova.streamfunction import SF
from amigo.geom import grid
from nep.DINA.read_plasma import read_plasma
from amigo.pyplot import plt
from amigo.time import clock
from amigo.geom import qrotate
import matplotlib.animation as manimation
from nep.rails import stress_allowable
from nep.DINA.read_dina import dina
import matplotlib.patches as mpatches
import pickle
from os.path import join, isfile
import matplotlib.lines as mlines
import os
import nep
from amigo.png_tools import data_load
from amigo.IO import class_dir


class VS3():

    def __init__(self):
        vs_geom = VSgeom()
        self.pf = vs_geom.pf

        pf_geom = PFgeom(VS=True)
        self.pf = pf_geom.pf
        #self.pf.mesh_coils(dCoil=0.25)

if __name__ == '__main__':

    vs3 = VS3()

