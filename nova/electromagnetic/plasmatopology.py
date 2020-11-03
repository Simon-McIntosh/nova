import pygmo as pg
import shapely.geometry
import numpy as np

from nova.electromagnetic.coilset import CoilSet
from nova.utilities.pyplot import plt


class Stencil:

    def __init__(self, grid_boundary):
        self.grid_boundary = grid_boundary

    def get_bounds(self):
        return (self.grid_boundary[::2], self.grid_boundary[1::2])


class FieldNull(Stencil):

    def __init__(self, grid_boundary, B_rbs, Psi_rbs):
        Stencil.__init__(self, grid_boundary)
        self.B_rbs = B_rbs
        self.Psi_rbs = Psi_rbs

    def fitness(self, x):
        return [self.B_rbs.ev(*x)]

    def get_nobj(self):
        return 1

    def gradient(self, x):
        return [self.B_rbs.ev(*x, dx=1), self.B_rbs.ev(*x, dy=1)]


if __name__ == '__main__':

    cs = CoilSet()
    '''
    polygon = shapely.geometry.Point(5, 1).buffer(0.5)
    cs.add_plasma(polygon, dPlasma=0.1)
    cs.add_coil(5, -0.5, 0.75, 0.75, dCoil=0.2)
    cs.plasmagrid.generate_grid(expand=1, n=2e4)  # generate plasma grid
    cs.Ic = [15e6, 15e6]

    cs.save_coilset('Xtest')
    '''
    cs.load_coilset('Xtest')
    #nl = pg.nlopt('slsqp')
    #nl.xtol_rel = 1E-6




    '''
    algo = pg.algorithm(pg.nlopt('mma'))
    algo.set_verbosity(1)
    prob = pg.problem(FieldNull(cs.plasmagrid.grid_boundary,
                                cs.plasmagrid.B_rbs,
                                cs.plasmagrid.Psi_rbs))


    #prob.c_tol = [1E-6]  # Set constraints tolerance to 1E-6
    #def ev():
    pop = pg.population(prob, 50)


    pop = algo.evolve(pop)


    plt.figure()
    cs.plot(True)
    cs.plasmagrid.plot_flux()

    plt.plot(*pop.get_x().T, 'o')
    plt.plot(*pop.champion_x, 'X')
    '''

