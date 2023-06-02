# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:44:26 2020

@author: mcintos
"""


class Stencil:
    def __init__(self, grid_boundary):
        self.grid_boundary = grid_boundary

    def get_bounds(self):
        return (self.grid_boundary[::2], self.grid_boundary[1::2])


class FieldNull(Stencil):
    def __init__(self, grid_boundary, interpolate):
        Stencil.__init__(self, grid_boundary)
        self.interpolate = interpolate

    def fitness(self, x):
        return [self.interpolate("B").ev(*x).item()]

    def get_nobj(self):
        return 1

    def gradient(self, x):
        return [
            self.interpolate("B").ev(*x, dx=1).item(),
            self.interpolate("B").ev(*x, dy=1).item(),
        ]

    def get_Xpoint_pygmo(self, xo):
        uda = pg.nlopt("slsqp")
        algo = pg.algorithm(uda)
        algo.extract(pg.nlopt).ftol_rel = 1e-8

        # print(algo)
        prob = pg.problem(FieldNull(self.grid_boundary, self.interpolate))
        pop = pg.population(prob, size=1)
        pop = algo.evolve(pop)
        # print(pop)
        return pop.champion_x, pop.champion_f

    def get_Xpoint(self, xo):
        """
        Return X-point coordinates.

        Resolve X-point location based on solution of field minimum in
        proximity to sead location, *xo*.

        Parameters
        ----------
        xo : array-like(float), shape(2,)
            Sead coordinates (x, z).

        Raises
        ------
        TopologyError
            Field minimization failure.

        Returns
        -------
        Xpoint: array-like(float), shape(2,)
            X-point coordinates (x, z).

        """

        """
        opt = nlopt.opt(nlopt.G_MLSL_LDS, 2)
        local = nlopt.opt(nlopt.LD_MMA, 2)
        '''
        local.set_ftol_rel(1e-4)
        local.set_min_objective(self._field_null)
        local.set_lower_bounds([4, -4])
        local.set_upper_bounds([8, 4])
        '''

        opt.set_local_optimizer(local)
        opt.set_min_objective(self._field_null)
        opt.set_ftol_rel(1e-4)
        opt.set_maxeval(50)
        # grid limits
        opt.set_lower_bounds([4, -4])
        opt.set_upper_bounds([8, 4])

        opt.set_population(2)

        x = opt.optimize(xo)

        print(opt)

        #print(self.grid_boundary[1::2])
        #print(x)
        """
        opt = nlopt.opt(nlopt.LD_MMA, 2)
        opt.set_min_objective(self._field_null)
        opt.set_ftol_rel(1e-6)
        opt.set_lower_bounds(self.grid_boundary[::2])
        opt.set_upper_bounds(self.grid_boundary[1::2])
        x = opt.optimize(xo)
        # print(self.interpolate('B').ev(*x))

        """
        res = scipy.optimize.minimize(
            self._field_null, xo, jac=self._field_gradient,
            # bounds=self.bounds,
            )
        if not res.success:
            raise TopologyError('Xpoint signed |B| minimization failure\n\n'
                                f'{res}.')
        return res.x
        """
        return x
