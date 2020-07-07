# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:27:36 2020

@author: mcintos
"""

from simulation data
def update_interaction(self, coil_index=None, **kwargs):
    self.generate_grid(**kwargs)  # add | append data targets
    self.add_targets(**kwargs)  # re-generate grid on demand
    if coil_index is not None:  # full update
        self.grid['update'] = True and self.grid['n'] > 0
        self.target['update'] = True
        self.target['targets']['update'] = True
    update_targets = self.grid['update'] or self.target['update']
    if update_targets or coil_index is not None:
        if coil_index is None:
            coilset = self.coilset  # full coilset
        else:
            coilset = self.subset(coil_index)  # extract subset
        bs = biot_savart(source=coilset, mutual=False)  # load coilset
        if self.grid['update'] and self.grid['n'] > 0:
            bs.load_target(self.grid['x2d'].flatten(),
                           self.grid['z2d'].flatten(),
                           label='G', delim='', part='grid')
            self.grid['update'] = False  # reset update status
        if self.target['update']:
            update = self.target['targets']['update']  # new points only
            targets = self.target['targets'].loc[update, :]  # subset
            bs.load_target(targets['x'], targets['z'], name=targets.index,
                           part='target')
            self.target['targets'].loc[update, 'update'] = False
            self.target['update'] = False
        M = bs.calculate_interaction()
        for matrix in M:
            if self.interaction[matrix].empty:
                self.interaction[matrix] = M[matrix]
            elif coil_index is None:
                drop = self.interaction[matrix].index.unique(level=1)
                for part in M[matrix].index.unique(level=1):
                    if part in drop:  # clear prior to concat
                        if part == 'target':
                            self.interaction[matrix].drop(
                                    points.index, level=0,
                                    inplace=True, errors='ignore')
                        else:
                            self.interaction[matrix].drop(
                                    part, level=1, inplace=True)
                self.interaction[matrix] = concat(
                        [self.interaction[matrix], M[matrix]])
            else:  # selective coil_index overwrite
                for name in coilset.coil.index:
                    self.interaction[matrix].loc[:, name] = \
                        M[matrix].loc[:, name]
                        
def solve_interaction(self, plot=False, color='gray', *args, **kwargs):
    'generate grid / target interaction matrices'
    self.update_interaction(**kwargs)  # update on demand
    for matrix in self.interaction:  # Psi, Bx, Bz
        if not self.interaction[matrix].empty:
            # variable = matrix.lower()
            #index = self.interaction[matrix].index
            #value = np.dot(
            #        self.interaction[matrix].loc[:, self.coil.data.index],
            #        self.coil.data.Ic)
            #value = self.interaction[matrix].dot(self.Ic)
            value = np.dot(self.interaction[matrix].to_numpy(), self.Ic)
            #coil = DataFrame(value, index=index)  # grid, target
            '''
            for part in coil.index.unique(level=1):
                part_data = coil.xs(part, level=1)
                part_dict = getattr(self, part)
                if 'n2d' in part_dict:  # reshape data to n2d
                    part_data = part_data.to_numpy()
                    part_data = part_data.reshape(part_dict['n2d'])
                    part_dict[matrix] = part_data
                else:
                    part_data = concat(
                            (Series({'t': self.t}), part_data),
                            sort=False)
                    part_dict[matrix] = concat(
                            (part_dict[matrix], part_data.T),
                            ignore_index=True, sort=False)
            '''
    if plot and self.grid['n'] > 0:
        if self.grid['levels'] is None:
            levels = self.grid['nlevels']
        else:
            levels = self.grid['levels']
        QuadContourSet = plt.contour(
                self.grid['x2d'], self.grid['z2d'], self.grid['Psi'],
                levels, colors=color, linestyles='-', linewidths=1.0,
                alpha=0.5, zorder=5)
        self.grid['levels'] = QuadContourSet.levels
        plt.axis('equal')
        #plt.quiver(self.grid['x2d'], self.grid['z2d'], 
        #           self.grid['Bx'], self.grid['Bz'])
    
    
    '''
    def index_part(self, M):
        M.loc[:, 'part'] = self.target.coil['part']
        M.set_index('part', append=True, inplace=True)
        return M
    
    def column_reduce(self, Mo):
        Mo = pd.DataFrame(Mo, index=self.target.subcoil.index,
                          columns=self.source.subcoil.index, dtype=float)
        Mcol = pd.DataFrame(index=self.target.subcoil.index,
                            columns=self.source.coil.index, dtype=float)
        for name in self.source.coil.index:  # column reduction
            index = self.source.coil.subindex[name]
            Mcol.loc[:, name] = Mo.loc[:, index].sum(axis=1)
        return Mcol

    def row_reduce(self, Mcol):
        Mrow = pd.DataFrame(columns=self.source.coil.index, dtype=float)
        if 'subindex' in self.target.coil.columns:
            #part = self.target.coil['part']
            for name in self.target.coil.index:  # row reduction
                index = self.target.coil.subindex[name]
                Mrow.loc[name, :] = Mcol.loc[index, :].sum(axis=0)
        else:
            #part = self.target.subcoil['part']
            Mrow = Mcol
        #Mrow['part'] = part
        #Mrow.set_index('part', append=True, inplace=True)
        return Mrow

    def calculate_inductance(self):
        self.colocate()  # set targets
        Mc = self.row_reduce(self.flux_matrix())  # line-current
        return Mc

    def calculate_interaction(self):
        self.assemble(offset=True)  # build interaction matrices
        M = {}
        M['Psi'] = self.flux_matrix()  # line-current interaction
        return M
    '''


class self_inductance:
    '''
    self-inductance methods for a single turn circular coil
    '''
    def __init__(self, x, cross_section='circle'):
        self.x = x  # coil major radius
        self.cross_section = cross_section  # coil cross_section
        self.cross_section_factor = \
            geometric_mean_radius.gmr_factor[self.cross_section]

    def minor_radius(self, L, bounds=(0, 1)):
        '''
        inverse method, solve coil minor radius for given inductance

        Attributes:
            L (float): target inductance Wb
            bounds (tuple of floats): bounds fraction of major radius

        Returns:
            dr (float): coil minor radius
        '''
        self.Lo = L
        r = minimize_scalar(self.flux_err, method='bounded',
                            bounds=bounds, args=(self.Lo),
                            options={'xatol': 1e-12}).x
        gmr = self.x * r
        dr = gmr / self.cross_section_factor
        return dr

    def flux_err(self, r, *args):
        gmr = r * self.x
        L_target = args[0]
        L = self.flux(gmr)
        return (L-L_target)**2

    def flux(self, gmr):
        '''
        calculate self-induced flux though a single-turn coil

        Attributes:
            a (float): coil major radius
            gmr (float): coil cross-section geometric mean radius

        Retuns:
            L (float): self inductance of coil
        '''
        if self.x > 0:
            L = self.x * ((1 + 3 * gmr**2 / (16 * self.x**2)) *
                          np.log(8 * self.x / gmr) -
                          (2 + gmr**2 / (16 * self.x**2)))
        else:
            L = 0
        return biot_savart.mu_o * L  # Wb