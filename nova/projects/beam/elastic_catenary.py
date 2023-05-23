import numpy as np
from scipy.optimize import brentq, minimize

from nova.structural.finiteframe import finiteframe
from nova.utilities import geom
import matplotlib.pyplot as plt


class catenary(finiteframe):
    
    g = 9.81

    def __init__(self, N=50):
        super().__init__(frame='3D')
        self.N = N
        self.add_shape('circ', r=0.02, ro=0.01)
        self.add_mat('bar', ['steel_cast'], [self.section])

    def extract_section(self):
        A = self.mat[0]['mat_o']['A']
        rho = self.mat[0]['mat_o']['rho']
        E = self.mat[0]['mat_o']['E']
        return A, E, rho

    def balance(self, H, V, A, E, L, Lo, rho):
        lhs = V / (H + np.sqrt(V**2 + H**2))
        rhs = (A - H / E) / np.sqrt(A**2 - (H / E)**2) *\
            np.tanh(rho * self.g * L / (2 * H) * np.sqrt(A**2 - (H / E)**2))
        return lhs - rhs

    def react(self, L, Lo, eps=1e-9):  # vertical+horizontal reactions
        A, E, rho = self.extract_section()
        V = rho * self.g * A * Lo  # vertical reaction
        H = brentq(lambda H: self.balance(H, V, A, E, L, Lo, rho),
                   eps, (1 - eps) * A * E, xtol=eps)  # horizontal reaction
        alpha = (A - H / E) / np.sqrt(A**2 - (H / E)**2)
        beta = rho * self.g / (2 * H) * np.sqrt(A**2 - (H / E)**2)
        return alpha, beta, V, H

    def curve(self, a, x):
        y = a * np.cosh(x / a)
        y -= a
        return y

    def diffrence(self, a, x, y):
        yo = self.curve(a, x)
        err = np.sum((y - yo)**2)
        return err

    def match_inelastic(self, x, y, plot=False):
        res = minimize(lambda a: self.diffrence(a, x, y), x0=0.5)
        a = res.x
        yo = self.curve(a, x)
        if plot:
            plt.plot(x, y, '-')
            plt.plot(x, yo, '--')
            plt.axis('equal')
        return yo, a

    def theory(self, L, Lo, plot=False):
        x, yo, a = self.inelastic(L, Lo)
        A, E, rho = self.extract_section()
        w = A * rho  # weight per length
        To = w * self.g * a  # horizontal tension
        dydx = np.sinh(x / a)  # gradient of curve
        psi = np.arctan2(dydx, 1)
        T = To / np.cos(psi)  # curve tension
        s = T / A  # axial stress
        L = geom.length(x, yo, norm=False)
        if plot:
            plt.plot(L, 1e-6*s, 'C3--')  # MPa
        return T

    def elastic(self, L, Lo, plot=False):
        alpha, beta = self.react(L, Lo)[:2]
        x = np.linspace(0, L, int(self.N/2))
        y = alpha / (beta * (1 - alpha**2)) * \
            (np.log(1 - alpha**2 * np.tanh(beta * x)**2) -
             2 * np.log(1 / np.cosh(beta * x)))
        x = np.append(-x[::-1][:-1], x)
        y = np.append(y[::-1][:-1], y)
        if plot:
            plt.figure()
            plt.plot(x, y)
            plt.axis('equal')
        return x, y

    def inelastic(self, L, Lo, plot=False):
        x, y = self.elastic(L, Lo, plot=plot)
        yo, a = self.match_inelastic(x, y)
        return x, yo, a

    def solve(self, shape, L, Lo, **kwargs):
        if shape == 'elastic':
            x, y = self.elastic(L, Lo)
        elif shape== 'inelastic':
            x, y = self.inelastic(L, Lo)[:2]
        elif shape == 'beam':
            x = np.linspace(0, L, self.N)
            y = np.zeros(self.N)
        elif shape == 'xy':
            x, y = kwargs['x'], kwargs['y']
        self.clfe()  # clear all
        X = np.zeros((len(x), 3))
        X[:, 0], X[:, 1] = x, y
        self.add_nodes(X)
        self.add_elements(part_name='chain', nmat='bar')
        self.add_bc('nrz', [0], part='chain', ends=0)
        self.add_bc('nrz', [-1], part='chain', ends=1)
        self.add_weight(g=[0, -1, 0])  # add weight to all elements
        
        # add tension
        if shape != 'beam':
            T = self.theory(L, Lo)
            self.update_rotation()  # check / update rotation matrix
            for i, el in enumerate(self.part['chain']['el']):
                tension = (T[i] + T[i+1]) / 2
                for nd, sign in zip(self.el['n'][el], [1, -1]):
                    f = np.array([sign * tension, 0, 0])
                    f = np.dot(self.T3[:, :, el].T, f)  # to local
                    self.add_nodal_load(nd, 'fx', f[0])
                    self.add_nodal_load(nd, 'fy', f[1])
            
        finiteframe.solve(self)
        self.plot(scale_factor=-0.2, projection='xy')


if __name__ == '__main__':

    L, Lo = 1, 1.5
    cat = catenary(N=51)
    
    #ax = plt.subplots(1, 1)[1]
    cat.solve('beam', L, Lo)
    
    #x, y = cat.X[0]-cat.D['x'], cat.X[0]-cat.D['x']
    #
    
    #cat.plot(scale_factor=1, projection='xy')
    
    cat.plot_moment()