import numpy as np
from itertools import combinations_with_replacement as cr
from itertools import combinations as cm
import sys
import datetime
#import nova.cross_coil as cc
from amigo import geom


def time_remaining(i, N, to, time):
    if time:
        t1 = time()
        t_fraction = 2 * ((i + 1) * (N - (i + 1) / 2)) / N**2
        t_wall = t1 - to
        t_total = t_wall / t_fraction
        t_remain = t_total - t_wall
        t_str = '\rtotal: ' + str(datetime.timedelta(
            seconds=np.round(t_total))) + '\t'
        t_str += 'complete:{:3.0f}%\t'.format(100 * t_fraction)
        t_str += 'remaining: ' + str(datetime.timedelta(
            seconds=np.round(t_remain)))
        sys.stdout.write(t_str)
        sys.stdout.flush()


class neumann(object):
    def __init__(self, **kwargs):
        if 'r' in kwargs:
            self.setr(kwargs.get('r'))
        if 'X' in kwargs:
            self.setX(kwargs.get('X'))
        if 'X_' in kwargs:
            self.setX_(kwargs.get('X_'))

    def dloop(self, X):
        dX = np.diff(np.append(X, X[0, :, None].T, axis=0), axis=0)
        return dX

    def setX(self, X):
        self.X = X
        self.dX = self.dloop(X)

    def setX_(self, X_):
        self.X_ = X_
        self.dX_ = self.dloop(X_)

    def setr(self, r):
        self.r = r

    def rotateX_(self, theta):
        self.X_ = np.dot(self.X, geom.rotate(theta))
        self.dX_ = np.dot(self.dX, geom.rotate(theta))

    def intergrate(self, ij, X, X_, dX, dX_, r):
        dr = np.linalg.norm(X[ij[0], :] - X_[ij[1], :])
        if dr > r / 2:  # mutual inductance
            L = np.dot(dX[ij[0], :], dX_[ij[1], :]) / dr
        else:  # self inductance
            L = 0
        if ij[0] != ij[1]:
            L *= 2  # upper + lower
        return L

    def substeps(self, ij):
        L = 0
        dl = np.linalg.norm(self.dX[ij[0], :])
        if dl > self.r:
            N = int(np.ceil(15 * dl / self.r))
            dx = dl / N
            X = np.zeros((N, 3))
            dX = np.zeros((N, 3))
            X[:, 0] = np.linspace(dx / 2, dl - dx / 2, N)
            dX[:, 0] = dx
            selfloop = cm(range(N), 2)
            for kl in selfloop:
                L += self.intergrate(kl, X, X, dX, dX, self.r)
        return L

    def calculate(self):
        loop = cr(range(len(self.X)), 2)
        L = 0
        if np.linalg.norm(self.X - self.X_) == 0:  # self inductance correction
            L += (np.sum(np.linalg.norm(self.dX, axis=1))) / 2
            mutual = False
        else:
            mutual = True
        for ij in loop:
            if ij[0] != ij[1] or mutual:
                L += self.intergrate(ij, self.X, self.X_,
                                     self.dX, self.dX_, self.r)
            else:
                dl = np.linalg.norm(self.dX[ij[0], :])
                if dl > self.r:
                    L += 2 * np.log(dl / self.r)
                # L += self.substeps(ij)  # self inductance sub-calc
        L *= cc.mu_o / (4 * np.pi)
        return L
