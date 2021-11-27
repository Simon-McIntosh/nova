#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 21:06:13 2021

@author: mcintos
"""

import numdifftools as nd
import numpy as np

from scipy.optimize import newton_krylov
from scipy.optimize.nonlin import InverseJacobian

def L2norm(x):
    """Return L2 norm of x."""
    return np.array([x[0]+2, x[1]**2, x[2]])

x = [200, -2.5, 30]
#x = [-1.70997595, 0.00195313, -10.00390626]

jac = nd.Jacobian(L2norm, method='forward', order=1)(x)

print(jac)
print(np.linalg.inv(jac))

sol = newton_krylov(L2norm, x, verbose=True)
print('\n***\n')
sol = newton_krylov(L2norm, x, inner_M=jac, verbose=True)
print(sol, L2norm(sol))
