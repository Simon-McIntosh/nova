#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:50:50 2023

@author: mcintos
"""

import numpy as np
from scipy.optimize import BFGS, minimize, LinearConstraint

def fun(x):
    return sum(x**2)

def dfun(x):
    return 2*x

A = np.ones(2)

sol = minimize(fun, [5, 6], method='trust-constr',
               constraints=LinearConstraint(A, 1.73, 3))

print(sol)
