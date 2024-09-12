# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:54:40 2024

@author: mcintos
"""

import pylops

from nova.linalg.basis import Bernstein

bernstein = Bernstein(131, 201)


operator = pylops.MatrixMult(bernstein.matrix)


y = bernstein.coordinate**2

coef = operator / y


fista = pylops.optimization.sparsity.fista(
    operator,
    y,
)[0]


bernstein.plot(coef)
bernstein.plot(fista)
