#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 07:31:37 2017

@author: Simon
"""

"""
notes for stress evaluation
beam element has three major components

[k][u] = [F]

# stiffness components
u -- AE/L
v -- 12EIzz/L**3
w -- 12EIyy/L**3
tx -- GJ/l
ty -- 2EIyy/L
tz -- 2EIzz/L

# axial
k = AE/l
F = ku
sx = ku/A = Eu/l = E epsilon

# torque
k = GJ/l
dtheata/dr = twist = tx/l
syz = Gr dtheta/dz = Gr tx/l

# bending  # interpolate from hermite polynomials
d2v -- v, ty
d2w -- w, tz
sigma = d2u E y
sx = d2w E z + d2v E y

svm = sqrt()


"""
