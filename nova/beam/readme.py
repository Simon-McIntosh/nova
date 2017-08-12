#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 07:31:37 2017

@author: Simon
"""

'''
notes for stress evaluation
beam elemeny has three major components

[k][u] = [F]

# stiffness components
u -- AE/l
v -- 3EIzz/2a**3
w -- 3EIyy/2a**3
tx -- GJ/l
ty -- EIyy/a
tz -- EIzz/a

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
sx = d2v E z + d2w E y

svm = sqrt()


'''