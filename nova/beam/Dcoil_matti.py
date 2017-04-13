# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:58:05 2016

@author: colemam
"""
import matplotlib.pyplot as mp
mp.rcParams.update({'font.size': 14})
import scipy as sc
from scipy import special
import numpy as np

R1 = 4173.5
R2 = 16020.5  # baseline
nr = 500
# def TFconstantT(R1,R2,nr):
R0 = np.sqrt(R1 * R2)
k = 0.5 * np.log(R2 / R1)
theta = np.linspace(np.pi, 4 * np.pi, 100)
dSUM = np.ones(len(theta))
nnn = np.linspace(1, nr, nr)
SUM = np.zeros([len(theta), len(nnn)])
z = np.zeros(len(theta))
r = np.zeros(len(theta))
for m in range(len(theta)):
    while dSUM[m] >= 0.001:
        for n in range(len(nnn)):
            SUM[m, n] = (1j / nnn[n]) * (np.exp(-1j * nnn[n] * theta[m]) - 1) * (1 + np.exp(1j * nnn[n] * (theta[m] + np.pi))) * (
                np.exp(1j * nnn[n] * np.pi / (2))) * 0.5 * (special.iv(nnn[n] - 1, k) + special.iv(nnn[n] + 1, k)) + SUM[m, n - 1]
            dSUM[m] = abs(SUM[m, n] - SUM[m, n - 1])
    z[m] = R0 * k * (theta[m] * special.iv(1, k) + SUM[m, n])
    r[m] = R0 * np.exp(k * np.sin(theta[m]))
    print(round(m / len(theta), 2))

f, ax = mp.subplots()
mp.plot(r / 1000, z / 1000)
mp.xlabel('$r$ [m]')
mp.ylabel('$z$ [m]')
mp.ylim([10, 35])
mp.xlim([0, 17])
mp.gca().set_aspect('equal', adjustable='box')
mp.locator_params(nbins=4)
mp.text(
    18, 20, r'$z=r_{0}k\left(I_{1}(k)\theta+ \sum_{n=1}^\infty\frac{i}{n}[e^{-in\theta}-1][1+e^{in(\theta+\pi)}]e^{in\frac{\pi}{2}}\frac{I_{n-1}(k)+I_{n+1}(k)}{2} \right) $', fontsize=14)
mp.text(18, 15, r'$r=r_{0}e^{k\sin{\theta}}$', fontsize=14)
mp.text(18, 30, r'$k=\frac{1}{2}\ln{\frac{r_2}{r_1}}=%s$' %
        round(k, 3), fontsize=14)
mp.text(18, 25, r'$r_{0}=\sqrt{r_1r_2}$')
# return [z,r,SUM,dSUM]
