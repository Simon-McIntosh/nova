import numpy as np
from amigo.pyplot import plt

E = 190e9
A = 0.001
rho = 1e3
g = -9.81

L = 3

w = g * rho * A  # weight per length


def extension(x):
    u = w / (E * A) * (L * x - x**2 / 2)
    return u


x = np.linspace(0, L, 100)
plt.plot(x, extension(x))
