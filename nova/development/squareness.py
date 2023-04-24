
import matplotlib.pyplot as plt
import numpy as np

u = np.linspace(0, 1)

elongation = 1.8
minor_radius = 2

x = lambda u: minor_radius * (1 - u**2) / (1 + u**2)
y = lambda u: elongation * minor_radius * 2*u / (1 + u**2)

_x = np.linspace(0, minor_radius)
_bisect = np.arctan(elongation)
plt.plot(_x, _x*np.tan(_bisect), '--', color='gray')

_u = np.sqrt(8)/2 - 1

plt.plot(x(u), y(u))
plt.plot(x(_u), y(_u), 'o')
plt.axis('equal')
