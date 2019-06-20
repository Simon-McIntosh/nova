import numpy as np
import matplotlib.pyplot as plt
from legacycontour._cntr import Cntr as cntr

from skimage import measure

fig, ax = plt.subplots()


# Construct some test data
x, y = np.mgrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
level = 0.5

cfield = cntr(x, y, r)

psi_line = cfield.trace(level, level, 0)

psi_line = psi_line[:len(psi_line) // 2]
for line in psi_line:
    ax.plot(line[:, 0], line[:, 1])
#lines.append(psi_line)


# Find contours at a constant value of 0.8
contours = measure.find_contours(r, level)

# Display the image and plot all contours found
#ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)
ax.pcolor(x, y, r)


'''
for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
'''