import numpy as np
from amigo.pyplot import plt
from scipy.interpolate import interp1d
from skimage import measure
from amigo.pyplot import cntr


if __name__ == "__main__":
    # Construct some test data
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 200)
    xm, ym = np.meshgrid(x, y, indexing="ij")
    r = np.sin(np.exp((np.sin(xm) ** 3 + np.cos(ym) ** 2)))
    level = 0.5

    cfield = cntr(xm, ym, r)

    levels = cfield.trace(0.5)

    cfield.plot(level)
