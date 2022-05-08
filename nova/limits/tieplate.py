import numpy as np

def get_tie_plate(wp='Soft', link=False):
    tie_plate = {'name': wp}
    tie_plate['preload'] = 201.33
    tie_plate['limit_load'] = -26  # minimum axial load
    tie_plate['mg'] = 1.18  # coil weight MN

    if wp == 'Ansys_old':
        alpha = -0.0071#6.85e-3
        beta = np.array([0.0165, 0.0489, 0.0812,
                         0.113, 0.145, 0.178])
        gamma = 2.95e-2
    elif wp == 'Stiff':
        alpha = -0.0025
        beta = np.array([0.0254, 0.0752, 0.1249,
                         0.1737, 0.2235, 0.2733])
        gamma = 0.0455
    elif wp == 'CSM1':
        alpha = -9.9e-3
        beta = np.array([0.0229, 0.0681, 0.113,
                         0.158, 0.203, 0.248])
        gamma = 4.27e-2
    elif wp == 'FEniCs':
        alpha = -9.37e-3
        beta = np.array([0.0218, 0.0647, 0.108,
                         0.15, 0.193, 0.236])
        gamma = 4.27e-2
    elif wp == 'Soft':
        alpha = -0.0018
        beta = np.array([0.0373, 0.1111, 0.1849,
                         0.2580, 0.3318, 0.4056])
        gamma = 0.0704
        tie_plate['preload'] = 207.8
    elif wp == 'Built':
        alpha = -0.0019
        beta = np.array([0.0389, 0.1161, 0.1933,
                         0.2696, 0.3468, 0.4239])
        gamma = 0.0739
        tie_plate['preload'] = 190

    if link:  # link central pair of CS modules
        beta = np.append(np.append(beta[:2], np.sum(beta[2:4])), beta[-2:])

    tie_plate['alpha'] = alpha  # alpha Fx (Poisson)
    tie_plate['beta'] = beta  # beta Fz
    tie_plate['gamma'] = gamma  # gamma Fc (crush)
    return tie_plate
