import numpy as np


def get_tensor(*args):
    s = catalogue_stress(*args)
    tensor = np.array(
        [
            [s["xx"], s["xy"], s["xz"]],
            [s["xy"], s["yy"], s["yz"]],
            [s["xz"], s["yz"], s["zz"]],
        ]
    )
    return tensor


def catalogue_stress(*args):
    narg = len(args)
    if narg == 1:
        s = args[0]
    elif narg == 3:  # principal stress
        s = {}
        for i, var in enumerate(["xx", "yy", "zz"]):
            s[var] = args[i]
        for i, var in enumerate(["xy", "yz", "xz"]):
            s[var] = 0 * s["xx"]
    elif narg == 6:
        s = {}
        for i, var in enumerate(["xx", "yy", "zz", "xy", "yz", "xz"]):
            s[var] = args[i]
    else:
        raise IndexError("args required as dict or 6 stress components")
    return s


def vonMises(*args):  # calculate von Mises stress
    s = catalogue_stress(*args)
    vm = np.sqrt(
        0.5
        * (
            (s["xx"] - s["yy"]) ** 2
            + (s["yy"] - s["zz"]) ** 2
            + (s["xx"] - s["zz"]) ** 2
            + 6 * (s["xy"] ** 2 + s["yz"] ** 2 + s["xz"] ** 2)
        )
    )
    return vm


def principal(*args):  # calculate principal stress components and vectors
    tensor = get_tensor(*args)  # calculate Tresca stress
    w = np.linalg.eigvals(tensor.T)  # eigenvalues
    w = w[np.arange(w.shape[0])[:, None], np.argsort(w, axis=1)]  # sort
    w = np.flip(w, axis=1)  # decending absolutes
    return w


def Tresca(*args):
    w = principal(*args)  # ordered principals
    delta = np.zeros(np.shape(w))
    for i in range(3):
        j = (i + 1) % 3
        delta[:, i] = w[:, i] - w[:, j]
    index = np.argmax(abs(delta), axis=1)
    max_delta = delta[range(len(delta)), index]  # signed maximum absolutes
    R = np.zeros(len(w))
    # R = w[range(len(delta)), index] / w[range(len(delta)), (index + 1) % 3]
    R[abs(R) > 1] = 1 / R[abs(R) > 1]  # ensure abs(R) < 1
    return max_delta, index, R, w
