"""
Ben Dudson - https://github.com/bendudson/freegs

Read and write 'G' formatted equilibria. This is an R-Z free boundary
format.

Format of G-EQDSK file is specified here:
  https://fusion.gat.com/THEORY/efit/g_eqdsk.html
"""

import numpy as np


def file_tokens(fp):
    """A generator to split a file into tokens"""
    toklist = []
    while True:
        line = fp.readline()
        if not line:
            yield "0"
        toklist = line.split()
        for tok in toklist:
            yield tok


def file_numbers(fp):
    """Generator to get numbers from a text file"""
    toklist = []
    while True:
        line = fp.readline()
        if not line:
            break
        if "E" in line:
            line = line.replace("E-", "*")
            line = line.replace("-", " -")  # add white space for split
            line = line.replace("*", "E-")
        toklist = line.split()
        for tok in toklist:
            yield tok


def read(f):
    """Reads a G-EQDSK file

    Parameters
    ----------

    f = Input file. Can either be a file-like object,
        or a string. If a string, then treated as a file name
        and opened.

    Returns
    -------

    """
    if isinstance(f, str):
        # If the input is a string, treat as file name
        with open(f) as fh:  # Ensure file is closed
            return read(fh)  # Call again with file object

    # Read the first line, which should contain the mesh sizes
    desc = f.readline()
    if not desc:
        raise IOError("Cannot read from input file")

    s = desc.split()  # Split by whitespace
    if len(s) < 3:
        raise IOError("First line must contain at least 3 numbers")
    name = s[0]
    header = desc  # file header
    sint = []  # extract intergers from first line
    for s_ in s:
        try:
            sint.append(int(s_))
        except:
            pass
    int(sint[-3])
    nxefit = int(sint[-2])
    nzefit = int(sint[-1])

    # Use a generator to read numbers
    token = file_numbers(f)

    xdim = float(next(token))
    zdim = float(next(token))
    xcentr = float(next(token))
    xgrid1 = float(next(token))
    zmid = float(next(token))

    xmagx = float(next(token))
    zmagx = float(next(token))
    simagx = float(next(token))
    sibdry = float(next(token))
    bcentr = float(next(token))

    Ip = float(next(token))
    simagx = float(next(token))
    float(next(token))
    xmagx = float(next(token))
    float(next(token))

    zmagx = float(next(token))
    float(next(token))
    sibdry = float(next(token))
    float(next(token))
    float(next(token))

    # Read arrays
    def read_array(n, name="Unknown"):
        data = np.zeros([n])
        try:
            for i in np.arange(n):
                data[i] = float(next(token))
        except:
            raise IOError("Failed reading array '" + name + "' of size ", n)
        return data

    def read_2d(nx, ny, name="Unknown"):
        data = np.zeros([ny, nx])
        for i in np.arange(ny):
            data[i, :] = read_array(nx, name + "[" + str(i) + "]")
        data = np.transpose(data)
        return data

    fpol = read_array(nxefit, "fpol")
    pres = read_array(nxefit, "pressure")
    ffprim = read_array(nxefit, "ffprim")
    pprime = read_array(nxefit, "pprime")
    psi = read_2d(nxefit, nzefit, "psi")
    qpsi = read_array(nxefit, "qpsi")

    # Read boundary and limiters, if present
    nbdry = int(next(token))
    nlim = int(next(token))
    if nbdry > 0:
        xbdry = np.zeros([nbdry])
        zbdry = np.zeros([nbdry])
        for i in range(nbdry):
            xbdry[i] = float(next(token))
            zbdry[i] = float(next(token))
    else:
        xbdry = [0]
        zbdry = [0]

    if nlim > 0:
        xlim = np.zeros([nlim])
        zlim = np.zeros([nlim])
        for i in range(nlim):
            xlim[i] = float(next(token))
            zlim[i] = float(next(token))
    else:
        xlim = [0]
        zlim = [0]

    # Read coil data
    try:
        ncoil = int(next(token))
    except:
        ncoil, xc, zc, dxc, dzc, It = 0, 0, 0, 0, 0, 0

    if ncoil > 0:
        xc = np.zeros(ncoil)
        zc = np.zeros(ncoil)
        dxc = np.zeros(ncoil)
        dzc = np.zeros(ncoil)
        It = np.zeros(ncoil)
        for i in range(ncoil):
            xc[i] = float(next(token))
            zc[i] = float(next(token))
            dxc[i] = float(next(token))
            dzc[i] = float(next(token))
            It[i] = float(next(token))
    else:
        xc, zc, dxc, dzc, It = 0, 0, 0, 0, 0
    # Construct X-Z mesh
    x = np.zeros(nxefit)
    z = np.zeros(nzefit)
    for i in range(nxefit):
        x[i] = xgrid1 + xdim * i / float(nxefit - 1)
    for j in range(nzefit):
        z[j] = (zmid - 0.5 * zdim) + zdim * j / float(nzefit - 1)

    # Create dictionary of values to return
    result = {
        "name": name,
        "header": header,  # first line of eqdsk file
        # Number of horizontal and vertical points
        "nx": nxefit,
        "nz": nzefit,
        "x": x,
        "z": z,  # Location of the grid-points
        "xdim": xdim,
        "zdim": zdim,  # Size of the domain in meters
        # Reference vacuum toroidal field (m, T)
        "xcentr": xcentr,
        "bcentr": bcentr,
        "xgrid1": xgrid1,  # R of left side of domain
        "zmid": zmid,  # Z at the middle of the domain
        "xmagx": xmagx,
        "zmagx": zmagx,  # Location of magnetic axis
        "simagx": simagx,  # Poloidal flux at the axis (Weber / rad)
        # Poloidal flux at plasma boundary (Weber / rad)
        "sibdry": sibdry,
        "Ip": Ip,  # plasma current
        "psi": psi,  # Poloidal flux in Weber/rad on grid points
        "fpol": fpol,  # Poloidal current function on uniform flux grid
        # "FF'(psi) in (mT)^2/(Weber/rad) on uniform flux grid"
        "ffprim": ffprim,
        # "P'(psi) in (nt/m2)/(Weber/rad) on uniform flux grid"
        "pprime": pprime,
        # Plasma pressure in nt/m^2 on uniform flux grid
        "pressure": pres,
        "qpsi": qpsi,  # q values on uniform flux grid
        "pnorm": np.linspace(0, 1, len(fpol)),  # uniform flux grid
        # Plasma boundary
        "nbdry": nbdry,
        "xbdry": xbdry,
        "zbdry": zbdry,
        "nlim": nlim,
        "xlim": xlim,
        "zlim": zlim,
        "ncoil": ncoil,
        "xc": xc,
        "zc": zc,
        "dxc": dxc,
        "dzc": dzc,
        "It": It,
    }  # coils
    return result


def carrage_return(f, i):
    if np.mod(i + 1, 5) == 0:
        f.write("\n")
    else:
        f.write(" ")


def write_line(f, data, var):
    for i, v in enumerate(var):
        fmat = "{:16.9f}" if v != "Ip" else "{:16.9e}"
        num = 0 if len(v) == 0 else data[v]
        f.write(fmat.format(num))
        carrage_return(f, i)


def write_array(f, val, c):
    if np.size(val) == 1:
        f.write("{:16.9e}".format(val))
        carrage_return(f, next(c))
        return
    for v in val:
        f.write("{:16.9e}".format(v))
        carrage_return(f, next(c))


def write(f, data):  # write a G-EQDSK file
    import time
    from itertools import count

    c = count(0)
    if isinstance(f, str):
        with open(f, "w") as fh:  # Ensure file is closed
            return write(fh, data)  # Call again with file object
    f.write("{:48s} ".format(data["name"] + "_" + time.strftime("%d%m%Y")))
    f.write("{:4d} {:4d} {:4d}\n".format(0, data["nx"], data["ny"]))
    write_line(f, data, ["xdim", "zdim", "xcentr", "xgrid1", "zmid"])
    write_line(f, data, ["xmagx", "zmagx", "simagx", "sibdry", "bcentr"])
    write_line(f, data, ["Ip", "simagx", "", "xmagx", ""])
    write_line(f, data, ["zmagx", "", "sibdry", "", ""])

    write_array(f, data["fpol"], c)
    write_array(f, data["pressure"], c)
    write_array(f, data["ffprim"], c)
    write_array(f, data["pprime"], c)
    write_array(f, data["psi"], c)
    write_array(f, data["qpsi"], c)

    f.write("{:5d} {:5d}\n".format(data["nbdry"], data["nlim"]))
    bdry = np.zeros(2 * data["nbdry"])
    bdry[::2], bdry[1::2] = data["xbdry"], data["zbdry"]
    write_array(f, bdry, c)
    lim = np.zeros(2 * data["nlim"])
    lim[::2], lim[1::2] = data["xlim"], data["ylim"]
    write_array(f, lim, c)

    f.write("{:5d}\n".format(data["ncoil"]))
    coil = np.zeros(5 * data["ncoil"])
    for i, v in enumerate(["xc", "zc", "dxc", "dzc", "It"]):
        coil[i::5] = data[v]
    write_array(f, coil, c)
