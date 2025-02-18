"""Rotate point cloud using quaternion transform."""

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


def rotate(theta, axis="z"):
    """Rotate 3D point cloud by theta radians about axis."""
    if axis == "z":
        X = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
    elif axis == "y":
        X = np.array(
            [
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)],
            ]
        )
    elif axis == "x":
        X = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
    else:
        errtxt = "incorrect roation axis {}".format(axis)
        errtxt += ", select from ['x','y','z']"
        raise ValueError(errtxt)
    return X


def rotate2D(theta, xo=0, yo=0, dx=0, dy=0):
    """Return 2D rotation matrix about xo with dx translation."""
    X = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    To = np.array([[1, 0, -xo], [0, 1, -yo], [0, 0, 1]])
    T1 = np.array([[1, 0, xo + dx], [0, 1, yo + dy], [0, 0, 1]])
    TRT = np.dot(T1, np.dot(X, To))[:2, :]
    return X[:2, :2], TRT


def rotate_vector2D(vector, theta):
    """Rotate 2d point cloud by theta about vector."""
    Vmag = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
    X = rotate2D(theta)[0]
    Vhat = (X * np.matrix([[vector[0]], [vector[1]]])).getA1() / Vmag
    return Vhat


def qrotate(point, **kwargs):
    """Rotate point cloud by angle theta around vector.

    right-hand coordinates
    kwargs defines rotation vector using:
        pair of points, p1=[x,y,z] and p2=[x,y,z]
        origin and vector, xo=[x,y,z] and dx=[x,y,z]
        vector dx=[x,y,z], assumes xo=[0,0,0]
        quaternion quart=Quaternion object

    points as numpy.array size=(:,3) or dict with entries 'x','y','z'
    theta=float (radian)

    requires ether:
    numpy quaternion dtype:
    conda install -c moble quaternion
    https://github.com/moble/quaternion/blob/master/
    or
    pyquaternion
    pip install pyquaternion

    From quaternion readme:
    Euler angles are awful
    Euler angles are pretty much the worst things ever and it makes me feel
    bad even supporting them. Quaternions are faster, more accurate,
    basically free of singularities, more intuitive, and generally
    easier to understand.
    """
    if "quart" in kwargs:
        quart = kwargs["quart"]
        xo = kwargs.get("xo", np.zeros(3))
    else:
        theta = kwargs["theta"]
        if "p1" in kwargs and "p2" in kwargs:
            p1, p2 = kwargs["p1"], kwargs["p2"]
            if not isinstance(p1, np.ndarray) or not isinstance(p1, np.ndarray):
                p1, p2 = np.array(p1), np.array(p2)
            xo = p1
            dx = p2 - p1
            dx = tuple(dx)
        elif "xo" in kwargs and "dx" in kwargs:
            xo, dx = kwargs["xo"], kwargs["dx"]
        elif "dx" in kwargs:
            dx = kwargs["dx"]
            if isinstance(dx, str):
                index = ["x", "y", "z"].index(dx)
                dx = np.zeros(3)
                dx[index] = 1
            xo = np.zeros(3)
        else:
            errtxt = "error in kwargs input\n"
            errtxt += "rotation vector input as ether:\n"
            errtxt += "\tpair of points, p1=[x,y,z] and p2=[x,y,z]\n"
            errtxt += "\torigin and vector, xo=[x,y,z] and dx=[x,y,z]\n"
            raise ValueError(errtxt)
        dx /= np.linalg.norm(dx)  # normalise rotation axis
        quart = Quaternion(axis=dx, angle=theta)
    if isinstance(point, dict):
        isdict = True
        p = np.zeros((len(point["x"]), 3))
        for i, var in enumerate(["x", "y", "z"]):
            p[:, i] = point[var]
        point = p
    else:
        isdict = False
    if np.ndim(point) == 1 and len(point) == 3:
        point = np.array([point])
    if np.shape(point)[1] != 3:
        errtxt = "point vector required as numpy.array size=(:,3)"
        raise ValueError(errtxt)
    trans = np.ones((len(point), 1)) * xo  # expand vector origin
    p = point - trans  # translate to rotation vector's origin (xo)
    rpoint = np.zeros(np.shape(point))
    for i, po in enumerate(p):
        rpoint[i, :] = quart.rotate(po)
    rpoint += trans  # translate from rotation vector's origion (xo)
    if isdict:  # return to dict
        p = {}
        for i, var in enumerate(["x", "y", "z"]):
            p[var] = rpoint[:, i]
        rpoint = p
    return rpoint


def normalize(vector):
    """Return normalized vector."""
    return vector / np.linalg.norm(vector)

def by_angle(axis: np.ndarray, angle: float):
    """Return rotation instance that rotates a vector by angle about axis."""
    axis = normalize(axis)
    return Rotation.from_rotvec(angle * axis)

def to_vector(axis: np.ndarray, vector: np.ndarray):
    """Return rotation instance that aligns vector to axis."""
    axis = normalize(axis)
    vector = normalize(vector)
    cross = np.cross(axis, vector)
    dot = np.dot(axis, vector)
    if np.isclose(dot, -1):
        # catch -pi rotation
        axis = np.cross(axis, np.roll(axis, -1))
        return Rotation.from_rotvec(-np.pi * axis)
    v_cross = np.array(
        [
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0],
        ]
    )
    Rmat = np.identity(3) + v_cross + np.dot(v_cross, v_cross) / (1 + dot)
    return Rotation.from_matrix(Rmat)


def to_axes(axes: np.ndarray, triad: np.ndarray):
    """Return rotation instance that aligns triad to axes."""
    mat = np.linalg.solve(triad, axes)
    Rmat = triad @ mat @ triad.T
    return Rotation.from_matrix(Rmat)
