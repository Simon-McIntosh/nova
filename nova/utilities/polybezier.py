import numpy as np
from scipy.special import binom as bn
from amigo.pyplot import plt
import copy


class polybezier:
    def __init__(self, npoints):
        self.clear(npoints)  # set default point spacing

    def clear(self, *args):
        try:
            self.npoints = args[0]  # points per unit
        except IndexError:
            pass  # retain default
        self.nodes = {"x": [], "z": [], "t": [], "L": []}
        self.po = []  # control points
        self.p = []  # shape

    def construct(self, x, z, theta=None, length=None, local=True):
        p0 = {"x": x[0], "z": z[0]}
        p3 = {"x": x[1], "z": z[1]}
        if np.shape(local) == ():
            local = local * np.ones(2, dtype=bool)
        if theta is not None:
            if np.shape(theta) == ():
                theta = theta * np.ones(2)
            p0["t"] = theta[0]
            p3["t"] = theta[1] + np.pi
            rotate = np.arctan2(np.diff(z), np.diff(x))
            if local[0]:
                p0["t"] += rotate
            if local[1]:
                p3["t"] += rotate
        if length is not None:
            if np.shape(length) == ():
                length = length * np.ones(2)
            p0["L"] = length[0]
            p3["L"] = length[1]
        p1, p2, dl = self.control(p0, p3)
        curve = self.segment([p0, p1, p2, p3], dl)
        self.po.append({"p0": p0, "p1": p1, "p2": p2, "p3": p3, "dl": dl})
        self.p.append(curve)

    def segment(self, p, dl):
        n = int(np.ceil(self.npoints * dl))  # segment point number
        t = np.linspace(0, 1, n)
        curve = {"x": np.zeros(n), "z": np.zeros(n)}
        for i, pi in enumerate(p):
            for var in ["x", "z"]:
                curve[var] += self.basis(t, i) * pi[var]
        return curve

    def basis(self, t, v):
        n = 3  # spline order
        return bn(n, v) * t**v * (1 - t) ** (n - v)

    @staticmethod
    def midpoints(p):  # convert polar to cartesian
        x = p["x"] + p["L"] * np.cos(p["t"])
        z = p["z"] + p["L"] * np.sin(p["t"])
        return x, z

    def control(self, p0, p3):  # control points (length and theta or midpoint)
        p1, p2 = {}, {}
        xm, ym = np.mean([p0["x"], p3["x"]]), np.mean([p0["z"], p3["z"]])
        dl = np.linalg.norm([p3["x"] - p0["x"], p3["z"] - p0["z"]])
        for p, pm in zip([p0, p3], [p1, p2]):
            if "L" not in p:  # add midpoint length
                p["L"] = dl / 2
            if "t" not in p:  # add midpoint angle
                p["t"] = np.arctan2(ym - p["y"], xm - p["x"])
            pm["x"], pm["z"] = self.midpoints(p)
        return p1, p2, dl

    def plot_control_points(self, po, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        for i, j in zip(["p0", "p2"], ["p1", "p3"]):
            ax.plot(
                [po[i]["x"], po[j]["x"]],
                [po[i]["z"], po[j]["z"]],
                "-",
                color="darkgray",
            )
        for i in ["p0", "p3"]:
            ax.plot(po[i]["x"], po[i]["z"], "o", color="gray")
        for i in ["p1", "p2"]:
            ax.plot(po[i]["x"], po[i]["z"], "s", color="gray")

    def concatanate(self, ses=[None, None, None]):
        x, z = np.array([]), np.array([])
        for p in self.p[slice(*ses)]:  # start, end, step
            x = np.append(x, p["x"])
            z = np.append(z, p["z"])
        return x, z

    def draw(self, ses=[None, None, None], plot=False, ax=None):
        self.p = []  # clear shape
        for po in self.po:
            p = [po[f"p{i}"] for i in range(4)]
            dl = po["dl"]
            curve = self.segment(p, dl)
            self.p.append(curve)
        x, z = self.concatanate(ses)  # start, end, step
        if plot:
            if ax is None:
                ax = plt.subplots(1, 1)[1]
            ax.plot(x, z, "C3--")
            for po in self.po:
                self.plot_control_points(po, ax=ax)
        return x, z

    def reverse_segment(self, po):
        p_ = copy.deepcopy(po)
        po["p0"] = p_["p3"]
        po["p1"] = p_["p2"]
        po["p2"] = p_["p1"]
        po["p3"] = p_["p0"]


if __name__ == "__main__":
    z = {"top": 4, "Xpoint": -3}

    zX = -3

    x = [-3, 0.5, 4]
    z = [0, -2, -0.2]

    L = 0.75

    pb = polybezier(21)
    pb.clear()
    pb.construct(x[:2], z[:2], [0.0, 0], L, [1, 0])
    pb.construct(x[1:], [-2, 0], [0, 0], L, [0, 1])
    # pb.construct(x[::-1][:2], [-2, 0], [0, 0.3], [0.6, 0.5], [0, 1])
    # pb.construct(x[::-1][1:], [0, -2], [0.05, 0], [0.5, 0.6], [1, 0])
    pb.draw(plot=True)
