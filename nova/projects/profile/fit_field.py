from amigo.polybezier import polybezier
from nep.DINA.scenario import scenario
import numpy as np
from amigo.pyplot import plt
from amigo import geom
from nova.streamfunction import SF
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import copy
import nova.cross_coil as cc
from scipy.integrate import cumtrapz


class PlasmaParameters:
    def __init__(self):
        self.initalize()

    def initalise(self):
        # set default parameters
        self.Xpoint = np.array([5, -5])  # lower Xpoint position
        self.dz = 9  # plasma hight
        self.Ip = 15e6  # plasma current

        # normalized field (lower, inner-midplane, upper, outer-midplane)
        self.Bn = [0, 0.8, 0.2, 0.6]
        # control vector length
        self.L = [1, 0.25, 0.5, 0.5, 0.25, 1]
        ft = x[6:9]  # normalized control vector angle (1==pi)

        self.profile = {"inner": {"Bn": [0, 0.8, 0.2]}}
        self.field = [0, -1, -0.2, -0.7]  # lower, inner mp, upper, outer mp
        self.length = 0.2 * np.ones(6)
        # self.theta =

    # def assemble(self):  # input parameters to

    # def set_plasma


class plasmafoil:
    def __init__(self):
        self.load_coilset()
        self.set_plasma()

        self.pp = PlasmaParameters  # create bare parameters file
        self.pb = polybezier(21)  # create polybezier instance
        self.loops = {"inner": {}, "outer": {}, "loop": {}}
        self.bndry = copy.deepcopy(self.loops)
        self.profile = {}
        self.label = None

    def load_coilset(self, coilset=None):  # load coil geometory (coilset dict)
        # requires: coilset, boundary, dCoil
        if coilset is None:  # use ITER default
            scn = scenario(dCoil=0.25, setname="link_f")
            scn.load_coilset()  # load ITER coilset
        self.coilset = scn.coilset  # link
        self.boundary = scn.boundary
        self.dCoil = scn.dCoil
        scenario.load_functions(self, setname=scn.setname)

    def set_plasma(
        self, Ipl=-15e6, x=6.22, z=0.49, a=None, kappa=None, beta=None, li=None
    ):
        plasma_parameters = {
            "Ipl": Ipl,
            "xcur": x,
            "zcur": z,
            "a": a,
            "kappa": kappa,
            "beta": beta,
            "li": li,
        }
        scenario.set_plasma(self, plasma_parameters)

    def load_eqdsk(self, eqdsk):  # extract coefficents from exsiting eqdsk
        scenario.update_boundary(self, eqdsk)  # update boundary
        self.sf.update_eqdsk(eqdsk)  # load eqdisk
        self.extract_boundary()

    def extract_boundary(self, plot=False):
        x, z = self.sf.get_boundary(
            alpha=1, reverse=False, locate="zmin", boundary_cntr=False
        )
        self.loop = loop(x, z)
        self.loop.update_field(self.sf.Bspline)
        self.bndry["loop"] = self.loop.extract_boundary("loop")
        self.bndry["inner"] = self.loop.extract_boundary("inner")
        self.bndry["outer"] = self.loop.extract_boundary("outer")
        self.polymatch()  # bezier coefficients
        self.polystore("fit")
        if plot:
            self.plot_boundary()

    def plot_boundary(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
            for segment in ["inner", "outer"]:
                ax.plot(self.bndry[segment]["x"], self.bndry[segment]["z"])
            ax.axis("equal")

    @staticmethod
    def polyfit(x, *args):
        pb = args[0]  # polybezier instance
        construct = args[1]  # construction function
        bndry = args[2]  # field profile
        Bmax = args[3]  # signed maximum loop field (normalization)
        ln, Bn = construct(x, pb)
        Bn_bndry = interp1d(ln, Bn, fill_value="extrapolate")(bndry["ln"])
        err = np.sum((bndry["Bt"] - Bn_bndry * Bmax) ** 2)
        return err

    @staticmethod
    def construct(x, pb):
        fB = x[:3]  # normalized field strength (lower, midplane, upper)
        L = x[3:6]  # control vector length (lower, midplane, upper)
        ft = x[6:9]  # normalized control vector angle (1==pi)
        pb.clear()  # refresh spline instance
        pb.construct([0, 0.5], fB[:2], np.pi * ft[:2], L[:2], False)
        pb.construct([0.5, 1], fB[1:], np.pi * ft[1:], L[1:], False)
        ln, Bn = pb.concatanate()
        return ln, Bn

    def polygen(self, segment, *args):
        fB = [0, 0, 0.0]
        L = [0.5, 0.5, 0.5]
        ft = [0, 0, 0]
        xo = fB + L + ft  # sead vector
        fLf, fL, dft = 1e-3, 1.5, 0.3
        bounds = [
            (0, 1),
            (0, 1),
            (0, 1),
            (fLf * fL, fL),
            (fLf * fL, fL),
            (fLf * fL, fL),
            (-dft, dft),
            (-dft, dft),
            (-dft, dft),
        ]
        for arg in args:  # constrain variables
            bounds[arg[0]] = (arg[1], arg[1])
        args = (
            self.pb,
            self.construct,
            self.bndry[segment],
            self.bndry["loop"]["Btmax"],
        )
        x = minimize(self.polyfit, xo, args=args, method="L-BFGS-B", bounds=bounds).x
        self.polyfit(x, *args)
        print(segment, x)
        return x, copy.deepcopy(self.pb.po)

    def polylink(self, x, po):
        self.pb.clear()  # refresh
        self.pb.po += po[0]  # inner
        self.pb.reverse_segment(po[1][0])  # outer
        self.pb.reverse_segment(po[1][1])  # outer
        self.pb.po += po[1][::-1]
        self.x = x  # store list inputs

    def polymatch(self):
        x_inner, po_inner = self.polygen("inner")
        x_outer, po_outer = self.polygen("outer")  # , [8, -x_inner[8]]
        self.polylink([x_inner, x_outer], [po_inner, po_outer])

    def get_slice(self, segment):
        if segment == "inner":
            sl = [0, 2, 1]
        elif segment == "outer":
            sl = [2, 4, 1]
        return sl

    def extract(self, label, segment):
        ses = self.get_slice(segment)  # start, end, step
        ln, B = self.pb.draw(ses)
        L = self.bndry[segment]["Lln"](ln)
        self.profile[label][segment]["ln"] = ln  # store
        self.profile[label][segment]["Bn"] = B
        self.profile[label][segment]["Bt"] = self.bndry["loop"]["Btmax"] * B
        self.profile[label][segment]["L"] = L
        self.profile[label][segment]["po"] = self.pb.po[slice(*ses)]

    def link(self, label):
        for var in ["ln", "Bt", "L"]:
            self.profile[label]["loop"][var] = np.append(
                self.profile[label]["inner"][var], self.profile[label]["outer"][var]
            )
        self.profile[label]["loop"]["po"] = copy.deepcopy(self.pb.po)

    def polystore(self, label):  # extract profile from polybezier
        self.label = label
        self.profile[label] = copy.deepcopy(self.loops)
        for segment in ["inner", "outer"]:
            self.extract(label, segment)
        self.link(label)
        self.profile[label]["loop"]["Ip"] = self.get_plasma_current(label)

    def adjust_spline(self, index, node, **kwargs):
        po = self.pb.po[index]
        for key in kwargs:
            po[node][key] = kwargs[key]
        po["p1"], po["p2"] = self.pb.control(po["p0"], po["p3"])[:2]

    def get_nodes(self, location):
        if location == "inner":
            nodes = 0, 1
        elif location == "outer":
            nodes = 2, 3
        elif location == "lower":
            nodes = 3, 0
        elif location == "upper":
            nodes = 1, 2
        return nodes

    def plot_profiles(self, label="fit"):
        ax = plt.subplots(2, 1)[1]
        ax[0].plot(self.bndry["loop"]["L"], self.bndry["loop"]["Bt"])
        ax[0].plot(self.profile[label]["loop"]["L"], self.profile[label]["loop"]["Bt"])
        for segment in ["inner", "outer"]:
            ax[0].plot(
                self.profile[label][segment]["L"],
                self.profile[label][segment]["Bt"],
                "--",
                color="C3",
            )
            ax[1].plot(
                self.bndry[segment]["ln"],
                self.bndry[segment]["Bt"] / self.bndry["loop"]["Btmax"],
                "C0-",
            )
            ax[1].plot(
                self.profile[label][segment]["ln"],
                self.profile[label][segment]["Bn"],
                "--",
                color="C3",
            )
        self.pb.draw(plot=True, ax=ax[1])
        ax[0].set_xlabel("$L$ m")
        ax[0].set_ylabel("$B_t$ T")
        ax[1].set_xlabel("$L*$")
        ax[1].set_ylabel("$B^*$")
        plt.despine()
        plt.tight_layout()

    def get_plasma_current(self, label):
        # intergral to extract signed 'loop' plasma current, A
        Ip = (
            np.trapz(
                self.profile[label]["loop"]["Bt"], self.profile[label]["loop"]["L"]
            )
            / cc.mu_o
        )
        return Ip

    def knudge(self, location, variable, factor):
        nodes = self.get_nodes(location)  # inner, outer, lower, upper
        if variable == "field":
            kwargs = [{"z": factor} for __ in range(2)]
        if variable == "slope":
            theta = factor * np.pi
            if location == "upper":
                theta += np.pi
            kwargs = [{"t": theta}, {"t": -theta}]
        if variable == "control":
            kwargs = [{"L": factor} for __ in range(2)]
        self.adjust_spline(nodes[0], "p3", **kwargs[0])
        self.adjust_spline(nodes[1], "p0", **kwargs[1])


class loop:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.bounding_box()
        self.sort()
        self.close()
        self.locate()

    def bounding_box(self):
        self.bb = [np.max(self.x) - np.min(self.x), np.max(self.z) - np.min(self.z)]

    def sort(self):
        self.x, self.z = geom.clock(self.x, self.z, reverse=False)
        imin = np.argmin(self.z)
        self.x = np.append(self.x[imin:], self.x[:imin])
        self.z = np.append(self.z[imin:], self.z[:imin])
        self.imax = np.argmax(self.z)  # top of plasma

    def close(self, eps=1e-3):  # close loop
        if np.sqrt(
            (self.x[0] - self.x[-1]) ** 2 + (self.z[0] - self.z[-1]) ** 2
        ) > eps * np.max(self.bb):
            self.x = np.append(self.x, self.x[0])
            self.z = np.append(self.z, self.z[0])

    def locate(self):
        self.L = geom.length(self.x, self.z, norm=False)
        self.xt, self.zt = geom.tangent(self.x, self.z)
        self.that = np.array([self.xt, self.zt])
        self.that /= np.linalg.norm([self.xt, self.zt], axis=0)

    def update_field(self, Bspline):
        self.B = {
            "x": Bspline[0].ev(self.x, self.z),
            "z": Bspline[1].ev(self.x, self.z),
        }
        self.B["t"] = np.array(
            [
                np.dot(b, t)
                for b, t in zip(np.array([self.B["x"], self.B["z"]]).T, self.that.T)
            ]
        )

    def update_target(self, Btarget, segment):
        index = self.get_index(segment)

        self.Bt = Btarget(self.L)

    def plot_field(self):
        ax = plt.subplots(1, 1)[1]
        ax.plot(self.L, self.B)

    def get_index(self, segment):
        if segment == "loop":
            index = slice(0, None)
        elif segment == "inner":
            index = slice(0, self.imax)
        elif segment == "outer":
            index = slice(-1, self.imax - 2, -1)
        return index

    def extract_boundary(self, segment):
        bndry = {}
        index = self.get_index(segment)
        bndry["x"] = self.x[index]
        bndry["z"] = self.z[index]
        bndry["L"] = self.L[index]
        bndry["ln"] = geom.length(self.x[index], self.z[index])
        bndry["Lln"] = interp1d(bndry["ln"], bndry["L"], fill_value="extrapolate")
        bndry["Lz"] = interp1d(bndry["z"], bndry["L"], fill_value="extrapolate")
        bndry["xt"] = self.xt[index]
        bndry["zt"] = self.zt[index]
        bndry["that"] = self.that[index]
        bndry["B"] = {}
        for key in self.B:
            bndry[f"B{key}"] = self.B[key][index]
            self.set_bound(bndry, f"B{key}")
        self.set_bound(bndry, "x")
        self.set_bound(bndry, "z")
        self.set_bound(bndry, "ln")
        return bndry

    def set_bound(self, bndry, variable):
        bndry[f"{variable}bound"] = np.array([bndry[variable][0], bndry[variable][-1]])
        bndry[f"{variable}lim"] = np.array(
            [np.min(bndry[variable]), np.max(bndry[variable])]
        )
        argmax = np.argmax(abs(bndry[f"{variable}lim"]))
        bndry[f"{variable}max"] = bndry[f"{variable}lim"][argmax]  # signed max
        bndry[f"d{variable}"] = np.diff(bndry[f"{variable}lim"])[0]
        bndry[f"{variable}mid"] = np.mean(bndry[f"{variable}bound"])

    def plot(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        ax.plot(self.x, self.z)
        ax.axis("equal")


if __name__ == "__main__":
    """
    scn = scenario(read_txt=False, setname='link_f')
    scn.load_file(folder='15MA DT-DINA2017-04_v1.2', read_txt=False)
    scn.update_scenario(600)
    eqdsk = scn.update_psi(n=2e3, limit=[2.5, 11, -6.5, 6.5], plot=False)
    """

    # plf = plasmafoil()
    # plf.load_eqdsk(eqdsk)

    """
    plf.plot_profiles()
    plf.knudge('upper', 'field', 0.7)
    plf.knudge('outer', 'field', 1.2)
    plf.knudge('inner', 'field', 0.8)

    plf.knudge('upper', 'slope', 0.4)
    plf.knudge('upper', 'control', 0.4)
    plf.knudge('lower', 'slope', 0)
    plf.knudge('lower', 'control', 0.3)

    plf.polystore('a1')
    plf.plot_profiles('a1')
    """

    plf = plasmafoil()

    plf.sf.fw_limit = False
    plf.set_plasma(x=5.5, z=2.5)
    scenario.update_psi(plf, n=2e3, limit=[2.5, 8, -6.5, 6.5])

    plf.extract_boundary(plot=True)

    scenario.set_limits(plf)

    # plf.inv.colocate(psi=True, field=False, Xpoint=False, targets=False)

    plf.inv.initialize_log()
    plf.inv.get_weight()
    plf.inv.set_background()
    plf.inv.set_foreground()
    plf.inv.rhs = True

    plf.inv.solve_slsqp(0)

    scenario.update_psi(plf, n=2e3)

    plf.plot_profiles()
    # plf.sf.set_Plimit()

    scenario.plot_plasma(plf)
