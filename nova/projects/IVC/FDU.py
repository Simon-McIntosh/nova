from nep.DINA.capacitor_discharge import power_supply
import numpy as np
from amigo.pyplot import plt
from amigo.geom import grid
import nova.cross_coil as cc
from nova.streamfunction import SF
from nep.DINA.read_eqdsk import read_eqdsk


class FDU:
    def __init__(self, vessel=False, invessel=True):
        self.ps = power_supply(
            nturn=4,
            vessel=vessel,
            scenario=-1,
            code="Nova",
            Ip_scale=15 / 15,
            read_txt=False,
            vessel_model="full",
            Io=0,
            sign=-1,
            t_pulse=1.8,
            origin="start",
            impulse=False,
            invessel=invessel,
        )
        self.grid_sf()
        self.load_first_wall()

    def grid_sf(self, n=5e3, limit=[2.5, 10.5, -7, 7.5]):
        self.x2d, self.z2d, self.x, self.z = grid(n, limit)[:4]

    def load_first_wall(self):
        eqdsk = read_eqdsk(file="burn").eqdsk
        self.xlim, self.zlim = eqdsk["xlim"], eqdsk["zlim"]

    def set_vs3_current(self, Ivs3):
        Ivs3 = float(Ivs3)  # store Ivs3 current
        Ic = {}
        coil_list = list(self.ps.ind.pf.coilset["coil"].keys())[:8]
        for name in coil_list:
            Ic[name] = -Ivs3 if "upper" in name else Ivs3
        self.ps.ind.pf.update_current(Ic)

    def current_update(self, t):  # update vs3 coil and structure
        self.Ivec = self.Ivec_fun(t)  # current vector
        self.set_vs3_current(self.Ivec[0])  # vs3 coil current
        coil_list = list(self.ps.ind.pf.coilset["coil"].keys())
        Ic = {}  # coil jacket
        for i, coil in enumerate(coil_list[8:12]):
            Ic[coil] = self.Ivec[1]  # lower VS jacket
        for i, coil in enumerate(coil_list[12:16]):
            Ic[coil] = self.Ivec[2]  # upper VS jacket
        self.ps.ind.pf.update_current(Ic)  # dissable to remove jacket field
        if self.ps.vessel or self.ps.pfcs:  # vv and trs currents
            Ic = {}  # vv and trs
            for i, coil in enumerate(coil_list[16:]):
                Ic[coil] = self.Ivec[i + 3]
            self.ps.ind.pf.update_current(Ic)  # dissable to remove vv field

    def solve(self, plot=False, **kwargs):
        self.Ivec_fun = self.ps.solve(**kwargs)  # solve power supply
        if plot:
            self.plot_contour()

    @staticmethod
    def plot_field(data, tmax=None, ax=None, label=None):
        if ax is None:
            ax = plt.subplots(2, 1, sharex=True)[1]
        Bx_max = np.max(abs(data["B"][:, 0]))
        if tmax:
            idx = np.argmin(abs(data["time"] - tmax))
        else:
            idx = len(data["B"])
        t10 = data["time"][np.argmin(abs(abs(data["B"][:idx, 0]) - 0.1 * Bx_max))]
        t90 = data["time"][np.argmin(abs(abs(data["B"][:idx, 0]) - 0.9 * Bx_max))]
        risetime = t90 - t10
        if label:
            label += r" $\tau$"
            label += f"={1e3*risetime:1.0f}ms"
        ax[0].plot(1e3 * data["time"], 1e-3 * data["Ivec"][:])
        ax[1].plot(1e3 * data["time"], 1e3 * data["B"][:, 0], label=label)
        ax[0].set_ylabel("$I_vs3$ kA")
        ax[1].set_ylabel("$B_x$ mT")
        plt.legend()
        plt.despine()
        plt.detick(ax)

    def update_sf(self, t):
        self.current_update(t)  # update VS3 and vessel currents
        psi = cc.get_coil_psi(
            self.x2d,
            self.z2d,
            self.ps.ind.pf.coilset["subcoil"],
            self.ps.ind.pf.coilset["plasma"],
        )
        eqdsk = {
            "x": self.x,
            "z": self.z,
            "psi": psi,
            "fw_limit": False,
            "xlim": self.xlim,
            "zlim": self.zlim,
        }
        self.sf = SF(eqdsk=eqdsk)

    def plot_contour(self, **kwargs):
        self.update_sf(1.8)
        levels = self.sf.get_levels(Nlevel=81, Nstd=4)
        ax = plt.subplots(2, 3, figsize=(10, 9))[1]
        ax = ax.flatten()
        for ax_, t in zip(ax, [0.1, 0.5, 1, 2.5, 5, 18]):
            self.update_sf(t)
            self.sf.contour(ax=ax_, levels=levels, boundary=False, Xnorm=False)
            self.sf.plot_firstwall(ax=ax_)
            self.ps.ind.plot(ax=ax_)
            ax_.set_title(f"t {t:1.0f}ms")


if __name__ == "__main__":
    fdu = FDU(vessel=False, invessel=True)
    fdu.solve(t_end=0.2, plot=True)
