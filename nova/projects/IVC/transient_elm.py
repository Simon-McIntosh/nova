import numpy as np
from amigo.pyplot import plt
from scipy.integrate import odeint
from nep.DINA.elm_eigen_decomposition import eigen
from nep.DINA.elm_flux import elm_flux
from nep.DINA.elm_discharge_xls import elm_data


class transient_elm(eigen):
    def __init__(self, folder, code, coil):
        super().__init__(Rp=3.02e-3)  # inherit eigen
        self.load_scenario(folder)
        self.load_coil(code, coil)

    def load_coil(self, code, coil):
        self.coil = coil
        if code == "LTC":
            self.solve_eigen(1e-3 * np.array([5.1, 61.7]), [2147.5, 12617.1])
        elif code == "ENP":
            self.solve_eigen(1e-3 * np.array([15.9, 82.7]), [2134.2, 12846.9])

    def load_scenario(self, folder):
        flux = elm_flux(reverse_current=True)
        flux.load_file(folder)
        self.t = flux.t
        self.name = flux.name
        self.Vinterp = flux.Vinterp

    def dIdt_scenario(self, I, t):
        V = np.array([self.Vinterp[self.coil](t), self.Vinterp[f"{self.coil}_bg"](t)])
        # V[1] = 0#V[0]
        dI = self.M @ I + self.Linv @ V
        return dI

    def solve_scenario(self, ax=None, plot=False):
        Io = [0, 0]  # inital current
        self.Iode = odeint(self.dIdt_scenario, Io, self.t).T
        if plot:
            self.plot_scenario(ax=ax)

    def plot_scenario(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        ax.plot(self.t, 1e-3 * self.Iode[0])
        plt.despine()
        ax.set_xlabel("$t$ s")
        ax.set_ylabel("$I$ kA")


if __name__ == "__main__":
    trans = transient_elm(0, "ENP", "lower_elm")

    ax = plt.subplots(1, 1)[1]
    trans.solve_scenario(plot=True, ax=ax)

    elm_data = elm_data()
    elm_data.plot_induced("ENP", coils=["lower"], ax=ax, ic=2)
    elm_data.plot_induced("LTC", coils=["lower"], ax=ax, ic=1)
    plt.legend()
    ax.set_xlim([0, 0.4])
"""
def dIdt(self, I, t):
    dI = self.M @ I
    return dI

def solve_profile(self, t, Io):
    Iode = odeint(self.dIdt, Io, t)
    return Iode
"""
