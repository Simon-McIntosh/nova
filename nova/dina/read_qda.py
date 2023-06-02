from read_dina import dina
from amigo.qdafile import QDAfile
from amigo.pyplot import plt
import numpy as np
from nep.coil_geom import PFgeom
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


class read_scenario:
    def __init__(self, database_folder="operations"):
        self.dina = dina(database_folder)

    def read_file(self, folder):
        filename = self.dina.locate_file("data2.qda", folder=folder)
        self.qdafile = QDAfile(filename)
        self.data = {}
        self.columns = {}
        for i, (var, nrow) in enumerate(zip(self.qdafile.headers, self.qdafile.rows)):
            var = var.decode()
            if nrow > 0:
                self.columns[var] = var.split(",")[0]
                self.data[self.columns[var]] = np.array(self.qdafile.data[i, :])
        if "time" in self.data:  # rename time field
            self.data["t"] = self.data["time"]
            self.data.pop("time")
        self.space_data()
        self.load_data()
        self.load_coils()

    def load_coils(self, plot=False):
        pf_geom = PFgeom()
        self.pf = pf_geom.pf
        if plot:
            self.pf.plot(label=True)

    def load_data(self):
        self.Icoil = {}
        for var in self.data:
            if var == "Ip":  # plasma
                self.Ip = 1e6 * self.data[var]  # MA to A
            elif "I" in var and "PSI" not in var:
                # kAturn to Aturn
                self.Icoil[var[1:].upper()] = 1e3 * self.data[var]
        # self.t = self.data['t']
        self.Icoil["VS"] = 4 * self.Icoil["VS3"]

    def space_data(self):  # generate interpolators and space timeseries
        to = np.copy(self.data["t"])
        dt = np.min(np.diff(to))
        self.tmax = np.nanmax(to)
        self.nt = int(self.tmax / dt)
        self.dt = self.tmax / (self.nt - 1)
        self.t = np.linspace(0, self.tmax, self.nt)
        self.fun = {}
        for var in self.data:  # interpolate
            self.fun[var] = interp1d(to, self.data[var])
            self.data[var] = self.fun[var](self.t)

    def plot_plasma(self, plot=True):
        I_target = 1e6 * np.round(1e-6 * np.nanmax(self.Ip))
        flat_top_index = np.zeros(2)
        flat_top_index[0] = next(
            (i for i, I in enumerate(self.Ip) if I > I_target), None
        )
        dt_window = 1
        nwindow = int(dt_window / self.dt)
        if nwindow % 2 == 0:
            nwindow += 1
        Ip_lp = savgol_filter(self.Ip, nwindow, 3, mode="mirror")  # lowpass
        dIpdt = np.gradient(Ip_lp, self.t)
        plt.figure()
        plt.plot(self.t, dIpdt)

        print(flat_top_index, nwindow)

        """
        Ip_lp = savgol_filter(self.Ip, 20001, 2, mode='mirror')  # lowpass filter
        plt.figure()
        plt.plot(self.t, 1e-6*self.Ip)
        plt.plot(self.t, 1e-6*Ip_lp)
        plt.ylim([14.9, 15.1])
        plt.despine()
        plt.xlabel('$t$ s')
        plt.ylabel('$I_p$ MA')
        """

    def plot_currents(self):
        for coil in self.Icoil:
            if "CS" in coil:
                color = "C1"
            elif "PF" in coil:
                color = "C0"
            elif coil == "VS":
                color = "C2"
            plt.plot(self.t, 1e-3 * self.Icoil[coil], color=color)
        plt.despine()
        plt.xlabel("$t$ s")
        plt.ylabel("$I$ kA.turn")


if __name__ is "__main__":
    scn = read_scenario()
    scn.read_file(folder=1)

    # scn.plot_currents()

    scn.plot_plasma()

    # plt.figure()
    # scn.load_coils(plot=True)
