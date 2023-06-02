import os

import pandas

from nova.definitions import root_dir
from nova.utilities.IO import readtxt
import matplotlib.pyplot as plt


class MultiFilament:
    def __init__(self):
        self.directory = os.path.join(root_dir, "data/Ansys")
        self.read()
        self.calc()
        self.plot()

    def read(self):
        filename = os.path.join(self.directory, "IVC_plasma_model.mac")
        filament_number = 0
        with readtxt(filename) as file:
            while True:
                try:
                    file.trim("set")
                    filament_number += 1
                except ValueError:
                    break
        self.plasma = pandas.DataFrame(
            index=range(filament_number), columns=["x", "z", "dx", "dz", "Ip"]
        )

        with readtxt(filename) as file:
            eval(file.readline().split()[2])
            file.trim("Ipl_scale", index=0, rewind=True)
            eval(file.readline().split()[-1])
            index = -1
            while True:
                try:
                    file.trim("set")
                    index += 1
                except ValueError:
                    break
                real_constant = file.readline().split()
                self.plasma.loc[index, "Ip"] = eval(real_constant[3])[0]
                self.plasma.loc[index, "dx"] = eval(real_constant[4])[0]
                self.plasma.loc[index, "dz"] = eval(real_constant[5])
        with readtxt(filename) as file:
            file.trim("n,", index=0, rewind=True)
            index = 0
            while True:
                node = file.readline().split()
                if node[0] != "n,":
                    break
                self.plasma.loc[index, "x"] = eval(node[2])[0]
                self.plasma.loc[index, "z"] = eval(node[4])
                file.skiplines(2)
                index += 1

    def calc(self):
        self.x_current = (
            sum(self.plasma.x**2 * self.plasma.Ip) / self.plasma.Ip.sum()
        ) ** 0.5
        self.z_current = sum(self.plasma.z * self.plasma.Ip) / self.plasma.Ip.sum()

    def plot(self):
        plt.axis("equal")
        plt.axis("off")
        plt.plot(self.plasma.x, self.plasma.z, "C0.")
        plt.plot(
            self.x_current,
            self.z_current,
            "C3o",
            label=f"current ({self.x_current:1.2f}, {self.z_current:1.2f})",
        )

        x_min, x_max = self.plasma.x.min(), self.plasma.x.max()
        z_min, z_max = self.plasma.z.min(), self.plasma.z.max()

        x_box = x_min + (x_max - x_min) / 2
        z_box = z_min + (z_max - z_min) / 2
        plt.plot(x_box, z_box, "ks", label=f"bbox ({x_box:1.2f}, {z_box:1.2f})")
        plt.legend()


if __name__ == "__main__":
    multi = MultiFilament()
