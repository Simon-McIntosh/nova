import scipy.optimize as op
import numpy as np
import pylab as plt
import matplotlib
from matplotlib import colors


class linelabel(object):
    def __init__(
        self, MaxLines=100, Ndiv=10, value="1.1f", postfix="", ax="", loc="end"
    ):
        self.ylabel = np.zeros(
            (MaxLines,),
            dtype=[
                ("x", "float"),
                ("y", "float"),
                ("text", "|S80"),
                ("color", "3float"),
                ("alpha", "float"),
                ("value", "|S20"),
                ("postfix", "|S20"),
            ],
        )
        self.index = 0
        self.Ndiv = Ndiv
        self.value = value
        self.postfix = postfix
        self.ax = ax
        self.loc = loc

    def add(self, label, loc="", value="", postfix="", **kwargs):
        if not self.ax:
            ax = plt.gca()
        else:
            ax = self.ax
        if len(loc) == 0:
            loc = self.loc
        line = ax.get_lines()[-1]
        color = kwargs.get("color", line.get_color())
        alpha = kwargs.get("alpha", line.get_alpha())
        if not alpha:
            alpha = 1
        data = line.get_data()
        if np.nanmax(data[1]) == np.min(data[1]):
            loc = "end"
        if loc == "max":
            n = np.argmax(data[1])
        elif loc == "min":
            n = np.argmin(data[1])
        elif loc == "start":
            n = 0
        elif loc == "xlim":
            xlim = ax.get_xlim()
            n = np.argmin(abs(xlim[-1] - data[0]))
        else:
            n = -1
        x, y = data[0][n], data[1][n]
        x = kwargs.get("x", x)
        if isinstance(color, str):  # convert to rgb
            color = colors.hex2color(color)
        self.ylabel[self.index] = (x, y, label, color[:3], alpha, value, postfix)
        self.index += 1

    def space(self, yo):
        y = np.copy(yo)
        y[1:] = yo[0] + np.cumsum(yo[1:])
        # if y[-1] > self.ylim[1]:
        #    y = y-(y[-1]-self.ylim[1])
        # if y[0] < self.ylim[0]:
        #    y = y+(self.ylim[0]-y[0])
        return y

    def fit(self, yo, args=()):
        y = self.space(yo)
        err = np.sum((y - args) ** 2)
        return err

    def plot(self, xscale="", yscale="", Ralign=False, Roffset=0, fs=None):
        if self.index == 0:
            raise ValueError("no lines defined - use self.add() after plt.")
        if fs is None:
            fs = matplotlib.rcParams["legend.fontsize"]
        if not self.ax:
            ax = plt.gca()
        else:
            ax = self.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self.ylim = ylim

        if not xscale:
            xscale = ax.get_xscale()
        if not yscale:
            yscale = ax.get_yscale()
        self.ylabel = self.ylabel[: self.index]  # trim
        self.ylabel = np.sort(self.ylabel, order="y")
        if xscale == "log":
            self.ylabel["x"][self.ylabel["x"] == 0] = xlim[0]
        if yscale == "log":
            self.ylabel["y"][self.ylabel["y"] == 0] = ylim[0]
        yref = self.ylabel["y"]
        if xscale == "log":
            dx = 10 * xlim[1]
        else:
            dx = np.diff(xlim)
        dy = np.diff(ylim) / self.Ndiv
        if yscale == "log":
            b = np.log10(ylim[1] / ylim[0]) / (ylim[1] - ylim[0])
            a = ylim[0] / 10 ** (b * ylim[0])
            yref = np.log10(yref / a) / b
            ylim = np.log10(ylim / a) / b
            # dy = np.log10(dy/a)/b

        yo = np.zeros(len(yref))
        yo[0] = yref[0]
        yo[1:] = yref[1:] - yref[:-1]  # spacing
        bounds = [(ylim[0] + dy, None)]
        # bounds = [(None,None)]
        for i in range(len(yo) - 1):
            bounds.append((dy, None))

        yo = op.minimize(
            self.fit,
            yo,
            args=(yref),
            bounds=bounds,
            method="L-BFGS-B",
            tol=1e-6,
            options={"disp": False},
        ).x
        y = self.space(yo)

        if yscale == "log":
            y = a * 10 ** (b * y)

        for i in range(self.index):
            if Ralign:
                xmax = np.nanmax(self.ylabel["x"]) + 0.06 * dx + Roffset
            else:
                xmax = self.ylabel["x"][i] + 0.06 * dx + Roffset
            ax.plot(
                [self.ylabel["x"][i] + 0.01 * dx, xmax - 0.01 * dx],
                [self.ylabel["y"][i], y[i]],
                "-",
                color=self.ylabel["color"][i],
                linewidth=1,
                alpha=0.5,
            )  # alpha=self.ylabel['alpha'][i]
            label = self.ylabel["text"][i].decode()
            value = self.ylabel["value"][i].decode()
            postfix = self.ylabel["postfix"][i].decode()

            if (self.value or value) and value != "None":
                if not value:
                    value = self.value
                if not postfix:
                    postfix = self.postfix
                label += (
                    " {:{value}}".format(self.ylabel["y"][i], value=value) + postfix
                )
            ax.text(
                xmax,
                y[i],
                label,
                va="center",
                color=self.ylabel["color"][i],
                fontsize=fs,
                alpha=self.ylabel["alpha"][i],
            )
        # ax.set_xticks(xticks)
        # sns.despine(trim=True)
