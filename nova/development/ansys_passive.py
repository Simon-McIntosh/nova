import os

import matplotlib.pyplot as plt
import moviepy.video.io.bindings
from moviepy.editor import VideoClip

import numpy as np
import scipy.interpolate
import scipy.integrate
import xarray

from nova.frame.coilset import CoilSet
from nova.imas.machine import Machine
from nova.graphics.plot import Plot1D, Plot2D

machine = Machine(
    pf_active="iter_md",
    pf_passive="iter_md",
    wall=False,
    tplasma="h",
    ngrid=1e4,
    ninductance=10,
    dshell=1.5,
)


def add_plasma_filaments(machine):
    """Add Ansys plasma filaments to machine geometry."""
    plasma = CoilSet(dcoil=-1, tcoil="r")
    plasma_data = xarray.open_dataset(
        os.path.join(machine.dirname, "MD_UP_16_APDL_plasma_currents.nc")
    )
    plasma.coil.insert(
        *plasma_data.position.T, 0.1, 0.1, label="plasma", turn="r", ifttt=False
    )
    machine += plasma
    machine.solve_biot()
    machine.store()


# add_plasma_filaments(machine)

coil_index = machine.loc["coil", :].index
passive_index = machine.loc["passive", :].index
source_passive = [attr in passive_index for attr in machine.inductance.data.source.data]
target_passive = [attr in passive_index for attr in machine.inductance.data.target.data]
mutual_passive = machine.inductance.data.Psi[target_passive, source_passive]


source_ex = [attr in coil_index for attr in machine.inductance.data.source.data]

self_vessel = np.diagonal(mutual_passive)
mutual_external = machine.inductance.data.Psi[target_passive, source_ex]

time = np.linspace(0, 0.05, 200)
current = np.zeros((time.shape[0], mutual_external.sizes["source"]))

plasma_data = xarray.open_dataset(
    os.path.join(machine.dirname, "MD_UP_16_APDL_plasma_currents.nc")
)
plasma_current = scipy.interpolate.interp1d(
    plasma_data.time, plasma_data.I_plasma, axis=0
)
current[:, -279:] = plasma_current(time)


current_rate = np.gradient(current, time, axis=0)

external_current = scipy.interpolate.interp1d(time, current, axis=0)
external_voltage = scipy.interpolate.interp1d(
    time, np.einsum("ij,kj->ik", current_rate, mutual_external), axis=0
)

resistance = self_vessel / 0.05
# resistance = np.zeros_like(self_vessel)


def fun(time, current, mutual_passive_inverse, resistance, external_voltage):
    """Return current derivative at requested time."""
    return mutual_passive_inverse @ (-resistance * current - external_voltage(time))


sol = scipy.integrate.solve_ivp(
    fun,
    (time[0], time[-1]),
    np.zeros(mutual_external.sizes["target"]),
    method="Radau",
    args=(
        np.linalg.inv(mutual_passive.data),
        resistance,
        external_voltage,
    ),
    t_eval=time,
)

plot = Plot1D()
plot.set_axes()

colors = {}
for i, part in enumerate(machine.sloc["passive", "part"]):

    name = machine.sloc["passive", :].index[i]
    nturn = machine.loc[name, "nturn"]
    turn_current = 1e-6 * nturn * sol.y[i]

    if part == "vs3j":
        part = machine.sloc["passive", :].index[i].split(" ")[-1][:4]
    if part not in colors:
        colors[part] = f"C{len(colors)}"
        plot.axes.plot(sol.t, turn_current, color=colors[part], label=part)
        continue
    plot.axes.plot(sol.t, turn_current, color=colors[part])

ylim = plot.axes.get_ylim()
plot.axes.plot(0.007 * np.ones(2), ylim, "gray", label="thermal quench")
plot.axes.plot(0.01 * np.ones(2), ylim, "black", label="current quench")


plot.axes.legend()
plot.axes.set_xlabel(r"time $t$ s")
plot.axes.set_ylabel(r"current $I$ MA")


def plot_frame(t):
    axes = plt.gca()
    axes.clear()

    index = np.argmin(abs(time - t))
    machine.sloc["coil", "Ic"] = external_current(sol.t[index])
    machine.sloc["passive", "Ic"] = sol.y[:, index]
    machine.plot()
    levels = np.linspace(-180, 0, 71)
    machine.grid.plot(levels=levels, nulls=False)
    fig = plt.gcf()
    fig.tight_layout(pad=0)
    return moviepy.video.io.bindings.mplfig_to_npimage(fig)


Plot2D().set_axes()
animation = VideoClip(plot_frame, duration=time[-1])
animation.speedx(0.1).write_gif(f"CS3U.gif", fps=50)
