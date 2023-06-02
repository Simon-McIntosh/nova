"""Specify Bokeh user interface."""
import os

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import (
    AutocompleteInput,
    NumericInput,
    Select,
    Slider,
    TextInput,
)
from bokeh.plotting import figure

from apps.pulsedesign import ids_attrs, Simulator
from nova.imas.equilibrium import EquilibriumData
from nova.imas.sample import Sample

equilibrium = EquilibriumData(**ids_attrs)  # load source equilibrium
sample = Sample(equilibrium.data)  # extract key features
simulator = Simulator(ids=sample.equilibrium_ids())  # pass to simulator instance
source = simulator.source

run = TextInput(value="", title="Run:", width=100)
occurrence = TextInput(value="", title="Occurrence:", width=100)
ids = column(
    NumericInput(value=ids_attrs["pulse"], title="pulse:", mode="int"),
    NumericInput(value=ids_attrs["run"], title="run:"),
    AutocompleteInput(value="iter", completions=["iter", "iter_md"]),
    Select(value="public", title="user:", options=["public", os.environ["USER"]]),
    name="ids",
)

"""
    machine: str = 'iter'
    occurrence: int = 1
    user: str = 'public'
    name: str | None = None
    backend: str = 'hdf5'
"""

poloidal = figure(name="poloidal", match_aspect=True, height=650)
poloidal.axis.visible = False

flux = poloidal.multi_line(
    "x", "z", source=source["levelset"], color="gray", alpha=0.5, level="overlay"
)
wall = poloidal.multi_line(
    "x", "z", source=source["wall"], color="gray", width=2, alpha=2
)
nulls = poloidal.scatter(
    "x", "z", source=source["x_points"], marker="x", size=8, line_color="red"
)
plasma = poloidal.multi_polygons(
    "x", "z", fill_alpha="ionize", line_alpha=0, source=source["plasma"]
)

pprime = figure(height=150)
pprime.line("psi_norm", "dpressure_dpsi", source=source["profiles"])

ffprime = figure(height=150)
ffprime.line("psi_norm", "f_df_dpsi", source=source["profiles"])

itime = Slider(
    title="itime",
    value=0,
    start=simulator.data.itime.data[0],
    end=simulator.data.itime.data[-1],
    step=1,
)

elongation = Slider(title="elongation", value=2, start=0, end=3, step=0.01)

sliders = column(itime, elongation, pprime, ffprime, name="sliders")


def update_itime(attr, old, new):
    """Implement itime update."""
    if old == new:
        return
    simulator.itime = new


simulator.itime = 0
itime.on_change("value", update_itime)

curdoc().add_root(sliders)
curdoc().add_root(poloidal)
