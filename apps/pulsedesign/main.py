"""Specify Bokeh user interface."""
import os

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import (
    AutocompleteInput,
    CheckboxButtonGroup,
    NumericInput,
    Select,
    Slider,
    TabPanel,
    Tabs,
    TextInput,
    Toggle,
    Patch,
)
from bokeh.plotting import figure

from apps.pulsedesign import ids_attrs, Simulator

simulator = Simulator(**ids_attrs)
source = simulator.source

# os.path.join(self.home, 'imasdb'

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

flux = poloidal.multi_line("x", "z", source=source["levelset"], color="gray", alpha=0.5)
wall = poloidal.multi_line(
    "x", "z", source=source["wall"], color="gray", width=2, alpha=2
)
nulls = poloidal.scatter(
    "x", "z", source=source["x_points"], marker="x", size=8, line_color="red"
)
plasma = poloidal.multi_polygons(
    "x", "z", fill_alpha="ionize", line_alpha=None, source=source["plasma"]
)

# coils = poloidal.multi_polygons(source=source['coil'])


poloidal_toggle = CheckboxButtonGroup(
    labels=["plasma", "wall", "nulls", "flux", "coils"],
    active=[0, 1, 2, 3],
    name="poloidal_toggle",
)

pprime = figure(height=150)
pprime.line("psi_norm", "dpressure_dpsi", source=source["profiles"])

ffprime = figure(height=150)
ffprime.line("psi_norm", "f_df_dpsi", source=source["profiles"])


# plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
# kwargs = Properties().patch_properties(simulator.frame.part, simulator.frame.area)
# plot.multi_polygons(**simulator.subframe.polygeo.polygons)


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


def poloidal_visibility(attr, old, new):
    """Update wall visibility."""
    if old == new:
        return
    for i, label in enumerate(poloidal_toggle.labels):
        setattr(label, "visible", i in new)


simulator.itime = 0

itime.on_change("value", update_itime)

# ids.on_change('')

# poloidal_toggle.on_change('active', poloidal_visibility)

ids_tab = TabPanel(child=ids, title="IDS")
profile_tab = TabPanel(child=sliders, title="Profile")


user_input = Tabs(tabs=[ids_tab, profile_tab], name="user_input")


# Set up layouts and add to document
curdoc().add_root(user_input)
curdoc().add_root(poloidal)
curdoc().add_root(poloidal_toggle)
