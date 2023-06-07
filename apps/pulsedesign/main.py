"""Specify Bokeh user interface."""
import os

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import (
    AutocompleteInput,
    CDSView,
    IndexFilter,
    NumericInput,
    RadioGroup,
    Select,
    Slider,
    Switch,
    TextInput,
)
from bokeh.plotting import figure

from apps.pulsedesign import ids_attrs, Simulator
from nova.imas.equilibrium import EquilibriumData
from nova.imas.sample import Sample

equilibrium = EquilibriumData(**ids_attrs)  # load source equilibrium
sample = Sample(equilibrium.data)  # extract key features
simulator = Simulator(ids=sample.equilibrium_ids())  # pass to simulator instance
simulator.itime = 0  # initialize itime
simulator.lock = False
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
points = poloidal.scatter(
    "x", "z", source=source["points"], marker="circle_cross", size=8
)
pprime = figure(height=150)
pprime.line("psi_norm", "dpressure_dpsi", source=source["profiles"])

ffprime = figure(height=150)
ffprime.line("psi_norm", "f_df_dpsi", source=source["profiles"])

view = CDSView(filter=IndexFilter([simulator.itime]))

current = figure(
    height=210, name="current", y_range=(-50, 50), sizing_mode="stretch_both"
)
current.xaxis.axis_label = "time, s"
current.yaxis.axis_label = "current, kA"
for coil_name in simulator.sloc["free", :].index:
    current.line("time", coil_name, source=source["current"])
    current.circle("time", coil_name, source=source["current"], view=view, color="red")

vertical_force = figure(
    height=210, name="vertical_force", y_range=(-500, 500), sizing_mode="stretch_both"
)
vertical_force.xaxis.axis_label = "time, s"
vertical_force.yaxis.axis_label = "vertical force, MN"
for coil_name in simulator.coil_name:
    vertical_force.line("time", coil_name, source=source["vertical_force"])
    vertical_force.circle(
        "time", coil_name, source=source["vertical_force"], view=view, color="red"
    )


field = figure(height=210, name="field", y_range=(0, 15), sizing_mode="stretch_both")
field.xaxis.axis_label = "time, s"
field.yaxis.axis_label = "magnetic field, T"
for coil_name in simulator.field.coil_name:
    field.line("time", coil_name, source=source["field"])
    field.circle("time", coil_name, source=source["field"], view=view, color="red")


itime = Slider(
    title="itime",
    value=0,
    start=simulator.data.itime.data[0],
    end=simulator.data.itime.data[-1],
    step=1,
)

topology = RadioGroup(
    labels=["limiter", "single_null"], active=int(not simulator.limiter), inline=True
)
minor_radius = Slider(
    title="minor radius",
    value=simulator["minor_radius"],
    start=0.75,
    end=3,
    step=0.1,
    disabled=not simulator.limiter,
)
minimum_gap = Slider(
    title="minimum gap",
    value=simulator["minimum_gap"],
    start=0,
    end=0.5,
    step=0.025,
    disabled=simulator.limiter,
)
elongation = Slider(
    title="elongation", value=simulator["elongation"], start=0.5, end=2.5, step=0.025
)
triangularity_upper = Slider(
    title="triangularity upper",
    value=simulator["triangularity_upper"],
    start=-0.5,
    end=1.5,
    step=0.025,
)
triangularity_lower = Slider(
    title="triangularity lower",
    value=simulator["triangularity_lower"],
    start=-0.5,
    end=1.5,
    step=0.025,
)
triangularity_inner = Slider(
    title="triangularity inner",
    value=simulator["elongation_lower"],  # TODO IDS update
    start=-0.5,
    end=0.5,
    step=0.025,
)
triangularity_outer = Slider(
    title="triangularity outer",
    value=simulator["elongation_upper"],  # TODO IDS update
    start=-0.5,
    end=0.5,
    step=0.025,
)
square = Switch(active=simulator.square)

sliders = column(
    itime,
    topology,
    minor_radius,
    minimum_gap,
    elongation,
    triangularity_upper,
    triangularity_lower,
    triangularity_inner,
    triangularity_outer,
    square,
    name="sliders",
)


def update_disabled():
    """Update disabled status."""
    if simulator.limiter:
        minimum_gap.disabled = True
        minor_radius.disabled = False
    else:
        minimum_gap.disabled = False
        minor_radius.disabled = True


def update_itime(attr, old, new):
    """Implement itime update."""
    if old == new:
        return
    simulator.lock = True
    simulator.itime = new
    view.filter = IndexFilter([new])
    minor_radius.value = simulator["minor_radius"]
    minimum_gap.value = simulator["minimum_gap"]
    elongation.value = simulator["elongation"]
    triangularity_upper.value = simulator["triangularity_upper"]
    triangularity_lower.value = simulator["triangularity_lower"]
    triangularity_inner.value = simulator["elongation_lower"]  # TODO IDS update
    triangularity_outer.value = simulator["elongation_upper"]  # TODO IDS update
    topology.active = int(not simulator.limiter)
    update_disabled()
    simulator.lock = False


def update_plasmapoints(attribute):
    """Return point attribute update function."""

    def _update(attr, old, new):
        if old == new or simulator.lock:
            return
        simulator[attribute] = new
        simulator.fit()
        if attribute != "minor_radius":
            minor_radius.value = simulator["minor_radius"]
        update_disabled()
        simulator.update()
        simulator._data["current"][simulator.itime] = simulator.current
        simulator._data["vertical_force"][simulator.itime] = simulator.force.fz
        simulator._data["field"][simulator.itime] = simulator.field.bp

    return _update


def update_square(attr, old, new):
    """Implement square switch."""
    if old == new:
        return
    simulator.square = new
    simulator.fit()
    simulator.update()


itime.on_change("value", update_itime)
topology.on_change("active", update_plasmapoints("boundary_type"))
minor_radius.on_change("value", update_plasmapoints("minor_radius"))
minimum_gap.on_change("value", update_plasmapoints("minimum_gap"))
elongation.on_change("value", update_plasmapoints("elongation"))
triangularity_upper.on_change("value", update_plasmapoints("triangularity_upper"))
triangularity_lower.on_change("value", update_plasmapoints("triangularity_lower"))
triangularity_inner.on_change("value", update_plasmapoints("elongation_lower"))
triangularity_outer.on_change("value", update_plasmapoints("elongation_upper"))
square.on_change("active", update_square)

curdoc().add_root(sliders)
curdoc().add_root(poloidal)
curdoc().add_root(current)
curdoc().add_root(vertical_force)
curdoc().add_root(field)
