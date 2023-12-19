"""Manage Bokeh widgets."""
from dataclasses import dataclass, field
from functools import cached_property
import os
from warnings import warn

from bokeh.layouts import column, row
from bokeh.model import Model
from bokeh.models import Button, Select, TabPanel, TextInput
from numpy.linalg import LinAlgError

from nova.imas.autocomplete import AutoComplete
from nova.imas.database import Database


@dataclass
class IdsInput(AutoComplete):
    """Manage IDS widget."""

    text: bool = False
    title: str | None = None
    attributes: list[str] = field(
        default_factory=lambda: ["user", "machine", "pulse", "run", "occurrence"]
    )
    buttons: list[str] = field(default_factory=list)
    models: dict[str, Model] = field(init=False, repr=False, default_factory=dict)

    def _setattr(self, attr, value):
        """Update instance attribute cast to attributes type."""
        setattr(self, attr, type(getattr(self, attr))(value))

    def __getitem__(self, attr):
        """Return Bokeh model."""
        try:
            return self.models[attr]
        except KeyError:
            self.models[attr] = self._get_model(attr)
            return self.models[attr]

    def __setitem__(self, attr, value):
        """Update model."""
        self[attr].value = str(value)
        self._setattr(attr, value)
        if self.text:
            return
        if str(value) not in (valid := getattr(self, f"{attr}_list")):
            raise ValueError(f"{value} is not a valid selection {valid}")
        if (child := self._child(attr)) is None:
            return
        self[child].options = getattr(self, f"{child}_list")
        if self[child].options:
            self[child] = self[child].options[0]
        else:
            for attr in self.attributes[self._index(child) :]:
                self[attr].options = []

    def _update_button_status(self, button, button_type):
        """Update button status."""
        if button not in self.buttons:
            return
        self[button].button_type = button_type

    def set_hooks(self, simulator, **kwargs):
        """Set attribute update hooks."""
        for attr in self.attributes:
            self[attr].on_change("value", self._ids_hook(attr, simulator, **kwargs))
        for button in self.buttons:
            self[button].on_click(self._button_hook(button, simulator, **kwargs))

    def _ids_hook(self, attr, simulator, **kwargs):
        """Return attribute updater function."""

        def updater(attribute, old, new):
            self[attr] = new
            if (save := kwargs.get("save", None)) is not None:
                save[attr] = new
                save._update_button_status("write", "primary")
            if self.ids_attrs != simulator.ids_attrs:
                button_type = "primary"
            else:
                button_type = "success"
            self._update_button_status("load", button_type)

        return updater

    def _button_hook(self, button, simulator, **kwargs):
        """Return button updater function."""
        itime = kwargs.get("itime", None)
        update_itime = kwargs.get("update_itime", None)

        def load(event):
            """Load IDS."""
            if self.name == "equilibrium":
                kwargs = self.ids_attrs
            else:
                kwargs = {self.name: self.ids_attrs}
            try:
                simulator.load_ids(**kwargs)
                if self.name == "equilibrium":
                    self.update_ids_attrs(**self.ids_attrs)
                if update_itime is not None:
                    update_itime("value", -1, 0)
                if itime is not None:
                    itime.value = 0
                    itime.end = simulator.data.sizes["time"] - 1
                self._update_button_status("load", "success")
            except (KeyError, ValueError, LinAlgError) as error:
                warn(f"IDS load unsucsessful \n{error}")
                self._update_button_status("load", "danger")

        def reset(event):
            """Reset IDS."""
            simulator.reset()
            self.update_ids_attrs(**self.ids_attrs)
            update_itime("value", -1, simulator.itime)
            self._update_button_status("load", "success")
            self._update_button_status("reset", "success")

        def write(event):
            """Write IDS."""
            if self.occurrence <= simulator.occurrence:
                self["occurrence"] = Database(**simulator.ids_attrs).next_occurrence()
            simulator.write_ids(**self.ids_attrs)
            self._update_button_status("write", "success")

        match button:
            case "load":
                return load
            case "reset":
                return reset
            case "write":
                return write
            case _:
                raise NotImplementedError(f"function hook for {button} not implemented")

    def update_ids_attrs(self, **ids_attrs):
        """Update ids attrs."""
        for attr in self.attributes:
            if attr not in ids_attrs:
                continue
            self[attr] = ids_attrs[attr]

    def _index(self, attr):
        """Return position of attr in attribute list."""
        return self.attributes.index(attr)

    def _child(self, attr):
        """Return attr's child from attributes list."""
        try:
            return self.attributes[self._index(attr) + 1]
        except IndexError:
            return None

    def _model_kwargs(self, attr):
        """Return model kwargs."""
        kwargs = {"value": str(getattr(self, attr)), "title": f"{attr}"}
        if self.text:
            return kwargs | {"sizing_mode": "stretch_width"}
        return kwargs | {"options": getattr(self, f"{attr}_list")}

    @property
    def _model(self):
        """Return Bokeh model."""
        if self.text:
            return TextInput
        return Select

    def _get_model(self, attr: str):
        """Return Bokeh model."""
        if attr in self.attributes:
            return self._model(**self._model_kwargs(attr))
        if attr in self.buttons:
            return Button(label=attr, button_type="success")
        raise AttributeError(
            f"attr {attr} not specified in attributes {self.attributes}"
        )

    @cached_property
    def user_list(self):
        """Return user list."""
        return ["public", os.environ["USER"]]

    @property
    def _ids(self):
        """Return ids row element."""
        return row(
            [
                self["pulse"],
                self["run"],
                self["occurrence"],
                self["machine"],
                self["user"],
            ]
        )

    @property
    def _buttons(self):
        """Return button row element."""
        return row([self[attr] for attr in self.buttons])

    @property
    def panel(self):
        """Return Bokeh ids tab."""
        name = f"{self.name}_panel"
        if self.buttons:
            return column([self._ids, self._buttons], name=name)
        return column(self._ids, name=name)

    @cached_property
    def tab(self):
        """Return Bokeh ids tab."""
        if (title := self.title) is None:
            title = self.name
        return TabPanel(child=self.panel, title=title)


if __name__ == "__main__":
    from apps.pulsedesign import ids_attrs

    ids = IdsInput(**ids_attrs)
    print(ids["pulse"])

    """

run = Select(value=str(ids_attrs["run"]), title="run:", options=equilibrium.run_list)
machine = Select(
    value=ids_attrs["machine"], title="machine:", options=equilibrium.machine_list
)
occurrence = Select(
    value=str(ids_attrs["occurrence"]),
    title="occurrence:",
    options=equilibrium.occurrence_list("equilibrium"),
)
user = Select(value="public", title="user:", options=)


    """
