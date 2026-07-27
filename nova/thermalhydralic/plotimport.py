"""Reach the plotting stack from plot methods only.

matplotlib and seaborn ship in an extra: the parsing, fitting and
frequency-response paths never touch them, so importing either at module scope
would put the whole cluster behind an optional install. These accessors keep the
import graph lean and turn an absent install into a message naming the extra
instead of a bare ImportError raised while a module is still loading.
"""

from importlib import import_module

from nova.utilities.importmanager import check_import


def pyplot():
    """Return matplotlib.pyplot."""
    with check_import("matplotlib"):
        return import_module("matplotlib.pyplot")


def gridspec():
    """Return matplotlib.gridspec."""
    with check_import("matplotlib"):
        return import_module("matplotlib.gridspec")


def seaborn():
    """Return seaborn."""
    with check_import("matplotlib"):
        return import_module("seaborn")


def clock(*args, **kwargs):
    """Return a progress ticker for a long read loop.

    nova.utilities.time reaches matplotlib at module scope, so the ticker is
    imported at the loop that uses it rather than beside the parsing code.
    """
    with check_import("matplotlib"):
        return import_module("nova.utilities.time").clock(*args, **kwargs)
