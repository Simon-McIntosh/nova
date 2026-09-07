"""Presentation media: one ink style, one layout, one set of scene painters.

The package renders prepared geometry. It reads no data source itself: adapters
under :mod:`nova.media.sources` turn a machine's stored equilibrium into the
frame records the painters consume, so a figure cannot silently depend on which
archive it came from.
"""

from nova.media.ink import DEFAULT_INK, InkStyle, poloidal_axes, trace_axes

__all__ = ["DEFAULT_INK", "InkStyle", "poloidal_axes", "trace_axes"]
