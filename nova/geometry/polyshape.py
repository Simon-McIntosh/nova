"""Provide polyshape key completion."""
from dataclasses import dataclass
import string
from typing import ClassVar


@dataclass
class PolyShape:
    """Provilde polyshape dict."""

    section: str
    polyshape: ClassVar[dict[str, str]] = \
        dict.fromkeys(['disc', 'dsc', 'd', 'o', 'disk', 'dsk',
                       'circle', 'c'], 'disc') | \
        dict.fromkeys(['ellipse', 'ellip', 'el', 'e'], 'ellipse') | \
        dict.fromkeys(['square', 'sq', 's'], 'square') | \
        dict.fromkeys(['rectangle', 'rect', 'rec', 'r'], 'rectangle') | \
        dict.fromkeys(['skin', 'sk', 'ring'], 'skin') | \
        dict.fromkeys(['polygon', 'poly'], 'polygon') |\
        dict.fromkeys(['shell', 'shl', 'sh'], 'shell') |\
        dict.fromkeys(['hexagon', 'hex', 'hx', 'h'], 'hexagon')

    @property
    def shape(self):
        """Return polygeom shape name."""
        section = self.section.rstrip(string.digits)
        try:
            return self.polyshape[section]
        except AttributeError as error:
            raise AttributeError(
                f'cross_section: {self.section} not implemented'
                f'\n specify as {self.polyshape}') from error
