"""Extend pandas.DataFrame to manage coil and subcoil data."""

from dataclasses import dataclass, field
from typing import Union

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.mesh import Mesh


@dataclass
class Section:
    """Set default sectional properties."""

    section: str = 'rectangle'
    turn: str = 'circle'
    turn_fraction: float = 1


@dataclass
class CoilFrame(Mesh, Section):
    """Mesh coils."""

    frame: Frame = field(init=False, repr=False)
    subframe: Frame = field(init=False, repr=False)
    metadata: dict[str, Union[str, dict]] = field(repr=False,
                                                  default_factory=dict)

    def __post_init__(self):
        """Init coil and subcoil."""
        required = ['x', 'z', 'dl', 'dt']
        additional = ['link', 'part', 'frame',
                      'dx', 'dz', 'dA', 'dl_x', 'dl_z', 'delta', 'nx', 'nz',
                      'section', 'turn', 'turn_fraction',
                      'Ic', 'It', 'Nt', 'Nf', 'Psi', 'Bx', 'Bz', 'B', 'acloss']
        metadata = {'section': self.section, 'turn': self.turn,
                    'turn_fraction': self.turn_fraction}
        metadata |= self.metadata
        self.frame = Frame(Required=required, Additional=additional,
                           Exclude=['dl_x', 'dl_z', 'frame'],
                           **metadata)
        self.subframe = Frame(Required=required, Additional=additional,
                              Exclude=['turn', 'turn_fraction',
                                       'Nf', 'delta'],
                              delim='_', **metadata)

    def add_poloidal(self, *required, iloc=None, mesh=True, **additional):
        """
        Add poloidal coil(s) to coilframe.

        Parameters
        ----------
        *required : Union[DataFrame, dict, list]
            Required input.
        iloc : int, optional
            Index before which coils are inserted. The default is None (-1).
        mesh : bool, optional
            Mesh coil. The default is True.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        delta = additional.pop('dpol', self.dpol)
        additional = {'delta': delta} | additional
        index = self.frame.insert(*required, iloc=iloc, **additional)
        if mesh:
            self.mesh_poloidal(index=index)


if __name__ == '__main__':

    coilframe = CoilFrame(dpol=0.05, metadata={'section': 'circle'})
    coilframe.add_poloidal(range(3), 1, 0.75, 0.75, link=True, delta=-1)
    coilframe.subframe.polyplot()
