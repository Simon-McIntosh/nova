
from dataclasses import dataclass, field

import numpy as np
import pandas
import shapely.geometry
import shapely.strtree

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.polygrid import PolyGrid


@dataclass
class Coil:
    """Mesh poloidal field coils (CS and PF)."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)
    delta: float

    def insert(self, *required, iloc=None, mesh=True, **additional):
        """
        Add poloidal field coil(s).

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
        additional = {'delta': self.delta} | additional
        index = self.frame.insert(*required, iloc=iloc, **additional)
        if mesh:
            self._mesh(index)

    def _mesh(self, index):
        """
        Mesh poloidal field coil(s).

        - Store filaments in subframe.
        - Link turns.

        """
        frame = self.frame.loc[index, ['poly', 'delta', 'turn', 'nturn']]
        # scale, fill
        subframe = []
        for i, name in enumerate(index):
            subframe.append(
                PolyGrid(**frame.iloc[i].to_dict(), tile=False, trim=True,
                         label=name).frame)
        self.subframe.concatenate(*subframe)




