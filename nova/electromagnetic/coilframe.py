"""Extend pandas.DataFrame to manage coil and subcoil data."""

from dataclasses import dataclass, field

from nova.electromagnetic.frame import Frame


@dataclass
class CoilFrame:

    required: list[str] = field(default_factory=lambda: ['x', 'z', 'dl', 'dt'])
    additional: list[str] = field(default_factory=lambda: [
        'dx', 'dz', 'dA', 'dl_x', 'dl_z', 'dCoil', 'nx', 'nz', 'part', 'coil',
        'section', 'turn', 'turn_fraction', 'skin_fraction',
        'link', 'Ic', 'It', 'Nt', 'Nf',
        'Psi', 'Bx', 'Bz', 'B', 'acloss'])
    coil: Frame = field(init=False)
    subcoil: Frame = field(init=False)

    def __post_init__(self):
        """Init coil and subcoil."""
        self.coil = Frame(
            Required=self.required, Additional=self.additional,
            Exclude=['dl_x', 'dl_z', 'coil'])
        self.subcoil = Frame(
            Required=self.required, Additional=self.additional,
            Exclude=['turn', 'turn_fraction', 'skin_fraction', 'Nf',
                     'dCoil'])

    def add_coil(self, *required, iloc=None, subcoil=True, **additional):
        """
        Add coil(s) to coilframe.

        Parameters
        ----------
        *required : Union[DataFrame, dict, list]
            Required input.
        iloc : int, optional
            Index before which coils are inserted. The default is None (-1).
        subcoil : bool, optional
            Mesh subcoil. The default is True.
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        None.

        """
        index = self.coil.add_frame(*required, iloc=iloc, **additional)
        #if subcoil:
        #    self.meshcoil(index=index)


if __name__ == '__main__':

    coilframe = CoilFrame()
    coilframe.add_coil(range(13), 1, 0.1, 0.1, link=True)
    print(coilframe.coil.subspace)

