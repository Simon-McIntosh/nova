"""Biot specific Frame class."""
import numpy as np

from nova.electromagnetic.framelink import FrameLink


_cross_section_factor = {'circle': np.exp(-0.25),  # circle-circle
                         'square': 2*0.447049,  # square-square
                         'skin': 1}  # skin-skin

_cross_section_key = {'rectangle': 'square',
                      'eliplse': 'circle',
                      'polygon': 'square',
                      'shell': 'square'}

#'cs_factor': self._cross_section_factor['square']}

# pylint: disable=too-many-ancestors


class BiotFrame(FrameLink):
    """Extend CoilFrame class with biot specific attributes and methods."""

    def __init__(self, data=None, index=None, columns=None, attrs=None,
                 **metadata):
        metadata = {'required': ['x', 'z'],
                    'available': ['link', 'section', 'poly']} | metadata
        super().__init__(data, index, columns, attrs, **metadata)

    def insert(self, *required, iloc=None, **additional):
        """
        Extend FrameArray.insert.

        Create link to metaframe.data.

        """
        if len(required) == 0:
            return None
        if self.isframe(required[0]):
            self.attrs['frame'] = required[0]
        return super().insert(*required, iloc=iloc, **additional)
        # self._update_cross_section_factor()


    @property
    def region(self):
        """
        Source / target region, read only.

        Set value via self.nT or self.nS'

        Returns
        -------
        region : str
            region type.

        """
        return self._region

    @property
    def nS(self):
        """
        Manage source filament number for target region.

        Parameters
        ----------
        value : int
            Set source filament number.

        Returns
        -------
        nS : int
            Number of source turns.

        """
        return self._nS

    @nS.setter
    def set_source_number(self, value):
        self._region = 'target'
        self._nT = self.nC
        self._nS = value

    @property
    def nT(self):
        """
        Manage target filament number for source region.

        Parameters
        ----------
        value : int
            Set target filament number.

        Returns
        -------
        nT : int
            Number of target turns.

        """
        return self._nT

    @nT.setter
    def set_target_number(self, value):
        self._region = 'source'
        self._nS = self.nC
        self._nT = value


if __name__ == '__main__':

    biotframe = BiotFrame({'x': range(12), 'z': 5, 'section': 'sq'})
    print(biotframe)
    biotframe.polyplot()


'''

    def update(self):
        """Update self."""
        self.drop()
        self.insert(self.frame)
        if self.frameindex != slice(None):
            self.index_coil(self.frameindex)

    def index_coil(self, index):
        """
        Drop coils, index coilframe.

        Parameters
        ----------
        index : Union[int, str, list[int], list[bool], list[str],
                      pandas.Index, slice]
            Coil index.

        Returns
        -------
        None.

        """
        if not isinstance(index, pandas.Index):
            index = self.coilframe.index[index]
        index = np.array([name in index for name in self.coilframe.index])
        self._frameindex = index[self.coilframe._reduction_index]
        drop_index = self.index[~index]
        CoilFrame.drop_coil(self, drop_index)
        self._update_cross_section_factor()

    def _link_coilframe(self, *args):
        """Link to coilframe instance to propagate future coilframe updates."""
        if self._is_coilframe(*args, accept_dataframe=False):
            self.coilframe = args[0]

    def update_coilframe(self):
        """
        Rebuild coilframe following geometric changes to coilset.

        Returns
        -------
        None.

        """
        if hasattr(self, 'coilframe'):
            if self.coilframe is not None:
                if self.coilframe.nC != self._framenumber:
                    self.update_coil()

    def _update_cross_section_factor(self):
        """Calculate factor applied to self inductance calculations."""
        cross_section = [cs if cs in self._cross_section_factor
                         else self._cross_section_key.get(cs, 'square')
                         for cs in self.cross_section]
        self.cs_factor = np.array([self._cross_section_factor[cs]
                                   for cs in cross_section])


    def __getattr__(self, key):
        """Assemble (nT,nS) matrix if key == _*_."""
        if key[0] == '_' and key[-1] == '_' \
                and key[1:-1] in self._dataframe_attributes:
            key = key[1:-1]
            value = CoilFrame.__getattr__(self, f'_{key}')
            if key in self._mpc_attributes:  # inflate
                value = value[self._mpc_referance]
            if self.nS is None or self.nT is None or self.region is None:
                err_txt = 'complementary source (self.nS) or target (self.nT) '
                err_txt += 'number not set'
                raise IndexError(err_txt)
            if self.region == 'source':  # assemble source
                value = np.dot(np.ones((self.nT, 1)),
                               value.reshape(1, -1)).flatten()
            elif self.region == 'target':  # assemble target
                value = np.dot(value.reshape(-1, 1),
                               np.ones((1, self.nS))).flatten()
            return value
        else:
            return CoilFrame.__getattr__(self, key)
'''