import numpy as np
import pandas

from nova.electromagnetic.coilframe import CoilFrame
from nova.electromagnetic.coilmatrix import CoilMatrix
from nova.utilities.pyplot import plt


class BiotAttributes:
    """Manage attributes to and from Biot derived classes."""

    _biot_attributes = []
    _default_biot_attributes = {}

    def __init__(self, **biot_attributes):
        self._append_biot_attributes(self._biot_attributes)
        self._append_biot_attributes(self._coilmatrix_attributes)
        self._append_biot_attributes(self._default_coilmatrix_attributes)
        self._default_biot_attributes = {
            **self._default_coilmatrix_attributes,
            **self._default_biot_attributes}
        self.biot_attributes = biot_attributes

    def _append_biot_attributes(self, attributes):
        self._biot_attributes += [attr for attr in attributes
                                  if attr not in self._biot_attributes]

    @property
    def biot_attributes(self):
        return {attribute: getattr(self, attribute) for attribute in
                self._biot_attributes}

    @biot_attributes.setter
    def biot_attributes(self, _biot_attributes):
        for attribute in self._biot_attributes:
            default = self._default_biot_attributes.get(attribute, None)
            value = _biot_attributes.get(attribute, None)
            if value is not None:
                if type(value) == BiotFrame:
                    BiotFrame.__init__(getattr(self, attribute), value)
                    self.target.rebuild_coildata()
                else:
                    setattr(self, attribute, value)  # set value
            elif not hasattr(self, attribute):
                setattr(self, attribute, default)  # set default


class BiotFrame(CoilFrame):
    """Extend CoilFrame class with biot specific attributes and methods."""

    _cross_section_factor = {'circle': np.exp(-0.25),  # circle-circle
                             'square': 2*0.447049,  # square-square
                             'skin': 1}  # skin-skin

    _cross_section_key = {'rectangle': 'square',
                          'eliplse': 'circle',
                          'polygon': 'square',
                          'shell': 'square'}

    def __init__(self, *args, reduce=False):
        CoilFrame.__init__(self, *args, coilframe_metadata={
            '_required_columns': ['x', 'z'],
            '_additional_columns': ['rms', 'dx', 'dz', 'Nt', 'cross_section',
                                    'cs_factor', 'coil', 'plasma', 'mpc'],
            '_default_attributes': {'dx': 0., 'dz': 0., 'rms': 0.,
                                    'Nt': 1, 'mpc': '', 'coil': '',
                                    'plasma': False,
                                    'cross_section': 'square',
                                    'cs_factor':
                                        self._cross_section_factor['square']},
            '_dataframe_attributes': ['x', 'z', 'rms', 'dx', 'dz', 'Nt',
                                      'cs_factor'] + self._mpc_attributes,
            '_coildata_attributes': {'region': '', 'nS': 0., 'nT': 0.,
                                     'reduce': reduce,
                                     'current_update': 'full',
                                     'frameindex': slice(None),
                                     'framenumber': 0},
            'mode': 'overwrite'})
        self.coilframe = None

    @property
    def frameindex(self):
        """Return frame index."""
        return self._frameindex

    @property
    def reduce(self):
        """Return reduction boolean."""
        return self._reduce

    def add_coil(self, *args, **kwargs):
        """
        Extend CoilFrame.add_coil.

        Create link to coilframe if passed as single argument.

        Parameters
        ----------
        *args : CoilFrame or _required_columns [x, z]
            Frame arguments.
        **kwargs : _additional columns
            Ancillary data.

        Returns
        -------
        None.

        """
        self._link_coilframe(*args)  # store referance to CoilFrame
        if self.coilframe is not None:
            if self.coilframe.empty:
                return
        CoilFrame.add_coil(self, *args, **kwargs)
        self._framenumber = self.nC
        self._update_cross_section_factor()

    def update_coil(self):
        """Update coilframe."""
        self.drop_coil()
        self.add_coil(self.coilframe)
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
    def nS(self, value):
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
    def nT(self, value):
        self._region = 'source'
        self._nS = self.nC
        self._nT = value

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


class BiotSet(CoilMatrix, BiotAttributes):

    def __init__(self, source=None, target=None, **biot_attributes):
        CoilMatrix.__init__(self)
        BiotAttributes.__init__(self, **biot_attributes)
        self.source = BiotFrame(reduce=self.reduce_source)
        self.target = BiotFrame(reduce=self.reduce_target)
        self._nS, self._nT = 0, 0
        self.load_biotset(source, target)

    def load_biotset(self, source=None, target=None):
        if source is not None:
            self.source.add_coil(source)
        if target is not None:
            self.target.add_coil(target)

    def assemble_biotset(self):
        self.source.update_coilframe()
        self.target.update_coilframe()
        self.assemble()

    @property
    def nS(self):
        return self._nS

    @nS.setter
    def nS(self, nS):
        self._nS = nS
        self.target.nS = nS  # update target source filament number

    @property
    def nT(self):
        return self._nT

    @nT.setter
    def nT(self, nT):
        self._nT = nT
        self.source.nT = nT  # update source target filament number

    def assemble(self):
        self.nS = self.source.nC  # source filament number
        self.nT = self.target.nC  # target point number
        self.nI = self.source.nC*self.target.nC  # total number of interactions

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.source.x, self.source.z, 'C1o', label='source')
        ax.plot(self.target.x, self.target.z, 'C2.', label='target')
        ax.legend()
        ax.set_axis_off()
        ax.set_aspect('equal')


if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet
    cs = CoilSet(dCoil=0.2, dPlasma=0.05, turn_fraction=0.5)
    cs.add_coil(3.943, 7.564, 0.959, 0.984, Nt=248.64, name='PF1', part='PF')
    cs.add_coil(1.6870, 5.4640, 0.7400, 2.093, Nt=554, name='CS3U', part='CS')
    #cs.add_coil(1.6870, 3.2780, 0.7400, 2.093, Nt=554, name='CS2U', part='CS')
    #cs.add_plasma(3.5, 4.5, 1.5, 2.5, It=-15e6, cross_section='ellipse')

    #cs.add_plasma(3.5, 4.5, 1.5, 2.5, dPlasma=0.5,
    #              It=-15e6, cross_section='circle')

    cs.plot(True)


    source = BiotFrame(cs.subcoil)
