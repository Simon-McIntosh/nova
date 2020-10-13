import numpy as np

from nova.electromagnetic.coilframe import CoilFrame
from nova.electromagnetic.coilmatrix import CoilMatrix
from nova.utilities.pyplot import plt


class BiotAttributes:
    """Manage attributes to and from Biot derived classes."""

    _biot_attributes = []
    _default_biot_attributes = {}

    def __init__(self, **biot_attributes):
        self._append_biot_attributes(self._biotset_attributes)
        self._append_biot_attributes(self._coilmatrix_attributes)
        self._default_biot_attributes = {**self._biotset_attributes,
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
                    BiotFrame.__init__(self.target, value)  # set biot frame
                    self.target.rebuild_coildata()
                else:
                    setattr(self, attribute, value)  # set value
            elif not hasattr(self, attribute):
                setattr(self, attribute, default)  # set default


class BiotFrame(CoilFrame):

    _cross_section_factor = {'circle': np.exp(-0.25),  # circle-circle
                             'square': 2*0.447049,  # square-square
                             'skin': 1}  # skin-skin

    _cross_section_key = {'rectangle': 'square',
                          'eliplse': 'circle',
                          'polygon': 'square',
                          'shell': 'square'}

    def __init__(self, *args):
        CoilFrame.__init__(self, *args, coilframe_metadata={
            '_required_columns': ['x', 'z'],
            '_additional_columns': ['rms', 'dx', 'dz', 'dl',
                                    'Nt', 'cross_section',
                                    'cs_factor', 'coil', 'plasma', 'mpc'],
            '_default_attributes': {'dx': 0., 'dz': 0., 'rms': 0.,
                                    'dl': 0., 'Nt': 1, 'mpc': '', 'coil': '',
                                    'plasma': False,
                                    'cross_section': 'square',
                                    'cs_factor':
                                        self._cross_section_factor['square']},
            '_dataframe_attributes': ['x', 'z', 'rms', 'dx', 'dz', 'Nt',
                                      'cs_factor'] + self._mpc_attributes,
            '_coildata_attributes': {'region': '', 'nS': 0., 'nT': 0.,
                                     'current_update': 'full'},
            'mode': 'overwrite'})
        self.coilframe = None

    def add_coil(self, *args, **kwargs):
        self.link_coilframe(*args)  # store referance to CoilFrame
        if self.coilframe is not None:
            if self.coilframe.empty:
                return
        CoilFrame.add_coil(self, *args, **kwargs)
        self.update_cross_section_factor()

    def link_coilframe(self, *args):
        'set link to coilframe instance to permit future coilframe updates'
        if self._is_coilframe(*args, accept_dataframe=False):
            self.coilframe = args[0]

    def update_coilframe(self, force_update=False):
        if hasattr(self, 'coilframe'):
            if self.coilframe is not None:
                if self.coilframe.nC != self.nC or force_update:
                    self.drop_coil()
                    CoilFrame.add_coil(self, self.coilframe)
                    self.update_cross_section_factor()

    def update_cross_section_factor(self):
        cross_section = [cs if cs in self._cross_section_factor
                         else self._cross_section_key.get(cs, 'square')
                         for cs in self.cross_section]
        self.cs_factor = np.array([self._cross_section_factor[cs]
                                   for cs in cross_section])

    @property
    def region(self):
        'source / target region - implicit - set via self.nT or self.nS'
        return self._region

    @property
    def nS(self):
        'source filament number'
        return self._nS

    @nS.setter
    def nS(self, value):
        'set source filament number for target region'
        self._region = 'target'
        self._nT = self.nC
        self._nS = value

    @property
    def nT(self):
        'target filament number'
        return self._nT

    @nT.setter
    def nT(self, value):
        'set target filament number for source region'
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

    _biotset_attributes = {'_solve': True,
                           'source_turns': True, 'target_turns': False,
                           'reduce_source': True, 'reduce_target': False}

    def __init__(self, source=None, target=None, **biot_attributes):
        CoilMatrix.__init__(self)
        BiotAttributes.__init__(self, **biot_attributes)
        self.source = BiotFrame()
        self.target = BiotFrame()
        self._nS, self._nT = 0, 0
        self.load_biotset(source, target)

    def load_biotset(self, source=None, target=None):
        if source is not None:
            self.source.add_coil(source)
        if target is not None:
            self.target.add_coil(target)

    def update_biotset(self):
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
        plt.legend()


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
