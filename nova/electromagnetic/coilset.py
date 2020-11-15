"""
Construct coilsets for electromagnetic calculation.

Coilsets constructed from a pair of CoilFrame objects derived from
pands.DataFrames. Fast access to and from CoilFrame objects managed via the
CoilData Class.

"""

from os import path
import inspect

import pandas as pd

from nova.definitions import root_dir
from nova.utilities.IO import pythonIO
from nova.electromagnetic.coilframe import CoilFrame
from nova.electromagnetic.coildata import CoilData
from nova.electromagnetic.biotmethods import BiotMethods
from nova.electromagnetic.plasmamethods import PlasmaMethods
from nova.electromagnetic.coilmethods import CoilMethods
from nova.electromagnetic.coilplot import CoilPlot


class CoilSet(pythonIO, BiotMethods, PlasmaMethods, CoilMethods, CoilPlot):
    """
    Instance wrapper for coilset data.

    Attributes
    ----------
        coil : CoilFrame
            Coil config.
        subcoil : CoilFrame
            Subcoil config.

    """

    # exchange coilset attributes
    _coilset_attributes = ['default_attributes',
                           'coilset_frames',
                           'coilset_metadata',
                           'coildata_attributes',
                           'biot_instances',
                           'biot_attributes']

    # main class attribures
    _coilset_frames = ['coil', 'subcoil']

    # additional_columns
    _coil_columns = ['dx', 'dz', 'dA', 'dCoil', 'nx', 'nz', 'subindex', 'part',
                     'cross_section',
                     'turn_section', 'turn_fraction', 'skin_fraction',
                     'patch', 'polygon',
                     'power', 'optimize', 'plasma', 'feedback',
                     'mpc', 'Nf', 'Nt',
                     'It', 'Ic', 'Psi', 'Bx', 'Bz', 'B']

    _subcoil_columns = ['dx', 'dz', 'dA', 'dl_x', 'dl_z', 'coil', 'part',
                        'cross_section', 'patch', 'polygon',
                        'power', 'optimize', 'feedback',
                        'plasma', 'mpc', 'Nt', 'It', 'Ic',
                        'Psi', 'Bx', 'Bz', 'B']

    _coildata_attributes = {'current_update': 'full'}

    # fast access np.array linked to CoilFrame via DataFrame
    _dataframe_attributes = ['x', 'z', 'dl', 'dt', 'rms', 'dx', 'dz',
                             'Ic', 'It', 'Nt', 'Psi', 'Bx', 'Bz', 'B',
                             'Fx', 'Fz', 'xFx', 'xFz', 'zFx', 'zFz', 'My']

    def __init__(self, **coilset):
        self._initialize_coilset()  # initialize coil and subcoil
        BiotMethods.__init__(self)  # initialize biotmethods
        PlasmaMethods.__init__(self)  # initialize plasma methods
        self.coilset = coilset  # exchange coilset and instance attributes

    @staticmethod
    def _filepath(filename, directory=None):
        if directory is None:
            directory = path.join(root_dir, 'data/Nova/coilsets')
        return path.join(directory, filename)

    def save_coilset(self, filename, directory=None):
        """
        Pickle coilset output.

        Parameters
        ----------
        filename : str
            Filename.
        directory : dir, optional
            Directory. The default is None (nova/data/Nova/coilsets/).

        Returns
        -------
        None.

        """
        filepath = self._filepath(filename, directory)
        self.solve_biot()  # update biot interaction matrices
        self._coilset = self.coilset  # link coilset for pythonIO save
        self.save_pickle(filepath, ['_coilset'])
        del self._coilset  # delete temp variable

    def load_coilset(self, filename, directory=None):
        """
        Load coilset pickle.

        Parameters
        ----------
        filename : str
            Filename.
        directory : dir, optional
            Directory. The default is None (nova/data/Nova/coilsets/).


        Raises
        ------
        LookupError
            File not found.

        Returns
        -------
        coilset : dict
            Coilset data.

        """
        filepath = self._filepath(filename, directory)
        if path.isfile(filepath + '.pk'):
            self.load_pickle(filepath)
            self._pickled_attributes = self._coilset['default_attributes']
            self.coilset = self._coilset
            del self._coilset  # delete temp variable
        else:
            raise LookupError(f'file {filepath} not found')
        return self.coilset

    def _initialize_coilset(self):
        self._extract_coilset_properties()
        self._initialize_default_attributes()
        coil_metadata = {'_additional_columns': self._coil_columns,
                         '_dataframe_attributes': self._dataframe_attributes,
                         '_coildata_attributes': {**self._coildata_attributes,
                                                  **{'subcoil': False}}}
        subcoil_metadata = {'_additional_columns': self._subcoil_columns,
                            '_dataframe_attributes':
                                self._dataframe_attributes,
                            '_coildata_attributes':
                                {**self._coildata_attributes,
                                 **{'subcoil': True}}}
        self.coil = CoilFrame(coilframe_metadata=coil_metadata)
        self.subcoil = CoilFrame(coilframe_metadata=subcoil_metadata)

    def _extract_coilset_properties(self):
        self._coilset_properties = [p for p, __ in inspect.getmembers(
            CoilSet, lambda o: isinstance(o, property))]

    def _initialize_default_attributes(self):
        self._default_attributes = {
            'dCoil': -1, 'dPlasma': 0.25, 'dShell': 0.5, 'dField': 0.2,
            'turn_fraction': 1, 'turn_section': 'circle'}

    @property
    def coilset(self):
        """
        Return dict of coilset attributes listed in self._coilset_attributes.

        Coilset property used to get and set coilset attributes.

        Parameters
        ----------
        coilset_attributes : dict or nested dict
            Setter for coilset attributes.

        Returns
        -------
        coilset_attributes : dict
            Coilset attributes listed in self._coilset_attributes.

        """
        coilset_attributes = {attribute: getattr(self, attribute)
                              for attribute in self._coilset_attributes}
        return coilset_attributes

    @coilset.setter
    def coilset(self, coilset_attributes):
        for attribute_name in self._coilset_attributes:
            if attribute_name in ['default_attributes', 'coilset_metadata',
                                  'coildata_attributes', 'biot_attributes']:
                default = coilset_attributes
            else:  # require attributes to be passed within attribute dict
                default = {}
            setattr(self, attribute_name,
                    coilset_attributes.get(attribute_name, default))

    @property
    def coildata_attributes(self):
        """
        Expose coildata_attributes from self.coil.

        Returns
        -------
        coildata_attributes : dict
            coildata attributes exposed from self.coil CoilFrame.

        """
        return self.coil.coildata_attributes

    @coildata_attributes.setter
    def coildata_attributes(self, coildata_attributes):
        """
        Set coildata_attriutes.

        Update only.
        Create new attributes via self.coilset_metadata[_coildata_attributes]

        Parameters
        ----------
        coildata_attributes : dict
            coildata_attributes.

        Returns
        -------
        None.

        """
        for attribute in self.coildata_attributes:
            if attribute in self._coilset_properties and \
                    attribute in coildata_attributes and \
                    not hasattr(self, f'_{attribute}'):
                setattr(self, attribute, coildata_attributes[attribute])

    def append_coilset(self, *args):
        """
        Append coilsets via coilset.setter.

        Parameters
        ----------
        *args : dict
            Coilset_attributes.

        Returns
        -------
        None.

        """
        for coilset in args:
            self.coilset = coilset

    def subset(self, index, invert=False):
        """
        Return Coilset subset.

        Parameters
        ----------
        index : int, list or Index
            Coil index.
        invert : bool, optional
            Invert selection. The default is False.

        Returns
        -------
        subset : CoilSet
            Subset extracted from parent CoilSet.

        """
        if not isinstance(index, pd.Index):
            index = self.coil.index[index]
        if not pd.api.types.is_list_like(index):
            index = [index]
        if invert:
            index = self.coil.loc[~self.coil.index.isin(index)].index
        subindex = []
        for _index in index:
            subindex.extend(self.coil.loc[_index, 'subindex'])
        coilset_frames = {'coil': self.coil.loc[index],
                          'subcoil': self.subcoil.loc[subindex]}
        return CoilSet(coilset_frames=coilset_frames)

    @property
    def default_attributes(self):
        for attribute in self._default_attributes:
            if attribute in self._coilset_properties \
                    and hasattr(self, f'_{attribute}'):  # update
                self._default_attributes[attribute] = getattr(self, attribute)
        return self._default_attributes

    @default_attributes.setter
    def default_attributes(self, default_attributes):
        for attribute in default_attributes:
            if attribute in self._coil_columns + self._subcoil_columns +\
                    self._coilset_properties:
                if attribute in self._coilset_properties and not \
                        hasattr(self, f'_{attribute}'):
                    setattr(self, attribute, default_attributes[attribute])
                self._default_attributes[attribute] = \
                    default_attributes[attribute]

    @property
    def coilset_frames(self):
        coilset_frames = {}
        for frame in self._coilset_frames:
            getattr(self, frame).refresh_dataframe()  # flush dataframe updates
            coilset_frames[frame] = getattr(self, frame)
        return coilset_frames

    @coilset_frames.setter
    def coilset_frames(self, coilset_frames):
        for frame in self._coilset_frames:
            coilframe = coilset_frames.get(frame, pd.DataFrame())
            if not coilframe.empty:
                CoilData.__init__(coilframe)  # re-initalize coildata
                getattr(self, frame).add_coil(coilframe)  # append coilframe

    @property
    def coilset_metadata(self):
        'extract coilset metadata from coilset frames (coil, subcoil)'
        coilset_metadata = {}
        for frame in self._coilset_frames:
            coilset_metadata[frame] = getattr(self, frame).coilframe_metadata
        return coilset_metadata

    @coilset_metadata.setter
    def coilset_metadata(self, coilset_metadata):
        'update coilframe metadata'
        for frame in self._coilset_frames:
            getattr(self, frame).coilframe_metadata = \
                coilset_metadata.get(frame, coilset_metadata)

    def update_coilframe_metadata(self, coilframe, **coilframe_metadata):
        'update coilset metadata coilframe in [coil, subcoil]'
        getattr(self, coilframe).coilframe_metadata = coilframe_metadata

    def _check_default(self, attribute):
        if not hasattr(self, f'_{attribute}'):
            setattr(self, f'_{attribute}', self._default_attributes[attribute])


if __name__ == '__main__':

    cs = CoilSet(dCoil=-1, current_update='coil', turn_fraction=0.5,
                 cross_section='circle')

    cs.add_coil(1.75, 0.5, 2.5, 2.5, name='PF13', part='PF', Nt=10, It=0,
                cross_section='circle', turn_fraction=1,
                dCoil=-15)

    cs.add_plasma([1, 3, 2, 3])
    cs.plot(True)
