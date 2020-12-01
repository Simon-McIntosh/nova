
import os

from nova.definitions import root_dir
from nova.thermalhydralic.attributes import Attributes

'''
#_attributes = ['experiment', 'testname', 'shot', 'mode']
#_default_attributes = {'mode': 'ac', 'read_txt': False}
#_input_attributes = ['testname', 'shot', 'mode']
'''


class LocalData(Attributes):
    """Manage local data structure, directorys and attributes."""

    def __init__(self, *args, **kwargs):
        Attributes.__init__(self)
        self.attributes = ['experiment']
        self.default_attributes = {'subfolder': None, 'experimentdir': None,
                                   'datadir': None, 'localdir': None}
        self.initialize_attributes()
        self.set_attributes(*args, **kwargs)

    @property
    def experiment(self):
        """
        Manage experiment identifier.

        Reinitialize if changed.

        Parameters
        ----------
        experiment : str
            Test directory name, evaluated as ftp/parentdir/experiment.

        Returns
        -------
        experiment : str

        """
        if self._experiment is None:
            raise IndexError(f'experiment not set.\n {self.listdir()}')
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        if self._experiment is None:
            self._experiment = experiment
        elif self._experiment != experiment:
            self.__init__(experiment)

    @property
    def subfolder(self):
        return self._subfolder

    @subfolder.setter
    def subfolder(self, subfolder):
        self._subfolder = subfolder
        self._experimentdir = None

    @property
    def experimentdir(self):
        if self._experimentdir is None:
            directory = os.path.join(root_dir, 'data')
            if self.subfolder:
                directory = os.path.join(directory, f'{self.subfolder}')
            directory = os.path.join(directory, f'{self.experiment}')
            if not os.path.isdir(directory):
                os.mkdir(directory)
            self._experimentdir = directory
        return self._experimentdir

    @property
    def datadir(self):
        return self._datadir

    def _set_experiment(self, experiment):
        self._experiment = experiment
        self._setdir()

    def _setdir(self):
        if self._experiment is not None:
            self.experimentdir = os.path.join(root_dir,
                                       f'data/Sultan/{self.experiment}')
            self.datadir = os.path.join(self.expdir, 'ftp')
            self.localdir = os.path.join(self.expdir, 'local')
            self._mkdir(['experiment', 'data', 'local'])

    def _mkdir(self, names):
        for name in names:
            directory = getattr(self, f'{name}dir')
            if not os.path.isdir(directory):
                os.mkdir(directory)

    def _rmdir(self, names):
        for name in names:
            directory = getattr(self, f'{name}dir')
            if os.path.isdir(directory):
                os.rmdir(directory)


if __name__ == '__main__':

    local = LocalData('Sultan')
    print(local.experiment)
    local.experiment = 'CSJA_3'
    print(local.experiment)
