"""Manage access to Naka datafiles."""
import os
import io
import json
from dataclasses import dataclass, field
from typing import Union

import regex
import mechanize
import urllib.request
import fitz
import pandas
import numpy as np
import scipy.signal

from nova.thermalhydralic.naka.nakaserver import NakaServer
from nova.thermalhydralic.naka.localdata import LocalData
from nova.thermalhydralic.naka.pipeline import PipeLine
from nova.utilities.time import clock
from nova.utilities.pyplot import plt


@dataclass
class NakaData:
    """Serve data from Naka data center."""

    _year: int
    _index: dict = field(default_factory=dict, init=False, repr=False)
    metadata: pandas.DataFrame = field(init=False, repr=False)
    pipeline: PipeLine = field(init=False, repr=False)

    def __post_init__(self):
        """Init database year."""
        self.pipeline = PipeLine()
        self.year = self._year

    @property
    def year(self):
        """Manage database year."""
        return self._year

    @year.setter
    def year(self, year):
        self._year = year
        self.local = \
            LocalData('', parent_dir=os.path.join('Naka', f'{self.year}'))
        self.load_index()
        self.load_metadata()

    @property
    def indexfile(self):
        """Return full path to local index file."""
        return os.path.join(self.local.parent_directory, 'index.json')

    def load_index(self):
        """Load directory index."""
        if os.path.isfile(self.indexfile):  # serve file localy
            with open(self.indexfile, 'r') as jsonfile:
                self._index = json.load(jsonfile)
        else:  # read from server
            self._index = self._read_index()

    def _read_index(self):
        """Return index extracted from Naka server."""
        makedir = ~self.local.checkdir()  # generate structure if requred
        if makedir:
            self.local.makedir()
        try:
            with NakaServer(self.year) as server:
                index = server.index
                with open(self.indexfile, 'w') as jsonfile:
                    json.dump(index, jsonfile, indent=4)
        except mechanize.HTTPError as http_error:
            if makedir:
                self.local.removedir()  # remove if generated bare
            raise FileNotFoundError(f'Year {self.year} not found on '
                                    'naka server') from http_error
        return index

    @property
    def index(self):
        """Return server file index."""
        return self._index

    @property
    def shot_index(self):
        """Return run index list."""
        return list(self.index.keys())

    @property
    def shot_number(self):
        """Return shot number."""
        return len(self.index)

    def locate(self, shot: Union[int, list[int, int], str], files=[]):
        """
        Locate file, download from Naka server if not found.

        Parameters
        ----------
        shot : Union[int, list[int, int], str]
            Shot identifier.
        files : list[str], optional
            List of file identifiers ['pdf', 'csv',...]. The default is [].

        Raises
        ------
        IndexError
            DESCRIPTION.

        Returns
        -------
        localfiles : list[str]
            List of local files

        """
        if isinstance(files, str):
            files = [files]
        if isinstance(shot, int):
            shot = self.shot_index[shot]
        if isinstance(shot, list):
            mrun, srun = shot
            shot = f'MRun{mrun:03}_SRun{srun:03}'
        if shot not in self.index:
            raise IndexError(f'shot {shot} not found in '
                             f'run index \n\n{self.run_index}')
        if files:
            names, urls = [], []
            identifier = regex.compile(r"\L<files>", files=files)
            for name, url in zip(self.index[shot]['names'],
                                 self.index[shot]['urls']):
                if identifier.search(name):
                    names.append(name)
                    urls.append(url)
        else:
            names = self.index[shot]['names']
            urls = self.index[shot]['urls']
        localfiles = ['' for __ in range(len(names))]
        for i, (name, url) in enumerate(zip(names, urls)):
            directory = self._get_directory(name)
            localfile = os.path.join(directory, name)
            if not os.path.isfile(localfile):
                self.pipeline.append(name, url)
            localfiles[i] = localfile
        return localfiles

    def _get_directory(self, name):
        """Return directory based on filename extension."""
        if name[-4:] == '.pdf':
            directory = self.local.metadata_directory
        else:
            directory = self.local.source_directory
        return directory

    def download(self):
        """Download files in pipeline."""
        if self.pipeline.count > 0:
            tick = clock(
                self.pipeline.count,
                header=f'Downloading {self.pipeline.count} files '
                f'from the Naka server.')
            with NakaServer(self.year) as server:
                _url = server.browser.geturl()
                for name, url in self.pipeline.serve():
                    url = f'{_url}/{url}'
                    directory = self._get_directory(name)
                    filename = os.path.join(directory, name)
                    with urllib.request.urlopen(url) as response:
                        with open(filename, 'wb') as localfile:
                            localfile.write(response.read())
                    tick.tock()
            self.pipeline.flush()

    def _download_metadata(self):
        """Download all metadata files form Naka server for specified year."""
        for name in self.index:
            self.locate(name, files=['pdf'])  # build pipeline
        self.download()  # download files

    def _read_metadata(self):
        """Return shot objective from metadata pdfs."""
        self._download_metadata()
        metadata = pandas.DataFrame(index=range(len(self.index)),
                                    columns=['shot', 'objective'])
        metadata.loc[:, 'shot'] = self.shot_index
        for i in range(self.shot_number):
            metadata.loc[i, 'objective'] = self.read_objective(i)
        metadata.to_csv(self.metafile)
        return metadata

    @property
    def metafile(self):
        """Return full path to local metadata file."""
        return os.path.join(self.local.parent_directory, 'metadata.csv')

    def load_metadata(self):
        """Load directory index."""
        if os.path.isfile(self.metafile):  # serve file localy
            self.metadata = pandas.read_csv(self.metafile)
        else:  # read from server
            self.metadata = self._read_metadata()

    def read_objective(self, shot):
        """Return shot objective."""
        metafile = self.locate(shot, files='pdf')[0]
        doc = fitz.open(metafile)
        text = doc.loadPage(0).getText("text")
        text = text.replace(':', '\n').split('\n')
        text = [label.strip() for label in text]
        if 'Object' in text:
            index = text.index('Object')+1
        else:
            index = text.index('~')+5
        objective = text[index]
        print(objective)
        return objective

    def select(self, subobjective):
        """Return metadata subset."""
        index = self.metadata.objective.str.contains(subobjective)
        return self.metadata[index]

    def select_download(self, subobjective):
        """Download selected files."""
        for shot in self.select(subobjective).index:
            self.locate(shot, files='csv')
        self.download()


if __name__ == '__main__':

    naka = NakaData(2015)
    naka.select_download('AC')
    shot = naka.shot_index[284]
    file = naka.locate(284, files='1-H')[0]
    dataframe = pandas.read_csv(file, skiprows=7)
    columns = {}
    for name in dataframe.columns:
        columns[name] = name.replace('(sec)', '')
        columns[name] = columns[name].replace('phy(', '').replace(')', '')
    dataframe.rename(columns=columns, inplace=True)
    dataframe.dropna(inplace=True, axis=1)

    print(dataframe.shape)

    '''
    #Tin = dataframe['ICS_TS_07L']
    #Tout = dataframe['ICS_TS_02L']

    #dT = Tout-Tin
    index = (dataframe.time > 1600) & (dataframe.time < 2000)
    dataframe = dataframe.loc[index, :]

    dp = dataframe['ICS_PT_IN']#-dataframe['ICS_PT_IN']
    #plt.plot(dataframe.time, dataframe['ICS_PT_IN'])
    plt.plot(dataframe.time, dp)
    #plt.xlim([0, 500])

    fs = 1/np.mean(np.diff(dataframe.time))
    f, Pxx = scipy.signal.welch(dp, fs)

    plt.figure()
    plt.loglog(f[1:], Pxx[1:])

    1/0.005
    '''

