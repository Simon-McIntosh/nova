
import os
import json
from dataclasses import dataclass, field
from typing import Union

import regex
import mechanize
from bs4 import BeautifulSoup
import http.cookiejar
import ssl
import urllib.request
import PyPDF2
import pandas
import numpy as np

from nova.thermalhydralic.naka.localdata import LocalData
from nova.utilities.time import clock


@dataclass
class NakaServer:
    """Manage access to Naka server."""

    year: int
    url: str = 'https://scml-svr.naka.qst.go.jp'
    browser: mechanize.Browser = field(init=False, repr=False)

    def __post_init__(self):
        """Configure browser."""
        ssl._create_default_https_context = ssl._create_unverified_context
        self._configure_browser()

    def _configure_browser(self):
        self.browser = mechanize.Browser()
        self.browser.set_cookiejar(http.cookiejar.CookieJar())
        self.browser.set_handle_robots(False)

    def __enter__(self):
        """Login to Naka website and navigate to year."""
        self.browser.open(f'{self.url}/index-e.php')
        self.browser.select_form(nr=1)
        self.browser.form['userid'] = 'FG'
        self.browser.form['password'] = 'edmly70a'
        self.browser.submit()
        self.browser.open(f'{self.url}/data/{self.year}')
        return self

    def __exit__(self, type, value, traceback):
        """Close browser."""
        self.browser.close()

    @property
    def links(self):
        """Return server links to datafiles."""
        html = self.browser.response().read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.find_all('a')

    @property
    def index(self):
        """Return packaged urls and filenames."""
        files = {}
        for link in self.links:
            href = link.get('href')
            if href[-4:] == '.pdf':
                run = href.split('/')[-1][:-4]
            elif href[-4:] == '.csv':
                run = '_'.join(href.split('_')[1:3])
            else:
                continue
            if run not in files:
                files[run] = {'urls': [], 'names': []}
            files[run]['urls'].append(href)
            files[run]['names'].append(href.split('/')[1])
        return files


@dataclass
class PipeLine:

    urls: list = field(default_factory=list)
    names: list = field(default_factory=list)
    count: int = 0

    def append(self,  name, url):
        """Append name, url."""
        self.names.append(name)
        self.urls.append(url)
        self.count += 1

    def serve(self):
        """Return name, url tuple."""
        for name, url in zip(self.names, self.urls):
            yield name, url

    def flush(self):
        """Reset url pipeline."""
        self.__init__()


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
            names = self.index[name]['names']
            urls = self.index[name]['urls']
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
                header=f'Downloading {self.pipeline.count} experimental runs '
                f'from Naka server.')
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
        with open(metafile, 'rb') as pdf:
            reader = PyPDF2.PdfFileReader(pdf)
            text = reader.getPage(0).extractText()
        text = text.replace(':', '').replace('\n', '').split()
        start_index = text.index('Object')+1
        text = text[start_index:]
        stop_index = []
        for stop_key in ['note', 'Note', 'ICS_IH_01', 'Step',
                         'CSM1', 'RH', 'CSMC']:
            if stop_key in text[1:]:
                stop_index.append(text.index(stop_key))
                break
        if stop_index:
            stop_index = np.min(stop_index)
        else:
            raise IndexError(f'stop key not found in text {text}')
        objective = ' '.join(text[slice(stop_index)])
        if ')' in objective:
            objective = objective[:objective.index(')')+1]
        return objective


if __name__ == '__main__':

    naka = NakaData(2015)
    #naka.locate(0, files=['pdf'])
    #naka.download()
