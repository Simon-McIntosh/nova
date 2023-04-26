"""Manage TFC fiducial data for coil and sector allignment."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import os

import numpy as np
import openpyxl
import pandas

from nova.definitions import root_dir

#with open('dict_of_dfs.pickle', 'wb') as f:
#    pickle.dump(d, f)


@dataclass
class FiducialSector:
    """Manage fiducial coil and sector assembly data."""

    file: str
    data: dict = field(init=False, repr=False)

    def __post_init__(self):
        """Build mesurment dataset."""
        #self.build()
        self.ccl_delta('SSAT BR')

    def ccl_delta(self, sheet):
        """Return dict of ccl deltas."""

        delta = {}
        with self.openbook():
            coil_names = self.coil_names('Nominal')
            for index, name in enumerate(coil_names):
                nominal = self.read_frame(index, 'Nominal')
                data = self.read_frame(index, sheet)
                #data = pandas.dataframe(index)

                print(data.loc[nominal.index, nominal.columns] -
                      nominal.loc[nominal.index, nominal.columns])
        return nominal


    @property
    def xls_file(self):
        """Return xls filename."""
        return os.path.join(root_dir, 'input/ITER', f'{self.file}.xlsx')

    def _initialize_data(self):
        """Initialize data as a bare nested dict with coil name entries."""
        self.data = {name: {} for name in self.coil_names('Nominal')}

    def build(self):
        """Build dataset."""
        with self.openbook():
            self._initialize_data()
            for worksheet in self.book:
                sheet = worksheet.title
                if sheet == 'Metadata':
                    continue
                for index, name in enumerate(self.coil_names(sheet)):
                    self.data[name][sheet] = self.read_frame(index, sheet)

    @contextmanager
    def openbook(self):
        """Manage access to source workbook."""
        self.book = openpyxl.load_workbook(self.xls_file, data_only=True)
        yield
        self.book.close()

    def locate(self, item: str, sheet: str):
        """Return item row/column locations in worksheet."""
        index = []
        for col in self.book[sheet].iter_cols():
            for cell in col:
                if cell.value == item:
                    index.append((cell.row, cell.column))
                    break
        assert len(index) == 2
        return index

    def coil_index(self, sheet: str):
        """Return list dataset origins."""
        return self.locate('Coil', sheet)

    def coil_names(self, sheet: str):
        """Return list of coil names."""
        name = []
        for row, cell in self.coil_index(sheet):
            name.append(self.book[sheet].cell(row+1, cell).value)
        return name

    def column_number(self, index, sheet: str):
        """Return column number."""
        for ncol, cell in enumerate(
                self.book[sheet].iter_cols(
                    min_row=index[0], max_row=index[0], min_col=index[1])):
            if cell[0].value is None:
                ncol -= 1
                break
        return ncol+1

    def read_frame(self, coil: int, sheet: str):
        """Return pandas dataframe from indexed sheet."""
        index = self.coil_index(sheet)[coil]
        ncol = self.column_number(index, sheet)
        usecols = list(range(index[1]-1, index[1]-1+ncol))
        data = pandas.read_excel(self.xls_file, sheet_name=sheet,
                                 skiprows=index[0]-1,
                                 usecols=usecols, index_col=[0, 1, 2],
                                 keep_default_na=False)
        data = data.rename(columns={col: col.split('.')[0]
                                    for col in data.columns})
        data.index.rename([name.split('.')[0] for name in data.index.names],
                          inplace=True)
        return data


if __name__ == '__main__':

    #sector = FiducialSector('Sector_Module_#6_CCL_as-built_data_8NQVKS_v2_0')

    sector = FiducialSector('Sector_Module_#7_CCL_as-built_data_8NR9J7_v2_0')
