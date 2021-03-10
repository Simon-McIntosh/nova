"""Extend pandas.DataFrame to manage coil and subcoil data."""

from typing import Optional, Collection, Any, Union

import pandas
import numpy as np
import shapely

from nova.electromagnetic.metaarray import MetaArray
from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.superframe import SuperFrame
from nova.electromagnetic.subspace import SubSpace
from nova.electromagnetic.dataspace import DataSpaceMixin


# pylint: disable=too-many-ancestors
# pylint:disable=unsubscriptable-object


class Frame(SuperFrame):  # DataSpaceMixin,
    """
    Extend SuperFrame.

    Adds boolean methods (add_frame, drop_frame...).
    """

    def __init__(self,
                 data=None,
                 index: Optional[Collection[Any]] = None,
                 columns: Optional[Collection[Any]] = None,
                 attrs: dict[str, Union[MetaArray, MetaFrame]] = None,
                 **metadata: Optional[dict]):
        super().__init__(data, index, columns, attrs, **metadata)
        #self.subspace = SubSpace(self)

    '''
    @property
    def line_current(self):
        return self.loc[:, 'Ic']

    @line_current.setter
    def line_current(self, current):
        self.loc[:, 'Ic'] = current
    '''

    def add_frame(self, *args, iloc=None, **kwargs):
        """
        Build frame from *args, **kwargs and concatenate with Frame.

        Parameters
        ----------
        *args : Union[float, array-like]
            Required arguments listed in self.metaframe.required.
        iloc : int, optional
            Row locater for inserted coil. The default is None (-1).
        **kwargs : dict[str, Union[float, array-like]]
            Optional keyword as arguments listed in self.metaframe.additional.

        Returns
        -------
        index : pandas.Index
            built frame.index.

        """
        self.metadata = kwargs.pop('metadata', {})
        insert = self._build_frame(*args, **kwargs)
        self.concat(insert, iloc=iloc)

    def concat(self, insert, iloc=None, sort=False):
        """Concatenate insert with DataFrame(self)."""
        dataframe = pandas.DataFrame(self)
        if not isinstance(insert, pandas.DataFrame):
            insert = pandas.DataFrame(insert)
        if iloc is None:  # append
            dataframes = [dataframe, insert]
        else:  # insert
            dataframes = [dataframe.iloc[:iloc, :],
                          insert,
                          dataframe.iloc[iloc:, :]]
        dataframe = pandas.concat(dataframes, sort=sort)  # concatenate
        Frame.__init__(self, dataframe, attrs=self.attrs)

    def _build_frame(self, *args, **kwargs):
        """
        Return Frame constructed from required and optional input.

        Parameters
        ----------
        *args : Union[Frame, pandas.DataFrame, array-like]
            Required arguments listed in self.metaframe.required.
        **kwargs : dict[str, Union[float, array-like, str]]
            Optional keyword arguments listed in self.metaframe.additional.

        Returns
        -------
        insert : pandas.DataFrame

        """
        args, kwargs = self._extract(*args, **kwargs)
        data = self._build_data(*args, **kwargs)
        index = self._build_index(data, **kwargs)
        insert = SuperFrame(data, index=index, attrs=self.attrs)
        return pandas.DataFrame(insert)

    def _extract(self, *args, **kwargs):
        """
        Return *args and **kwargs with data extracted from frame.

        If args[0] is a *frame, replace *args and update **kwargs.
        Else pass *args, **kwargs.

        Parameters
        ----------
        *args : Union[Frame, DataFrame, list[float], list[array-like]]
            Arguments.
        **kwargs : dict[str, Union[float, array-like]]
            Keyword arguments.

        Raises
        ------
        IndexError
            Input argument length must be greater than 0.
        ValueError
            Required arguments not present in *frame.
        IndexError
            Output argument number len(args) != self.metaframe.required_number.

        Returns
        -------
        args : list[Any]
            Return argument list, replaced input arg[0] is *frame.
        kwargs : dict[str, Any]
            Return keyword arquments, updated if input arg[0] is *frame.

        """
        if len(args) == 0:
            raise IndexError('len(args) == 0, argument number must be > 0')
        if self.isframe(args[0], dataframe=True) and len(args) == 1:
            dataframe = args[0]
            missing = [arg not in dataframe for arg in self.metaframe.required]
            if np.array(missing).any():
                required = np.array(self.metaframe.required)[missing]
                raise ValueError(f'required arguments {required} '
                                 'not specified in dataframe '
                                 f'{dataframe.columns}')
            args = [dataframe.loc[:, col] for col in self.metaframe.required]
            if not isinstance(dataframe.index, pandas.RangeIndex):
                kwargs['name'] = dataframe.index
            kwargs |= {col: dataframe.loc[:, col] for col in
                       self.metaframe.additional if col in dataframe}
        if len(args) != self.metaframe.required_number:
            raise IndexError(
                'incorrect required argument number (*args)): '
                f'{len(args)} != '
                f'{self.metaframe.required_number}\n'
                f'required *args: {self.metaframe.required}\n'
                f'additional **kwargs: {self.metaframe.additional}')
        return args, kwargs

    @staticmethod
    def isframe(obj, dataframe=True):
        """
        Return isinstance(arg[0], Frame | DataFrame) flag.

        Parameters
        ----------
        obj : Any
            Input.
        dataframe : bool, optional
            Accept pandas.DataFrame. The default is True.

        Returns
        -------
        isframe: bool
            Frame / pandas.DataFrame isinstance flag.

        """
        if isinstance(obj, Frame):
            return True
        if isinstance(obj, pandas.DataFrame) and dataframe:
            return True
        return False

    def _build_data(self, *args, **kwargs):
        """Return data dict built from *args and **kwargs."""
        data = {}  # python 3.6+ assumes dict is insertion ordered
        for key, arg in zip(self.metaframe.required, args):
            data[key] = np.array(arg, dtype=float)  # add required arguments
        # current_label = self._current_label(**kwargs)
        for key in self.metaframe.additional:
            if key in kwargs:
                data[key] = kwargs.pop(key)
            else:
                data[key] = self.metaframe.default[key]
        additional = []
        for key in kwargs:
            if key in self.metaframe.default:
                additional.append(key)
                data[key] = kwargs[key]
        for key in additional:
            del kwargs[key]
        if len(additional) > 0:  # extend aditional arguments
            self.metaframe.metadata = {'additional': additional}
        # self._propogate_current(current_label, data)
        unset = [key not in self.metaframe.tag for key in kwargs]
        if np.array(unset).any():
            unset_kwargs = np.array(list(kwargs.keys()))[unset]
            default = {key: '_default_value_' for key in unset_kwargs}
            raise IndexError(
                f'unset kwargs: {unset_kwargs}\n'
                'enter default value in self.metaframe.defaults\n'
                f'set as self.metaframe.meatadata = {{default: {default}}}')
        return data

    @staticmethod
    def _data_length(data):
        """
        Return maximum item length in data.

        Parameters
        ----------
        data : dict[str, Union[float, array-like]]

        Returns
        -------
        data_length : int
            Maximum item item in data.

        """
        try:
            data_length = np.max(
                [len(data[key]) for key in data
                 if pandas.api.types.is_list_like(data[key])])
        except ValueError:
            data_length = 1  # scalar input
        return data_length

    def drop_frame(self, index=None):
        """Drop frame(s)."""
        if index is None:
            index = self.index
        self.multipoint.drop(index)
        self.drop(index, inplace=True)

    def translate(self, index=None, xoffset=0, zoffset=0):
        """Translate coil(s)."""
        if index is None:
            index = self.index
        elif not pandas.api.types.is_list_like(index):
            index = [index]
        if xoffset != 0:
            self.loc[index, 'x'] += xoffset
        if zoffset != 0:
            self.loc[index, 'z'] += zoffset
        for name in index:
            self.loc[name, 'poly'] = \
                shapely.affinity.translate(self.loc[name, 'poly'],
                                           xoff=xoffset, yoff=zoffset)
            self.loc[name, 'patch'] = None  # re-generate coil patch
        # TODO regenerate Biot arrays...

    def _current_label(self, **kwargs):
        """Return current label, Ic or It."""
        current_label = None
        if 'Ic' in self.metaframe.required or 'Ic' in kwargs:
            current_label = 'Ic'
        elif 'It' in self.metaframe.required or 'It' in kwargs:
            current_label = 'It'
        return current_label

    @staticmethod
    def _propogate_current(current_label, data):
        """
        "Propogate current data, Ic->It or It->Ic.

        Parameters
        ----------
        current_label : str
            Current label, Ic or It.
        data : Union[pandas.DataFrame, dict]
            Current / turn data.

        Returns
        -------
        None.

        """
        if current_label == 'Ic':
            data['It'] = data['Ic'] * data['Nt']
        elif current_label == 'It':
            data['Ic'] = data['It'] / data['Nt']


if __name__ == '__main__':


    frame = Frame(Required=['x', 'z'], optimize=True,
                  dCoil=5, Additional=['Ic'])

    frame.add_frame(4, range(3), link=True)
    frame.add_frame(4, range(2), link=False)
    frame.add_frame(4, range(4000), link=True)

    print(frame)

    def set_current():
        frame.subspace.loc['Coil4':'Coil5', 'Ic'] = \
            np.random.rand(2)


    #frame.x = [1, 2, 3]
    #frame.x[1] = 6

    #print(frame)

    #frame.metaarray._lock = False
    #newframe = Frame()

