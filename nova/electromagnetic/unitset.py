"""Configure superframe. Inherit DataArray for fast access else DataFrame."""
from dataclasses import dataclass, field
from typing import Collection, Any

import numpy as np
import pandas
import shapely

from nova.electromagnetic.dataarray import (
    ArrayLocMixin,
    ArrayIndexer,
    DataArray
    )
from nova.electromagnetic.dataframe import DataFrame
from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.multipoint import MultiPoint
from nova.electromagnetic.energize import Energize


class UnitSetLocMixin(ArrayLocMixin):
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        col = self.obj.get_col(key)
        value = self.obj.format_value(col, value)
        if self.obj.metaframe.hascol('energize', col):
            if self.obj.metaframe.lock('energize') is False:
                return self.obj.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Refresh subspace items prior to return."""
        col = self.obj.get_col(key)
        if self.obj.metaframe.hascol('energize', col):
            if self.obj.metaframe.lock('energize') is False:
                return self.obj.energize._get_item(super(), key)
        return super().__getitem__(key)


class UnitSetIndexer(ArrayIndexer):
    """Extend pandas indexer."""

    @property
    def loc_mixin(self):
        """Return LocIndexer mixins."""
        return UnitSetLocMixin


@dataclass
class Methods:
    """Manage frame MetaMethods."""

    frame: DataFrame
    attrs: dict[Any] = field(repr=False, default_factory=dict)

    def __post_init__(self):
        """Define methods, update frame.columns and initialize methods."""
        self.frame.add_methods()
        self.initialize()

    def __repr__(self):
        """Return method list."""
        return f'{list(self.attrs)}'

    def initialize(self):
        """Init attrs derived from MetaMethod."""
        self.frame.update_columns()
        if self.frame.empty:
            return
        attrs = [attr for attr in self.attrs
                 if isinstance(self.attrs[attr], MetaMethod)]
        for attr in attrs:
            if self.attrs[attr].generate:
                self.attrs[attr].initialize()


class UnitSet(UnitSetIndexer, DataArray):
    """
    Extend DataArray.

    - Add boolean methods (insert, drop...).
    - Frame singleton (no subspace, select, geometory or plot methods)

    """

    def __init__(self,
                 data=None,
                 index: Collection[Any] = None,
                 columns: Collection[Any] = None,
                 attrs: dict[str, Collection[Any]] = None,
                 **metadata: dict[str, Collection[Any]]):
        super().__init__(data, index, columns, attrs, **metadata)
        self.attrs['methods'] = Methods(self, self.attrs)

    def add_methods(self):
        """Define singleton attributes - extend to add additional methods."""
        self.attrs['multipoint'] = MultiPoint(self)
        self.attrs['energize'] = Energize(self)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if self.metaframe.hascol('energize', col):
            if self.metaframe.lock('energize') is False:
                return self.energize._get_item(super(), col)
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        value = self.format_value(col, value)
        if self.metaframe.hascol('energize', col):
            if self.metaframe.lock('energize') is False:
                return self.energize._set_item(super(), col, value)
        return super().__setitem__(col, value)

    def insert(self, *required, iloc=None, **additional):
        """
        Insert frame(s).

        Assemble insert from *args, **kwargs and concatenate with self.

        Parameters
        ----------
        *required : Union[float, array-like]
            Required arguments listed in self.metaframe.required.
        iloc : int, optional
            Row locater for inserted coil. The default is None (-1).
        **additional : dict[str, Union[float, array-like]]
            Optional keyword as arguments listed in self.metaframe.additional.

        Returns
        -------
        index : pandas.Index
            built frame.index.

        """
        self.metadata = additional.pop('metadata', {})
        insert = self.assemble(*required, **additional)
        self.concatenate(insert, iloc=iloc)
        return insert.index

    def concatenate(self, *insert, iloc=None, sort=False):
        """Concatenate insert with self."""
        frame = pandas.DataFrame(self)
        if iloc is None:  # append
            frames = [frame, *insert]
        else:  # insert
            frames = [frame.iloc[:iloc, :], *insert, frame.iloc[iloc:, :]]
        frame = pandas.concat(frames, sort=sort)  # concatenate
        self.__init__(frame, attrs=self.attrs)

    def drop(self, index=None):
        """Drop frame(s) from index."""
        if index is None:
            index = self.index
        self.multipoint.drop(index)
        super().drop(index, inplace=True)
        self.__init__(self, attrs=self.attrs)

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

    def assemble(self, *args, **kwargs):
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
        args, kwargs = self._extract_frame(*args, **kwargs)
        data = self._build_data(*args, **kwargs)
        index = self._build_index(data, **kwargs)
        return UnitSet(data, index=index, attrs=self.attrs)

    def _extract_frame(self, *args, **kwargs):
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
            Output argument number len(args) != len(self.metaframe.required).

        Returns
        -------
        args : list[Any]
            Return argument list, replaced input arg[0] is *frame.
        kwargs : dict[str, Any]
            Return keyword arquments, updated if input arg[0] is *frame.

        """
        if len(args) == 0:
            raise IndexError('len(args) == 0, argument number must be > 0')
        if self.isframe(args[0], frame=True) and len(args) == 1:
            frame = args[0]
            missing = [arg not in frame for arg in self.metaframe.required]
            if np.array(missing).any():
                required = np.array(self.metaframe.required)[missing]
                raise ValueError(f'required arguments {required} '
                                 'not specified in frame '
                                 f'{frame.columns}')
            args = [frame.loc[:, col] for col in self.metaframe.required]
            if not isinstance(frame.index, pandas.RangeIndex):
                kwargs['name'] = frame.index
            kwargs |= {col: frame.loc[:, col] for col in
                       self.metaframe.additional if col in frame}
        if len(args) != len(self.metaframe.required):
            raise IndexError(
                'incorrect required argument number (*args)): '
                f'{len(args)} != {len(self.metaframe.required)}\n'
                f'required *args: {self.metaframe.required}\n'
                f'additional **kwargs: {self.metaframe.additional}')
        return args, kwargs

    def _build_data(self, *args, **kwargs):
        """Return data dict built from *args and **kwargs."""
        data = {}  # python 3.6+ assumes dict is insertion ordered
        attrs = self.metaframe.required + list(kwargs)  # record passed attrs
        for attr, arg in zip(self.metaframe.required, args):
            data[attr] = np.array(arg, dtype=float)  # add required arguments
        for attr in self.metaframe.additional:  # set additional to default
            data[attr] = self.metaframe.default[attr]
        additional = []
        for attr in list(kwargs.keys()):
            if attr in self.metaframe.tag:
                kwargs.pop(attr)  # skip tags
            elif attr in self.metaframe.default:
                data[attr] = kwargs.pop(attr)  # add keyword arguments
                if attr not in self.metaframe.additional:
                    additional.append(attr)
        if len(additional) > 0:  # extend aditional arguments
            self.metaframe.metadata = {'additional': additional}
        if 'It' in attrs and 'Ic' not in attrs:  # patch line current
            data['Ic'] = \
                data['It'] / data.get('Nt', self.metaframe.default['Nt'])
        if len(kwargs) > 0:  # ckeck for unset kwargs
            unset_kwargs = np.array(list(kwargs.keys()))
            default = {key: '_default_value_' for key in unset_kwargs}
            raise IndexError(
                f'unset kwargs: {unset_kwargs}\n'
                'enter default value in self.metaframe.defaults\n'
                f'set as self.metaframe.meatadata = {{default: {default}}}')
        return data


if __name__ == '__main__':

    unitset = UnitSet(Required=['x', 'z'], available=['section', 'link'])
    unitset.insert(range(2), 1, label='PF')
    unitset.insert(range(4), 1, link=True)
    unitset.insert(range(2), 1, label='PF')
    unitset.insert(range(4), 1, link=True)
    print(unitset)
