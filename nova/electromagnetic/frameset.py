"""Configure superframe. Inherit DataArray for fast access else DataFrame."""
from typing import Collection, Any

import numpy as np
import pandas
import shapely

from nova.electromagnetic.dataarray import (
    ArrayLocMixin,
    ArrayIndexer,
    DataArray
    )
from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.energize import Energize
from nova.electromagnetic.multipoint import MultiPoint
from nova.electromagnetic.polygon import Polygon


class SubSpaceError(IndexError):
    """Prevent direct access to frame's subspace variables."""

    def __init__(self, name, col):
        super().__init__(
            f'{name} access is restricted for subspace attributes. '
            f'Use frame.subspace.{name}[:, {col}] = *.\n\n'
            'Lock may be overridden via the following context manager '
            'but subspace will still overwrite (Cavieat Usor):\n'
            'with frame.metaframe.setlock(None):\n'
            f'    frame.{name}[:, {col}] = *')


class SetLocMixin(ArrayLocMixin):
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        col = self.obj.get_col(key)
        value = self.obj.format_value(col, value)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace') is True:
                raise SubSpaceError(self.name, col)
        if self.obj.in_field(col, 'energize'):
            if self.obj.metaframe.lock('energize') is False:
                return self.obj.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Refresh subspace items prior to return."""
        col = self.obj.get_col(key)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace') is True:
                self.obj.set_frame(col)
        if self.obj.in_field(col, 'energize'):
            if self.obj.metaframe.lock('energize') is False:
                return self.obj.energize._get_item(super(), key)
        return super().__getitem__(key)


class SetIndexer(ArrayIndexer):
    """Extend pandas indexer."""

    @property
    def loc_mixin(self):
        """Return LocIndexer mixins."""
        return SetLocMixin


class FrameSet(SetIndexer, DataArray):
    """
    Extend pandas.DataFrame.

    - Add boolean methods (add_frame, drop_frame...).

    """

    def __init__(self,
                 data=None,
                 index: Collection[Any] = None,
                 columns: Collection[Any] = None,
                 attrs: dict[str, Collection[Any]] = None,
                 **metadata: dict[str, Collection[Any]]):
        super().__init__(data, index, columns, attrs, **metadata)
        self.update_attrs()

    def update_attrs(self):
        """Extract frame attrs from data and initialize."""
        self.attrs['energize'] = Energize(self)
        self.attrs['multipoint'] = MultiPoint(self)
        self.attrs['polygon'] = Polygon(self)
        self.update_columns()
        for attr in self.attrs:
            attribute = self.attrs[attr]
            if isinstance(attribute, MetaMethod) and not self.empty:
                if attribute.generate:
                    attribute.initialize()

    def __repr__(self):
        """Propagate frame subspace variables prior to display."""
        self.update_frame()
        return super().__repr__()

    def __getitem__(self, key):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        col = self.get_col(key)
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__getitem__(col)
            if self.metaframe.lock('subspace') is False:
                self.set_frame(col)
        if self.in_field(col, 'energize'):
            if self.metaframe.lock('energize') is False:
                return self.energize._get_item(super(), key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        col = self.get_col(key)
        #value = self.format_value(col, value)
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__setitem__(key, value)
            if self.metaframe.lock('subspace') is False:
                raise SubSpaceError('setitem', col)
        if self.in_field(col, 'energize'):
            if self.metaframe.lock('energize') is False:
                return self.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if self.hasattrs('subspace'):
            for col in [col for col in self.subspace if col in self]:
                self.set_frame(col)

    def set_frame(self, col):
        """Inflate subspace variable and setattr in frame."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(True, 'subspace'):
            value = getattr(self, col)
            if not isinstance(value, np.ndarray):
                value = value.to_numpy()
        with self.metaframe.setlock(None):
            if hasattr(self, 'subref'):  # inflate
                value = value[self.subref]
            super().__setitem__(col, value)

    def get_frame(self, col):
        """Return inflated subspace variable."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(False, 'subspace'):
            return super().__getitem__(col)

    def add_frame(self, *args, iloc=None, **kwargs):
        """
        Build frame from *args, **kwargs and concatenate with DataFrame.

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
        frame = pandas.DataFrame(self)
        if not isinstance(insert, pandas.DataFrame):
            insert = pandas.DataFrame(insert)
        if iloc is None:  # append
            frames = [frame, insert]
        else:  # insert
            frames = [frame.iloc[:iloc, :], insert, frame.iloc[iloc:, :]]
        frame = pandas.concat(frames, sort=sort)  # concatenate
        self.__init__(frame, attrs=self.attrs)

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
        return FrameSet(data, index=index, attrs=self.attrs)

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
            if attr in self.metaframe.default:
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

    frameset = FrameSet(Required=['Ic'], Array=['Ic'])
    frameset.add_frame(range(3))
    frameset.Ic = 7.7
    print(frameset.Ic)
