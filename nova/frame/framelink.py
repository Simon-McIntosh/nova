"""Extend DataArray - add multipoint and link methods."""
from contextlib import contextmanager
from copy import copy
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas
import shapely

from nova.frame.dataarray import (
    ArrayLocMixin,
    ArrayIndexer,
    DataArray
    )
from nova.geometry.polygeom import PolyGeom
from nova.geometry.polygon import Polygon

from nova.frame.metamethod import Energize, MultiPoint, Select


# pylint: disable=too-many-ancestors


class LinkLocMixin(ArrayLocMixin):
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __getitem__(self, key):
        """Extend pandas.indexer getitem. Compute turn current."""
        col = self.obj.get_col(key)
        if self.obj.hascol('energize', col):
            if self.obj.lock('energize') is False:
                return self.obj.energize._get_item(super(), key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Extend pandas.indexer setitem. Update energize variables."""
        col = self.obj.get_col(key)
        value = self.obj.format_value(col, value)
        if self.obj.hascol('energize', col):
            if self.obj.lock('energize') is False:
                return self.obj.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)


class LinkIndexer(ArrayIndexer):
    """Extend pandas indexer."""

    @property
    def loc_mixin(self):
        """Return LocIndexer mixins."""
        return LinkLocMixin


class FrameLink(LinkIndexer, DataArray):
    """
    Extend DataArray.

    - Extend boolean methods (insert, ...).
    - Implement multipoint link and energize methods.

    """

    def __init__(self, data=None, index=None, columns=None,
                 attrs=None, **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(Select, MultiPoint, Energize)

    def __setattr__(self, name, value):
        """Extend DataFrame.__setattr__. (frame.*)."""
        if self.hascol('energize', name):
            if self.lock('energize') is False:
                return self.energize._set_item(super(), name, value)
        return super().__setattr__(name, value)

    def __getitem__(self, col):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        if self.hasattrs('energize'):
            if self.hascol('energize', col):
                if self.lock('energize') is False:
                    return self.energize._get_item(super(), col)
        return super().__getitem__(col)

    def __setitem__(self, col, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        value = self.format_value(col, value)
        if self.hasattrs('energize'):
            if self.hascol('energize', col):
                if self.lock('energize') is False:
                    return self.energize._set_item(super(), col, value)
        return super().__setitem__(col, value)

    @staticmethod
    def isframe(obj, dataframe=True):
        """
        Return isinstance(arg[0], obj | DataFrame) flag.

        Parameters
        ----------
        obj : Any
            Input.
        dataframe : bool, optional
            Accept pandas.DataFrame. The default is True.

        Returns
        -------
        isframe: bool
            isinstance flag.

        """
        if isinstance(obj, FrameLink):
            return True
        if isinstance(obj, pandas.DataFrame) and dataframe:
            return True
        return False

    @contextmanager
    def insert_required(self, required=None):
        """Manage local required arguments."""
        _required = self.metaframe.required.copy()
        if required is None:
            required = _required
        self.update_metaframe(dict(Required=required))
        yield
        self.update_metaframe(dict(Required=_required))

    @staticmethod
    def _unpack_add(other):
        """Return required, iloc and additional input for insert operator."""
        if isinstance(other, pandas.DataFrame):
            return [other], dict(), other.columns.to_list()
        if isinstance(other, dict):
            return [], other, None
        return other, dict(), None

    def __copy__(self):
        """Return copy of frame."""
        frame = self.__class__()
        frame.__init__(self)
        return frame

    def __add__(self, other: Union[pandas.DataFrame, dict, npt.ArrayLike]):
        """Return union of self and other."""
        frame = copy(self)
        frame += other
        return frame

    def __iadd__(self, other: Union[pandas.DataFrame, dict, npt.ArrayLike]):
        """Return self augmented by other."""
        args, kwargs, required = self._unpack_add(other)
        with self.insert_required(required):
            self.insert(*args, **kwargs)
        return self

    def insert(self, *args, iloc=None, **kwargs):
        # pylint: disable=arguments-differ
        """
        Override pandas.DataFrame.insert for column managed DataFrame.

        Insert row(s).

        Assemble insert from *args, **kwargs and concatenate with self.

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
        self.metaframe.metadata = kwargs.pop('metadata', {})
        insert = self.assemble(*args, **kwargs)
        self.concatenate(insert, iloc=iloc)
        return insert.index

    def concatenate(self, *insert, iloc=None, sort=False):
        """Concatenate insert with self."""
        if self.hasattrs('subspace'):  # unlink subspace
            self.overwrite_array(self.subspace.metaframe.data)
        frame = pandas.DataFrame(self)
        if iloc is None:  # append
            frames = [frame, *insert]
        else:  # insert
            frames = [frame.iloc[:iloc, :], *insert, frame.iloc[iloc:, :]]
        frame = pandas.concat(frames, sort=sort)  # concatenate
        self.__init__(frame, attrs=self.attrs,
                      metadata=self.metaframe.metadata)
        self.update_version()
        return self

    def drop(self, index=None):
        """Drop frame(s) from index."""
        if index is None:
            index = self.index
        self.multipoint.drop(index)
        super().drop(index, inplace=True)
        self.__init__(self, attrs=self.attrs)
        self.update_version()

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
        Return FrameLink constructed from required and optional input.

        Parameters
        ----------
        *args : Union[DataFrame, array-like]
            Required arguments listed in self.metaframe.required.
        **kwargs : dict[str, Union[float, array-like, str]]
            Optional keyword arguments listed in self.metaframe.additional.

        Returns
        -------
        insert : pandas.DataFrame

        """
        args, kwargs = self._extract_frame(*args, **kwargs)
        args, kwargs = self._extract_polygon(*args, **kwargs)
        data = self._build_data(*args, **kwargs)
        index = self._build_index(data, **kwargs)
        return FrameLink(data, index=index, attrs=self.attrs)

    def _extract_frame(self, *args, **kwargs):
        """
        Return *args and **kwargs with data extracted from frame.

        If args[0] is a frame, replace *args and update **kwargs.
        Else pass *args, **kwargs.

        Parameters
        ----------
        *args : Union[DataFrame, Polygon, list[float], list[array-like]]
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
        if len(args) != 1:
            args += tuple(kwargs.pop(arg) for arg in self.metaframe.required
                          if arg in kwargs)
            return args, kwargs
        if not self.isframe(args[0], dataframe=True):
            return args, kwargs
        frame = args[0]
        missing = [arg not in frame for arg in self.metaframe.required]
        if np.array(missing).any():
            required = np.array(self.metaframe.required)[missing]
            raise KeyError(f'required arguments {required} '
                           f'from metaframe.required {self.metaframe.required}'
                           ' not specified in frame '
                           f'{frame.columns}')
        args = [frame[col] for col in self.metaframe.required]
        _ = [kwargs.pop(attr, None) for attr in self.metaframe.required]
        if not isinstance(frame.index, pandas.RangeIndex):
            kwargs['name'] = frame.index
        kwargs |= {col: frame[col] for col in
                   self.metaframe.columns if col in frame}
        if len(args) != len(self.metaframe.required):
            raise IndexError(
                'incorrect required argument number (*args)): '
                f'{len(args)} != {len(self.metaframe.required)}\n'
                f'required *args: {self.metaframe.required}\n'
                f'additional **kwargs: {self.metaframe.additional}')
        return args, kwargs

    def _extract_polygon(self, *args, **kwargs):
        """
        Return *args and **kwargs with data extracted from frame.

        If args[0].., replace *args and update **kwargs.
        Else pass *args, **kwargs.

        """
        if len(args) != 1:
            return args, kwargs
        import vedo
        if len(self.metaframe.required) == 1 and \
            (not isinstance(args[0], (shapely.geometry.Polygon,
                                      shapely.geometry.MultiPolygon, dict))
                or isinstance(args[0], vedo.Mesh)):
            return args, kwargs
        if isinstance(args[0], list) and all(
                [isinstance(poly, (Polygon, shapely.geometry.Polygon))
                 for poly in args[0]]):
            multipoly = {attr: [] for attr in PolyGeom(args[0][0]).geometry}
            for poly in args[0]:
                geometry = PolyGeom(poly).geometry
                for attr in multipoly:
                    multipoly[attr].append(geometry[attr])
            kwargs = kwargs | multipoly
        else:
            kwargs = kwargs | PolyGeom(args[0]).geometry
        args = [kwargs.pop(attr) for attr in self.metaframe.required]
        return args, kwargs

    def _build_data(self, *args, **kwargs):
        """Return data dict built from *args and **kwargs."""
        data = {}  # python 3.6+ assumes dict is insertion ordered
        kwargs = self._exclude(kwargs)
        attrs = self.metaframe.required + list(kwargs)  # record passed attrs
        self._build_required(data, *args)
        self._build_additional(data, **kwargs)
        self._patch_current(data, attrs)
        return data

    def _exclude(self, kwargs):
        """Return kwargs with exclude attributes removed."""
        for attr in self.metaframe.exclude:  # remove exclude attrs
            if attr in kwargs:
                del kwargs[attr]
        return kwargs

    def _build_required(self, data, *args):
        """Populate required attributes from args."""
        if len(args) != len(self.metaframe.required):
            raise IndexError(f'len(args) {len(args)} != '
                             'len(self.metaframe.required) '
                             f'{len(self.metaframe.required)}')
        for attr, arg in zip(self.metaframe.required, args):  # required
            try:
                data[attr] = np.array(arg, dtype=float)
            except (TypeError, ValueError):
                data[attr] = arg  # non-numeric input

    def _build_additional(self, data, **kwargs):
        """Populate data with additional attributes from kwargs."""
        for attr in self.metaframe.additional:  # set additional to default
            data[attr] = self.metaframe.default[attr]
        additional = []
        for attr in list(kwargs.keys()):
            if attr in self.metaframe.tag:
                kwargs.pop(attr)  # skip tags
            elif attr in self.metaframe.default:
                value = kwargs.pop(attr)
                try:
                    data[attr] = np.array(value)  # add keyword attrs
                except ValueError:
                    data[attr] = value
                if attr not in self.metaframe.additional:
                    additional.append(attr)
        if len(additional) > 0:  # extend aditional arguments
            self.metaframe.metadata = {'additional': additional}
        if len(kwargs) > 0:  # ckeck for unset kwargs
            unset_kwargs = np.array(list(kwargs.keys()))
            default = {key: '_default_value_' for key in unset_kwargs}
            raise IndexError(
                f'unset kwargs: {unset_kwargs}\n'
                'enter default value in self.metaframe.defaults\n'
                f'set as self.metaframe.metadata = {{default: {default}}}')

    def _patch_current(self, data, attrs=None):
        """Patch line current."""
        if attrs is None:
            attrs = data
        if 'It' in attrs and 'Ic' not in attrs:
            data['Ic'] = \
                data['It'] / data.get('nturn', self.metaframe.default['nturn'])
        elif 'It' in attrs and 'Ic' in attrs:
            data['It'] = \
                data['Ic'] * data.get('nturn', self.metaframe.default['nturn'])


if __name__ == '__main__':

    framelink = FrameLink(required=['x', 'z'], Available=['It'], Array=['Ic'])

    framelink.insert([-4, -5], 1, Ic=6.5, name='PF1',
                     active=False, plasma=True, frame='coil1')
    framelink.insert(range(4), 3, Ic=4, nturn=20, label='PF', link=True)

    #framelink.multipoint.link(['PF1', 'PF5'], factor=1)
