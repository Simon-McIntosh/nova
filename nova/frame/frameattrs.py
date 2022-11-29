"""Manage DataFrame metamethods."""
from contextlib import contextmanager
from typing import Collection, Any

import numpy as np
import pandas
import xxhash

from nova.frame.metaframe import MetaFrame
from nova.frame.metamethod import MetaMethod
from nova.frame.error import ColumnError
from nova.frame.indexer import LocIndexer

# pylint: disable=too-many-ancestors


class FrameAttrs(pandas.DataFrame):
    """
    Extend pandas.DataFrame.

    - Manage DataFrame metadata (metaarray, metaframe).

    """

    def __init__(self,
                 data=None,
                 index: Collection[Any] = None,
                 columns: Collection[Any] = None,
                 attrs: dict[str, Collection[Any]] = None,
                 **metadata: dict[str, Collection[Any]]):
        super().__init__(data, index, columns)
        self.update_metadata(data, columns, attrs, metadata)

    @property
    def version(self):
        """Return metaframe version container."""
        return self.attrs['metaframe'].version

    def __getattr__(self, name):
        """Extend pandas.DataFrame.__getattr__. (frame.*)."""
        if name in self.attrs:
            return self.attrs[name]
        self.check_column(name)
        return super().__getattr__(name)

    def check_column(self, name):
        """If name in metaframe.default, raise error if name in not columns."""
        if name in self.metaframe.default and name not in self.columns:
            raise ColumnError(name)

    def __setitem__(self, key, value):
        """Extend pandas.DataFrame setitem, check that key is in columns."""
        if self.lock('column') is False:
            self.check_column(key)
        super().__setitem__(key, value)

    def frame_attrs(self, *args):
        """Update metamethod attrs."""
        for metamethod in (arg(self) for arg in args):
            print(metamethod.name)
            if not metamethod.generate:
                continue
            print(metamethod.name, 'generate')
            method = metamethod()
            if not self.hasattrs(method.name):
                self.update_columns()
            self.attrs[method.name] = method
            self.attrs[method.name].initialize()

    def frame_attr(self, method, *method_args):
        """Update single metamethod."""
        name = method.name
        if method(self, *method_args).generate:
            self.update_columns()
            self.attrs[name] = method(self)
            self.attrs[name].initialize()

    def update_metadata(self, data, columns, attrs, metadata):
        """Update metadata. Set default and meta*.metadata."""
        self.extract_attrs(data, attrs)
        self.trim_columns(columns)
        self.extract_available(data, columns)
        self.update_metaframe(metadata)
        self.format_data(data)

    def extract_attrs(self, data, attrs):
        """Extract metaframe / metaarray from data / attrs."""
        if data is None:
            data = {}
        if attrs is None:
            attrs = {}
        if hasattr(data, 'attrs'):
            for attr in data.attrs:  # update metadata from data
                if isinstance(data.attrs[attr], (
                        MetaFrame, MetaMethod, LocIndexer, pandas.DataFrame)):
                    self.attrs[attr] = data.attrs[attr]
        for attr in attrs:  # update from attrs (replacing data.attrs)
            if isinstance(attrs[attr], (
                    MetaFrame, MetaMethod, LocIndexer, pandas.DataFrame)):
                self.attrs[attr] = attrs[attr]
        if not self.hasattrs('metaframe'):
            self.attrs['metaframe'] = MetaFrame(self.index)

    def trim_columns(self, columns):
        """Trim metaframe required / additional to columns."""
        if columns:  # trim to columns
            required = [attr for attr in self.metaframe.required
                        if attr in columns]
            additional = [attr for attr in self.metaframe.additional
                          if attr in columns]
            available = [attr for attr in self.metaframe.available
                         if attr in columns]
            self.metaframe.metadata = {'Required': required,
                                       'Additional': additional,
                                       'Available': available}

    def extract_available(self, data, columns):
        """Update metaframe.available."""
        try:
            data_columns = list(data)
        except TypeError:
            data_columns = []
        if columns is None:
            columns = []
        frame_columns = list(dict.fromkeys(list(columns)))
        self.metaframe.metadata = {'available': data_columns+frame_columns}

    def update_metaframe(self, metadata):
        """Update metaframe, appending available columns if required."""
        if isinstance(metadata.get('version', None), list):
            metadata['version'] = dict.fromkeys(metadata['version'])
        self.metaframe.update(metadata)
        if self.metaframe.columns:
            self.metaframe.metadata = {'available': self.metaframe.columns}
        self.match_columns()

    def hash_array(self, attr):
        """Return hash array."""
        if self.hasattrs('subspace') and attr in self.subspace:
            return getattr(self.subspace, attr)
        return getattr(self, attr)

    def loc_hash(self, attr) -> int:
        """Return xxhash of loc attribute."""
        try:
            return xxhash.xxh64(value := self.hash_array(attr)).intdigest()
        except TypeError:
            try:
                return xxhash.xxh64(value := value.to_numpy()).intdigest()
            except ValueError:
                return xxhash.xxh64(np.ascontiguousarray(value)).intdigest()
        except (ColumnError, KeyError):
            return None

    def update_version(self):
        """Update metaframe version hash dict."""
        metadata = dict(version={attr: self.loc_hash(attr) for attr in
                                 self.version})
        self.attrs['metaframe'].update(metadata)

    def match_columns(self):
        """Intersect metaframe.required with self.columns if not empty."""
        if not self.columns.empty:
            required = [attr for attr in self.metaframe.required
                        if attr in self.columns]
            self.metaframe.metadata = {'Required': required}

    def format_data(self, data):
        """Apply default formating to data passed as dict."""
        if isinstance(data, dict):
            with self.setlock(True):
                for col in self.columns:
                    self.loc[:, col] = self.format_value(col, self[col])

    def hasattrs(self, attr):
        """Return True if attr in self.attrs."""
        return attr in self.attrs

    def hascol(self, attr, col):
        """Expose metaframe.hascol."""
        return self.metaframe.hascol(attr, col)

    def format_value(self, col, value):
        """Return vector with dtype as type(metaframe.default[col])."""
        if not self.hasattrs('metaframe') or col == 'link':
            return value
        try:
            dtype = type(self.metaframe.default[col])
        except (KeyError, TypeError):  # no default type, isinstance(col, list)
            return value
        if isinstance(self.metaframe.default[col], type(None)):
            return value
        if value is None:
            return dtype(0)
        if pandas.api.types.is_list_like(value):
            return np.array(value, dtype)
        return dtype(value)

    def lock(self, key=None):
        """
        Return metaframe lock status.

        Parameters
        ----------
        key : str
            Lock label.

        """
        if key is None:
            return self.metaframe.lock
        return self.metaframe.lock[key]

    @contextmanager
    def setlock(self, status, keys=None):
        """
        Manage access to subspace frame variables.

        Parameters
        ----------
        status : Union[bool, None]
            Subset lock status.
        keys : Union[str, list[str]]
            Lock label, if None set all keys in self.lock.

        Returns
        -------
        None.

        """
        if keys is None:
            keys = list(self.metaframe.lock.keys())
        if isinstance(keys, str):
            keys = [keys]
        lock = {key: self.metaframe.lock[key] for key in keys}
        self.metaframe.lock |= {key: status for key in keys}
        yield
        self.metaframe.lock |= lock
