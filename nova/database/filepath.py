"""Manage file data access for frame and biot instances."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from functools import wraps
from importlib import import_module
import os
from pathlib import Path

import appdirs
import fsspec

import nova
from nova.definitions import root_dir


def starpath(func):
    """Return resolved path with '.' replaced with '*'."""
    @wraps(func)
    def wrapper(path: str) -> str:
        return func(path).replace('.', '*')
    return wrapper


@dataclass
class FilePath:
    """Manage to access to data via store and load methods."""

    filename: str | None = None
    dirname: Path | str = ''
    basename: Path | str = 'user_data'
    hostname: str | None = None
    mkdepth: int = 3
    fsys: fsspec.filesystem = field(init=False, repr=False)

    def __post_init__(self):
        """Set host and path. Forward post init for cooperative inheritance."""
        self.host = self.hostname
        self.path = self.dirname
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    def appname_attrs(self, appname):
        """Return app name attributes."""
        if appname == 'dir':
            return {}

        if appname == 'nova':
            return dict(appname=nova.__name__,
                        version=nova.__version__.split('+')[0])
        if appname == 'imas':
            try:
                imas_name = import_module('imas').__name__
                version = f'{nova.__version__.split("+")[0]}/{imas_name}'
                return dict(appname=nova.__name__, version=version)
            except ImportError:
                pass
        return dict(appname=appname)

    @staticmethod
    @starpath
    def _resolve_absolute(path: str) -> str:
        """Return resolved absolute path."""

        def get_appdir(path: str) -> str:
            """Return appdir path."""
            try:
                return getattr(appdirs, f'{path}_dir')()
            except AttributeError as error:
                raise AttributeError(
                    f'{path} is not a valid appdirs path') from error

        match path.split('_'):
            case ['root']:
                return root_dir
            case ['user', 'cache' | 'config' | 'data']:
                return get_appdir(path)
            case ['site', 'config' | 'data']:
                return get_appdir(path)
            case _:
                raise ValueError(f'unable to resolve absolute path {path}')

    @staticmethod
    @starpath
    def _resolve_relative(path: str) -> str:
        """Return resolved relative path."""
        match path:
            case 'nova':
                return os.path.join(nova.__name__,
                                    nova.__version__.split('+')[0])
            case 'imas':
                try:
                    return import_module('imas').__name__
                except ImportError:
                    return 'imas'
            case str(path):
                return path
            case _:
                raise ValueError(f'unable to resolve relative path {path}')

    @property
    def path(self):
        """Manage file path."""
        return self.dirname

    @path.setter
    def path(self, dirname: str):
        if isinstance(dirname, Path):
            self.dirname = dirname
            self.checkpath()
            return
        match dirname.split('.'):
            case [str(path)] if path[:1] == os.path.sep:
                self.path = Path(dirname.replace('*', '.'))
            case [str(path), *subpath] if path[:1] != os.path.sep:
                if path == '':
                    path = self.basename
                path = self._resolve_absolute(path)
                self.path = '.'.join((path, *subpath))
            case [str(path), str(subpath), *rest]:
                subpath = self._resolve_relative(subpath)
                path = os.path.join(path, subpath)
                self.path = '.'.join((path, *rest))
            case _:
                raise IndexError(f'unable to match dirname {dirname}')

    def checkpath(self) -> str:
        """Return dirname at which path exists. Raise if not found."""
        dirname = self.path
        for mkdepth in range(self.mkdepth):
            if self.fsys.isdir(str(dirname)):
                return dirname
            dirname, tail = os.path.split(dirname)
            if dirname == os.path.sep:
                raise FileNotFoundError('root directory not found '
                                        f'{os.path.join(dirname, tail)}')
        raise FileNotFoundError(f'directory not found at mkdepth '
                                f'{self.mkdepth} for {self.path}')

    def mkdir(self, subpath=None):
        """Make path if not found."""
        path = self.path
        if subpath is not None:
            if not (dirname := os.path.dirname(subpath)):
                path = os.path.join(path, subpath)
        self.fsys.makedirs(path, exist_ok=True)
        return path

    def get_directory(self, filename, path):
        """Return directory."""
        if filename is None:
            return self.mkdir(path)
        return os.path.join(self.mkdir(path), filename)

    def get_filepath(self, filename=None, path=None):
        """Return filepath."""
        if filename is None:
            filename = self.filename
        return self.get_directory(filename, path)

    @property
    def filepath(self):
        """Return full filepath."""
        return self.get_filepath(self.filename)

    @property
    def host(self):
        """Manage filesysetm on host."""
        return self.hostname

    @host.setter
    def host(self, hostname: str | None):
        self.hostname = hostname
        self.update_filesystem()

    def update_filesystem(self):
        """Update fsspec filesystem attribute."""
        match self.hostname:
            case str():
                self.fsys = fsspec.filesystem('ssh', host=self.hostname)
            case None:
                self.fsys = fsspec.filesystem('file')

    @staticmethod
    def mode(filepath: str, mode=None) -> str:
        """Return file access mode."""
        if mode is not None:
            return mode
        if os.path.isfile(filepath):
            return 'a'
        return 'w'


if __name__ == '__main__':

    filepath = FilePath(mkdepth=2, basename='root')

    filepath.path = '.nova'
    print(filepath.path)
