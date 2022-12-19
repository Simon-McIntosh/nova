"""Manage file data access for frame and biot instances."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import wraps
from importlib import import_module
import os
from pathlib import Path

import appdirs
import fsspec

import nova
from nova.definitions import root_dir


def stardot(func):
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
    parents: int = 2
    fsys: fsspec.filesystem = field(init=False, repr=False)

    def __post_init__(self):
        """Set host and path. Forward post init for cooperative inheritance."""
        self.host = self.hostname
        self.path = self.dirname
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    @property
    def host(self):
        """Manage filesysetm on host."""
        return self.hostname

    @host.setter
    def host(self, hostname: str | None):
        match hostname:
            case str():
                self.fsys = fsspec.filesystem('ssh', host=hostname)
            case None:
                self.fsys = fsspec.filesystem('file')
            case _:
                raise NotImplementedError('filesystem for hostname '
                                          f'{hostname} not implemented')
        self.hostname = hostname

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
                    path = str(self.basename)
                path = self._resolve_absolute(path)
                self.path = '.'.join((path, *subpath))
            case [str(path), str(subpath), *rest]:
                subpath = self._resolve_relative(subpath)
                path = os.path.join(path, str(subpath))
                self.path = '.'.join((path, *rest))
            case _:
                raise IndexError(f'unable to match dirname {dirname}')

    @staticmethod
    @stardot
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
    @stardot
    def _resolve_relative(path: str) -> str:
        """Return resolved relative path."""
        match path:
            case str('nova'):
                return os.path.join(nova.__name__,
                                    nova.__version__.split('+')[0])
            case str('imas'):
                try:
                    return import_module('imas').__name__
                except ImportError:
                    return 'imas'
            case str(path):
                return path
            case _:
                raise ValueError(f'unable to resolve relative path {path}')

    def checkpath(self) -> str:
        """Return existing parent. Raise if not found beyond self.parents."""
        for i, parent in zip(range(self.parents), self.path.parents):
            if self.fsys.isdir(str(parent)):
                return parent
        raise FileNotFoundError(f'directory not found for {parent} with a '
                                f'parent number of {self.parents} and '
                                f'{len(self.path.parents)} parents')

    def is_file(self) -> bool:
        """Return status of filesystem isfile evaluated on host."""
        return self.fsys.isfile(str(self.filepath))

    def is_path(self) -> bool:
        """Return status of filesystem isdir evaluated on host."""
        return self.fsys.isdir(str(self.path))

    def makepath(self):
        """Make path if not found."""
        if not self.is_path():
            self.fsys.makedirs(str(self.path), exist_ok=True)

    @property
    def filepath(self):
        """Return full filepath."""
        if self.filename is None:
            raise FileNotFoundError('filename is None')
        self.makepath()
        return self.path / self.filename

    @filepath.setter
    def filepath(self, filepath):
        path = Path(filepath)
        self.path = path.parent
        self.filename = path.name

    '''
    def mkdir(self, subpath=None):
        """Make path if not found."""
        print('mkdir', subpath)
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
    '''

if __name__ == '__main__':

    filepath = FilePath(parents=2, basename='root', filename='test')

    filepath.path = '.nova'
    print(filepath.filepath)

    filepath.filepath = '/home/mcintos/Code/nova/nova/2022.3.0/tests'
    filepath.filename
