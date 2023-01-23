import appdirs
from contextlib import contextmanager
import fsspec
import os
import paramiko
from pathlib import Path
import pytest

import nova
from nova.database.filepath import FilePath
from nova.definitions import root_dir
try:
    import imas
    IMPORT_IMAS = True
    IMASNAME = imas.__name__
except ImportError:
    IMPORT_IMAS = False
    IMASNAME = 'imas'

HOSTNAME = 'sdcc-login01.iter.org'

try:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(HOSTNAME)
    VALID_HOST = True
except:
    VALID_HOST = False


mark_imas = pytest.mark.skipif(not IMPORT_IMAS, reason='failed to import imas')
mark_ssh = pytest.mark.skipif(not VALID_HOST,
                              reason='unable to connect to ssh host')


KEYPATH = dict(nova=os.path.join(nova.__name__,
                                 nova.__version__.split('+')[0]),
               imas=IMASNAME, root=root_dir)


def test_filepath():
    filepath = FilePath(filename='tmp.nc', dirname='/tmp')
    assert filepath.filepath == Path('/tmp/tmp.nc')


def test_filepath_error():
    filepath = FilePath(filename='', dirname='/tmp')
    with pytest.raises(FileNotFoundError):
        filepath.filepath


@pytest.mark.parametrize('path', [
    '/tmp', '/tmp.nova', '/tmp.nova.imas', '/tmp.nova.subpath',
    '/tmp.imas.subpath/data', '/tmp.imas.nova', 'site_config',
    'site_data', 'user_cache', 'user_config', 'user_data', 'user_data.nova',
    'site_data.nova.subpath', 'user_cache.nova.imas', '.magnets/data',
    'root', 'root.diagnostic/data', 'root.nova', 'root.nova.subpath',
    'root.subpath.nova', 'root.nova.imas', 'root.subpath'])
def test_path(path):
    filepath = FilePath(parents=4)
    filepath.path = path
    default = filepath.basename
    paths = (path if len(path) > 0 else default for path in path.split('.'))
    paths = (getattr(appdirs, '_'.join(path.split('_', 3)[:2])+'_dir')() if
             path[:4] in ['user', 'site'] else path for path in paths)
    resolved_path = os.path.join(*(KEYPATH.get(path, path) for path in paths))
    assert filepath.path == Path(resolved_path)


def test_local_filesystem():
    filepath = FilePath()
    assert isinstance(filepath.fsys,
                      fsspec.implementations.local.LocalFileSystem)


@mark_ssh
def test_ssh_filesystem():
    filepath = FilePath(hostname=HOSTNAME, dirname='/tmp')
    assert isinstance(filepath.fsys,
                      fsspec.implementations.sftp.SFTPFileSystem)


@mark_ssh
def test_ssh_appdirs_error():
    with pytest.raises(FileNotFoundError):
        FilePath(hostname=HOSTNAME, dirname='user_data')


def test_mkdepth_error():
    filepath = FilePath(parents=2)
    with pytest.raises(FileNotFoundError):
        filepath.path = 'root.imas.nova.data'


@contextmanager
def clear(path):
    filepath = FilePath(parents=4)
    if filepath.fsys.isdir(path):
        filepath.fsys.delete(path, True)
    yield filepath
    if filepath.fsys.isdir(path):
        filepath.fsys.delete(path, True)


@pytest.mark.parametrize('subpath', ['', 'signal'])
def test_checkdir(subpath):
    path = '/tmp/_filepath/data'
    with clear(path) as filepath:
        filepath.path = path
        filepath.path /= subpath
        filepath.makepath()
        assert filepath.is_path()


@mark_ssh
def test_checkdir_ssh():
    path = '/tmp/_filepath'
    with clear(path) as filepath:
        filepath.host = HOSTNAME
        filepath.path = path
        filepath.path /= '.nova'
        filepath.makepath()
        assert filepath.is_path()


def test_filepath_setter():
    filepath = FilePath()
    filepath.filepath = '/tmp/data/file.nc'
    assert filepath.path == Path('/tmp/data')
    assert filepath.filename == 'file.nc'


if __name__ == '__main__':

    pytest.main([__file__])
