import appdirs
from contextlib import contextmanager
import fsspec
import os
from pathlib import Path
import pytest
import sys
import tempfile

import nova
from nova.database.filepath import (
    WORKTREE_PATH_EXEMPTIONS,
    WORKTREE_ROOT,
    FilePath,
    _worktree_path_text,
    relativize_worktree_paths,
    repository_relative,
    worktree_path_candidates,
    worktree_path_offenders,
)
from nova.definitions import root_dir
from nova.utilities.importmanager import mark_import

ROOT = Path(__file__).resolve().parents[1]

HOSTNAME = "sdcc-login04.iter.org"

with mark_import("ssh") as mark_ssh:
    from nova.database.connectssh import ConnectSSH

    mark_connect = ConnectSSH(HOSTNAME).mark

if any(mark_ssh.args[0]):
    mark_connect = mark_ssh

mark_win32 = pytest.mark.skipif(
    sys.platform == "win32", reason="Skip ssh filesystem check on win32 platform"
)

IMAS_VERSION = os.environ.get("IMAS_VERSION", "")
TEMPDIR = tempdir = tempfile.gettempdir()

KEYPATH = dict(
    nova=os.path.join(
        nova.__name__, nova.__version__.replace(".post", "+").split("+")[0]
    ),
    imas=os.path.join("imas", IMAS_VERSION),
    root=root_dir,
)


def test_filepath():
    filepath = FilePath(filename="tmp.nc", dirname=TEMPDIR)
    assert filepath.filepath == Path(TEMPDIR) / "tmp.nc"


def test_filepath_error():
    filepath = FilePath(filename="", dirname="/tmp")
    with pytest.raises(FileNotFoundError):
        filepath.filepath


@pytest.mark.parametrize(
    "path",
    [
        "/tmp",
        "/tmp.nova",
        "/tmp.nova.imas",
        "/tmp.nova.subpath",
        "/tmp.imas.subpath/data",
        "/tmp.imas.nova",
        "site_config",
        "site_data",
        "user_cache",
        "user_config",
        "user_data",
        "user_data.nova",
        "site_data.nova.subpath",
        "user_cache.nova.imas",
        ".magnets/data",
        "root",
        "root.diagnostic/data",
        "root.nova",
        "root.nova.subpath",
        "root.subpath.nova",
        "root.nova.imas",
        "root.subpath",
    ],
)
def test_path(path):
    filepath = FilePath(parents=4)
    filepath.path = path
    default = filepath.basename
    paths = (path if len(path) > 0 else default for path in path.split("."))
    paths = (
        (
            getattr(appdirs, "_".join(path.split("_", 3)[:2]) + "_dir")()
            if path[:4] in ["user", "site"]
            else path
        )
        for path in paths
    )
    resolved_path = os.path.join(*(KEYPATH.get(path, path) for path in paths))
    assert filepath.path == Path(resolved_path)


def test_local_filesystem():
    filepath = FilePath()
    assert isinstance(filepath.fsys, fsspec.implementations.local.LocalFileSystem)


@mark_win32
@mark_ssh
@mark_connect
def test_ssh_filesystem():
    filepath = FilePath(hostname=HOSTNAME, dirname="/tmp")
    assert isinstance(filepath.fsys, fsspec.implementations.sftp.SFTPFileSystem)


@mark_ssh
@mark_connect
def test_ssh_appdirs_error():
    with pytest.raises(FileNotFoundError):
        FilePath(hostname=HOSTNAME, dirname="/tmp.nova", parents=1)


def test_mkdepth_error():
    filepath = FilePath(parents=2)
    with pytest.raises(FileNotFoundError):
        filepath.path = "root.imas.nova.data"


@contextmanager
def clear(path):
    filepath = FilePath(parents=4)
    if filepath.fsys.isdir(path):
        filepath.fsys.delete(path, True, 1)
    yield filepath
    if filepath.fsys.isdir(path):
        filepath.fsys.delete(path, True, 1)


@pytest.mark.parametrize("subpath", ["", "signal"])
def test_checkdir(subpath):
    path = Path(TEMPDIR) / "_filepath" / "data"
    with clear(path) as filepath:
        filepath.path = path
        filepath.path /= subpath
        filepath.makepath()
        assert filepath.is_path()


@mark_win32
@mark_ssh
@mark_connect
def test_checkdir_ssh():
    path = "/tmp/nova_test_filepath"
    with clear(path) as filepath:
        filepath.host = HOSTNAME
        filepath.path = path
        filepath.path /= ".nova"
        filepath.makepath()
        assert filepath.is_path()


def test_filepath_setter():
    filepath = FilePath()
    filepath.filepath = "/tmp/data/file.nc"
    assert filepath.path == Path("/tmp/data")
    assert filepath.filename == "file.nc"


def test_repository_relative_resolves_against_its_own_root():
    target = Path(root_dir) / "nova" / "database" / "filepath.py"
    assert repository_relative(target) == "nova/database/filepath.py"


def test_repository_relative_prefers_the_nearest_repository_root(tmp_path):
    """A path nested in another checkout resolves against that root, not ours."""
    inner = tmp_path / "checkout"
    (inner / ".git").mkdir(parents=True)
    target = inner / "docs" / "figure.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    assert repository_relative(target) == "docs/figure.png"


def test_relativize_maps_a_worktree_path_to_its_trailing_name():
    path = (
        f"/home/user/Code/.{WORKTREE_ROOT}"
        "/nova-a0f1e0938fc2/session-a/node-b/docs/figures/x/receipt.json"
    )
    rewritten, unmapped = relativize_worktree_paths(f"see {path} for details")
    assert rewritten == "see docs/figures/x/receipt.json for details"
    assert unmapped == []


def test_relativize_maps_a_bare_worktree_root_to_dot():
    path = f"/home/user/Code/.{WORKTREE_ROOT}/nova-a0f1e0938fc2/session-a/node-b"
    rewritten, unmapped = relativize_worktree_paths(path)
    assert rewritten == "."
    assert unmapped == []


def test_relativize_covers_the_legacy_cache_root():
    path = (
        f"/home/user/.cache/{WORKTREE_ROOT}"
        "/nova-a0f1e0938fc2/session-a/node-b/nova/database/filepath.py"
    )
    rewritten, unmapped = relativize_worktree_paths(path)
    assert rewritten == "nova/database/filepath.py"
    assert unmapped == []


def test_relativize_changes_nothing_outside_the_matched_span():
    """The same bytes elsewhere and the same line ending survive."""
    path = (
        f"/home/user/Code/.{WORKTREE_ROOT}"
        "/nova-a0f1e0938fc2/session-a/node-b/docs/x.json"
    )
    text = f'line one\r\n{{"source": "{path}", "n": 1}}\r\n'
    rewritten, unmapped = relativize_worktree_paths(text)
    assert rewritten == 'line one\r\n{"source": "docs/x.json", "n": 1}\r\n'
    assert unmapped == []


def test_relativize_is_idempotent():
    path = (
        f"/home/user/Code/.{WORKTREE_ROOT}"
        "/nova-a0f1e0938fc2/session-a/node-b/docs/x.json"
    )
    once, _ = relativize_worktree_paths(path)
    twice, unmapped = relativize_worktree_paths(once)
    assert twice == once
    assert unmapped == []


def test_relativize_reports_a_foreign_worktree_unmapped_and_unchanged():
    path = (
        f"/home/user/Code/.{WORKTREE_ROOT}/ambix-deadbeef/session-a/node-b/docs/x.json"
    )
    rewritten, unmapped = relativize_worktree_paths(path)
    assert rewritten == path
    assert unmapped == [path]


def test_relativize_reports_an_incomplete_layout_unmapped():
    path = f"/home/user/.cache/{WORKTREE_ROOT}"
    rewritten, unmapped = relativize_worktree_paths(path)
    assert rewritten == path
    assert unmapped == [path]


def test_repository_relative_maps_a_removed_worktree_lexically():
    """A worktree path maps through the shared matcher without a .git marker."""
    path = (
        f"/home/nobody/Code/.{WORKTREE_ROOT}"
        "/nova-a0f1e0938fc2/session-a/node-b/docs/figures/x/figure.png"
    )
    assert repository_relative(path) == "docs/figures/x/figure.png"


def test_worktree_path_offenders_skips_the_exemption_set():
    contents = {
        "docs/figures/x/a.json": f"see .{WORKTREE_ROOT}/nova-a0f1e0938fc2/s/n/f.json",
        "docs/plans/p.html": f"mentions .{WORKTREE_ROOT} as a subject",
        "docs/research/r.html": f".{WORKTREE_ROOT}",
        "docs/state/s.json": f".{WORKTREE_ROOT}",
        "nova/database/filepath.py": f'WORKTREE_ROOT = "{WORKTREE_ROOT}"',
        "nova/database/scripts.py": "calls the subcommand, no literal",
    }
    offenders = worktree_path_offenders(contents, contents.__getitem__)
    assert offenders == ["docs/figures/x/a.json"]
    assert len(WORKTREE_PATH_EXEMPTIONS) == 4


def test_worktree_path_text_decodes_a_binary_file():
    """A reader returns decoded text, not '' -- a binary offender is visible."""
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "artifact.bin").write_bytes(
            b"\x00\xff" + WORKTREE_ROOT.encode() + b"/nova-abc/s/n/x\x00"
        )
        text = _worktree_path_text(directory, "artifact.bin")
    assert text != ""
    assert WORKTREE_ROOT in text


def test_worktree_path_candidates_search_the_working_tree():
    candidates = worktree_path_candidates(ROOT)
    assert "nova/database/filepath.py" in candidates


def test_worktree_path_candidates_search_the_index():
    candidates = worktree_path_candidates(ROOT, cached=True)
    assert "nova/database/filepath.py" in candidates


if __name__ == "__main__":
    pytest.main([__file__])
