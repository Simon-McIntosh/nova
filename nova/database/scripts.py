"""Manage user generated data."""

import subprocess

import click
import shutil
from pathlib import Path

from nova.database.filepath import (
    FilePath,
    relativize_worktree_paths,
    worktree_path_offenders,
)


@click.group(
    invoke_without_command=True,
    context_settings={"show_default": True, "max_content_width": 160},
)
@click.option("-dir", "dirname", default=".nova", type=str)
@click.option("-base", "basename", default="user_data", type=str)
@click.version_option(package_name="nova", message="%(package)s %(version)s")
@click.pass_context
def filepath(ctx, dirname, basename):
    """Manage nova filepath."""
    ctx.obj = FilePath(dirname=dirname, basename=basename)


@filepath.command
@click.pass_context
def clear(ctx):
    """Clear local file cache."""
    if ctx.obj.is_path():
        shutil.rmtree(ctx.obj.path)


def _staged_files():
    """Return the paths staged in the index, one entry per file."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        capture_output=True,
        check=True,
    )
    return [name for name in result.stdout.decode("utf-8").split("\0") if name]


def _staged_text(path):
    """Return the staged (index) content of ``path``."""
    result = subprocess.run(
        ["git", "show", f":{path}"], capture_output=True, check=False
    )
    return result.stdout.decode("utf-8", errors="replace")


@filepath.command
@click.option(
    "--check",
    "check",
    is_flag=True,
    help="Check the staged files only; exit nonzero naming each offender.",
)
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def relativize(files, check):
    """Rewrite nova worktree paths in FILE... to repository-relative names.

    Each file is rewritten in place, preserving every byte outside the matched
    spans.  Each file changed and each worktree-path span left unmapped is
    printed.  With ``--check`` the staged files are inspected instead and no
    file is modified: each staged file outside the exemptions that names a
    worktree path is reported as an offender and the command exits nonzero.
    """
    if check:
        offenders = worktree_path_offenders(_staged_files(), _staged_text)
        for path in offenders:
            click.echo(
                f"{path}: names a worktree path; "
                f"run `filepath relativize {path}` to rewrite it"
            )
        if offenders:
            raise SystemExit(1)
        return
    for name in files:
        source = Path(name)
        text = source.read_bytes().decode("utf-8")
        rewritten, unmapped = relativize_worktree_paths(text)
        if rewritten != text:
            source.write_bytes(rewritten.encode("utf-8"))
            click.echo(f"rewrote {name}")
        for span in unmapped:
            click.echo(f"unmapped {name}: {span}")


if __name__ == "__main__":
    filepath()
