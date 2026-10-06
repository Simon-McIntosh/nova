"""Manage user generated data."""

import click
import shutil
from pathlib import Path

from nova.database.filepath import (
    FilePath,
    _worktree_path_text,
    relativize_worktree_paths,
    worktree_path_candidates,
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


@filepath.command
@click.option(
    "--check",
    "check",
    is_flag=True,
    help="Check the index only; exit nonzero naming each offender.",
)
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def relativize(files, check):
    """Rewrite nova worktree paths in FILE... to repository-relative names.

    Each file is rewritten in place, preserving every byte outside the matched
    spans.  Each file changed and each worktree-path span left unmapped is
    printed.  With ``--check`` the index is searched instead and no file is
    modified: each index entry outside the exemptions whose content names a
    worktree path is reported as an offender and the command exits nonzero.
    """
    if check:
        root = Path(__file__).resolve().parents[2]
        candidates = worktree_path_candidates(root, cached=True)
        offenders = worktree_path_offenders(
            candidates, lambda path: _worktree_path_text(root, path)
        )
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
