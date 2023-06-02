"""Manage user generated data."""
import click
import shutil

from nova.database.filepath import FilePath


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
