"""Manage post install configuration for virtual environments."""

from pathlib import Path
import sys
from textwrap import dedent

import click

from nova.definitions import root_dir


@click.command()
def configure():
    """Generate module file for virtual environment installations."""
    if sys.prefix == sys.base_prefix:  # installed in base environment
        return
    prefix = Path(sys.prefix)
    startup = Path(root_dir).joinpath("scripts/startup.py")
    version = prefix.name
    moduledir = Path.joinpath(Path.home(), "modules/modulefiles/nova")
    Path(moduledir).mkdir(parents=True, exist_ok=True)

    module = """\
    #%Module

    if { [module-info mode load] || [module-info mode switch2] } {
        puts stdout "source _prefix/bin/activate;"
    } elseif { [module-info mode remove] && ![module-info mode switch3] } {
        puts stdout "deactivate;"
    }

    setenv PYTHONSTARTUP _startup
    setenv MYPYPATH _mypy
    """

    module = module.replace("_prefix", str(prefix))
    module = module.replace("_startup", str(startup))
    module = module.replace("_mypy", str(root_dir))
    modulefile = Path.joinpath(moduledir, version)

    with open(modulefile, "w") as file:
        file.write(dedent(module))

    modulefile.chmod(0o644)


if __name__ == "__main__":
    configure()
