"""Manage script access to extrapolate class."""
from importlib import import_module

import click

from nova.imas.extrapolate import Extrapolate

class ResType(click.ParamType):
    """
    Define resolution combined type.

    resolution: int | str

    """

    name: str = 'RES'

    def convert(self, value, param, ctx):
        """Return value with converted type."""
        try:
            return int(value)
        except ValueError:
            if isinstance(value, str):
                return value
        self.fail(f'{value!r} is not a valid interger or string', param, ctx)


class LimType(click.ParamType):
    """
    Define limit combinted.

    resolution: float | str

    """

    name: str = 'LIM'

    def convert(self, value, param, ctx):
        """Return value with converted type."""
        try:
            return float(value)
        except ValueError:
            pass
        if value == 'ids':
            return value
        self.fail(f'{value!r} is not a valid float or "ids"',
                  param, ctx)


@click.group(invoke_without_command=True,
             context_settings={'show_default': True, 'max_content_width': 160})
@click.argument('pulse', type=int)
@click.argument('run', type=int)
@click.option('-ngrid', 'ngrid', type=ResType(), default=5000,
              help="""\b
                      interpolation grid node number (aprox.)
                          int: user defined node number
                          ids: node number is read from equilibrium ids
                        """)
@click.option('-nplasma', 'nplasma', type=int, default=2500,
              help='plasma filiment number (aprox.)')
@click.option('-limit', 'limit', type=LimType(), default=0.25,
              help="""\b
                      interpolation grid limits
                          float: expansion factor applied to index
                          str: {ids} grid limits are read from equilibrium ids
                        """)
@click.option('-index', 'index',
              type=click.Choice(['coil', 'plasma'], case_sensitive=True),
              default='plasma',
              help='coil subset, used iif type(limit) == float')
@click.option('-pf_active', 'pf_active', nargs=2, type=int,
              default=(111001, 202),
              help='pf_active machine description ids')
@click.option('-wall', 'wall', nargs=2, type=int,
              default=(116000, 2),
              help='first wall machine description ids')
@click.option('-scenario_db', 'scenario_db',
              nargs=2, type=str, default=('public', 'iter'),
              help='scenario database (user, machine)')
@click.option('-machine_db', 'machine_db',
              nargs=2, type=str, default=('public', 'iter_md'),
              help='machine database (user, machine)')
@click.option('-backend', 'backend', default='HDF5',
              type=click.Choice(['HDF5', 'MDSPLUS', 'MEMORY',
                                 'ASCII'], case_sensitive=True),
              help='access layer backend')
@click.version_option(package_name='nova', message='%(package)s %(version)s')


@click.pass_context
def extrapolate(ctx, pulse, run, ngrid, nplasma,
                limit, index, pf_active, wall,
                scenario_db, machine_db, backend):
    """
    Extrapolate poloidal flux and magnetic field beyond separatrix.

    Reads flux functions from equilibrium IDS and solves for coil currents
    in a least squares sense to fit known poloidal flux interior to separatrix.

    This funciton is a command line script.
    Python workflows should use the nova.imas.extrapolate.Extrapolate class
    as a imas actor.

    \b
    Examples
    --------
    Extrapolate from LCFS CORSICA solution to standard grid and plot
    radial field.

    >>> extrapolate 130506 403 plot 25 -attr br
    """
    backend = dict(HDF5=13, MDSPLUS=12,
                   MEMORY=14, ASCII=11).get(backend.upper())
    machine_db = dict(zip(['user', 'machine'], machine_db))
    machine_db |= dict(backend=backend)
    pf_active = machine_db | dict(zip(['pulse', 'run'], pf_active))
    wall = machine_db | dict(zip(['pulse', 'run'], wall))
    ctx.obj = import_module('nova.imas.extrapolate').Extrapolate(
        pulse, run, ngrid=ngrid, nplasma=nplasma, limit=limit, index=index,
        user=scenario_db[0], machine=scenario_db[1], backend=backend,
        pf_active=pf_active, wall=wall)


@extrapolate.command()
@click.argument('itime', type=int)
@click.option('-attr', default='psi',
              type=click.Choice(['psi', 'br', 'bz']))
@click.pass_context
def plot(ctx, itime, attr):
    """Plot extrapolation result at itime."""
    plt = import_module('matplotlib.pyplot')
    ctx.obj.ionize(itime)
    ctx.obj.plot_2d(attr)
    plt.show()
