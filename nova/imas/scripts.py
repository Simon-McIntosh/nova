
import click

from nova.imas.database import Database
from nova.imas.extrapolate import Extrapolate


'''
    resolution: int | str = 2500
    limit: float | list[float] | str = 0.25
    index: Union[str, slice, npt.ArrayLike] = 'plasma'
'''


class ResType(click.ParamType):
    """
    Define resolution combinted.

    resolution: int | str

    """

    name: str = '[int|ids]'

    def convert(self, value, param, ctx):
        """Return value with converted type."""
        try:
            return int(value)
        except ValueError:
            if isinstance(value, str):
                return value
        self.fail(f'{value!r} is not a valid interger or string', param, ctx)


@click.group(invoke_without_command=True,
             context_settings={'show_default': True, 'max_content_width': 160})
@click.option('-eq', '--equilibrium', 'equilibrium', nargs=2, type=int,
              help='equilibrium ids (pulse, run)')
@click.option('-res', '--resolution', 'resolution', type=ResType(),
              default=2000,
              help='interpolation grid resolution')
@click.option('-re', '--reslution', 'resoltion', type=ResType(),
              default=2000,
              help='interpolation grid resolution')
@click.option('-pf', '--pf_active', 'pf_active', nargs=2, type=int,
              default=(111001, 202),
              help='pf_active machine description ids')
@click.option('-fw', '--first_wall', 'wall', nargs=2, type=int,
              default=(116000, 2),
              help='first wall machine description ids')
@click.option('-s', '--scenario_database', 'scenario_database',
              nargs=2, type=str, default=('public', 'iter'),
              help='scenario database (user, machine)')
@click.option('-md', '--machine_database', 'machine_database',
              nargs=2, type=str, default=('public', 'iter_md'),
              help='machine description database (user, machine)')
@click.option('-b', '--backend', 'backend', default='MDSPLUS',
              type=click.Choice(['MDSPLUS', 'HDF5'], case_sensitive=True),
              help='access layer backend')
@click.version_option(package_name='nova',
                      message=f'{Extrapolate.__module__}.%(prog)s, '
                              'version %(version)s')
@click.pass_context
def extrapolate(ctx, equilibrium, pf_active, wall,
                scenario_database, machine_database, backend):
    """
    Extrapolate poloidal flux and magnetic field beyond separatrix.

    Reads flux functions from equilibrium IDS and solves coil currents
    in a least squares sense to fit poloidal flux interior to separatrix.

    """
    equilibrium_ids = Database(*equilibrium, 'equilibrium', machine='iter').ids
    ctx.obj = Extrapolate(ids=equilibrium_ids,
                          dplasma=-200, resolution=500,
                          limit='ids')


@extrapolate.command()
@click.option('-i', '--itime', 'itime', default=0)
@click.option('--attr', 'attr', default='psi',
              type=click.Choice(['psi', 'br', 'bz']))
@click.pass_context
def plot(ctx, itime, attr):
    """Define input wall ids."""
    click.echo(f'wall {ctx}')
    ctx.obj.ionize(itime)
    ctx.obj.plot(attr)


@extrapolate.command()
@click.pass_context
def put(ctx):
    pass
