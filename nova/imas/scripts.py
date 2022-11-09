
import click

try:
    import imas
    IMPORT_IMAS = True
except ImportError:
    IMPORT_IMAS = False
from nova.imas.database import Database
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
        self.fail(f'{value!r} is not a valid float, list[float], or "ids"',
                  param, ctx)


@click.group(invoke_without_command=True,
             context_settings={'show_default': True, 'max_content_width': 160})
@click.option('-eq', '--equilibrium', 'equilibrium', nargs=2, type=int,
              required=True, help='equilibrium ids [pulse, run]')
@click.option('-ngrid', '--grid_number', 'grid_number',
              type=ResType(), default=5000,
              help="""\b
                      interpolation grid node number (aprox.)
                          int: user defined node number
                          ids: node number from equilibrium ids
                        """)
@click.option('-nplasma', '--plasma_number', 'plasma_number',
              type=click.IntRange(min=1),
              default=2000, help='plasma filiment number (aprox.)')
@click.option('-lim', '--limit', 'limit', type=LimType(), default=0,
              help="""\b
                      interpolation grid limits
                          float: expansion factor applied to index
                          ids: grid limits from equilibrium ids
                        """)
@click.option('-idx', '--index', 'index',
              type=click.Choice(['coil', 'plasma'], case_sensitive=True),
              default='plasma',
              help='coil subset, used iif type(limit) == float')
@click.option('-pf', '--pf_active', 'pf_active', nargs=2, type=int,
              default=(111001, 202),
              help='pf_active machine description ids')
@click.option('-fw', '--first_wall', 'wall', nargs=2, type=int,
              default=(116000, 2),
              help='first wall machine description ids')
@click.option('-sdb', '--scenario_db', 'scenario_db',
              nargs=2, type=str, default=('public', 'iter'),
              help='scenario database (user, machine)')
@click.option('-mdb', '--machine_db', 'machine_db',
              nargs=2, type=str, default=('public', 'iter_md'),
              help='machine database (user, machine)')
@click.option('-b', '--backend', 'backend', default='HDF5',
              type=click.Choice(['MDSPLUS', 'HDF5'], case_sensitive=True),
              help='access layer backend')
@click.version_option(package_name='nova',
                      message=f'{Extrapolate.__module__}.%(prog)s, '
                              'version %(version)s')
@click.pass_context
def extrapolate(ctx, equilibrium, grid_number, plasma_number,
                limit, index, pf_active, wall,
                scenario_db, machine_db, backend):
    """
    Extrapolate poloidal flux and magnetic field beyond separatrix.

    Reads flux functions from equilibrium IDS and solves for coil currents
    in a least squares sense to fit known poloidal flux interior to separatrix.

    """
    equilibrium_ids = Database(
        *equilibrium, 'equilibrium',
        **dict(zip(['user', 'machine'], scenario_db))).ids
    backend = getattr(imas.imasdef, f'{backend}_BACKEND')
    ctx.obj = Extrapolate(
        ids=equilibrium_ids, resolution=grid_number, nplasma=plasma_number,
        limit=limit, index=index,
        geometry=dict(zip(['pf_active', 'wall'], [pf_active, wall])),
        **dict(zip(['user', 'machine'], machine_db)), backend=backend)


@extrapolate.command()
@click.option('-i', '--itime', 'itime', default=0)
@click.option('--attr', 'attr', default='psi',
              type=click.Choice(['psi', 'br', 'bz']))
@click.pass_context
def plot(ctx, itime, attr):
    """Define input wall ids."""
    ctx.obj.ionize(itime)
    ctx.obj.plot(attr)


@extrapolate.command()
@click.pass_context
def put(ctx):
    pass
