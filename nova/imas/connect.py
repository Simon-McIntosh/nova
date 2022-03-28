"""Query and copy / rsync IMAS ids from SDCC to local filesytem."""
from dataclasses import dataclass, field
import io
import subprocess
from typing import Union, ClassVar

import fabric
import pandas


@dataclass
class Connect:
    """Copy and rsync IMAS ids."""

    cluster: str = 'sdcc-login02.iter.org'
    modules: tuple[str] = ('IMAS',)
    frame: pandas.DataFrame = field(init=False, repr=False)
    columns: list[str] = field(
        default_factory=lambda: ['shot', 'run', 'database', 'ip', 'b0',
                                 'fuelling', 'workflow', 'ref_name',
                                 'confinement'])

    _space_columns: ClassVar[list[str]] = ['ref_name', 'confinement']

    def __post_init__(self):
        """Connect to cluster."""
        self.ssh = fabric.Connection(self.cluster)
        self.username = self.ssh.run('whoami', hide=True).stdout.strip()

    @property
    def _module_load_string(self):
        """Return module load command."""
        return ' && '.join([f'ml load {module}' for module in self.modules])

    def module_run(self, command: str, hide=True):
        """Run command and return stdout."""
        command = f'{self._module_load_string} && {command}'
        return self.ssh.run(command, hide=hide).stdout

    def _read_summary(self, columns: str, select: str) -> str:
        """Return scenario summary."""
        return self.module_run(f'scenario_summary -c {columns} -s {select}')

    def unique(self, column: str):
        """Print unique column values."""
        text = self.module_run(f'scenario_summary -c {column}')
        frame = self._to_dataframe(text, delimiter=r'\s\s+')
        print(frame.iloc[:, 0].unique())

    def _to_dataframe(self, summary_string, delimiter=r'\s+'):
        """Convert summart string to pandas dataframe."""
        return pandas.read_csv(io.StringIO(summary_string),
                               delimiter=delimiter, skiprows=[0, 2],
                               skipfooter=1, engine='python')

    def load_frame(self, key: str, value: Union[str, int, float]):
        """Load scenario summary to dataframe, filtered by key value pair."""
        columns = [col for col in self.columns
                   if col not in self._space_columns]
        if key not in columns:
            columns.append(key)
        space_columns = [col for col in self._space_columns
                         if col in self.columns]
        if key not in space_columns:
            space_columns.append(key)
        self.frame = self._to_dataframe(self._read_summary(
            ','.join(columns), value))
        if len(space_columns) == 0:
            return
        space_frame = self._to_dataframe(self._read_summary(
            ','.join(space_columns), value), delimiter=r'\s\s+')
        for i, col in enumerate(space_frame):
            self.frame[col] = space_frame.iloc[:, i]
        return self.frame

    def _copy_command(self, ids: str, backend: str):
        """Return ids copy string."""
        command = [f'idscp {ids} -u public -si {shot} -ri {run} '
                   f'-so {shot} -ro {run} -do iter -bo {backend}'
                   for shot, run in zip(self.frame.Pulse, self.frame.Run)]
        return ' && '.join(command)

    def copy_frame(self, *ids_names: str, backend='MDSPLUS', hide=False):
        """Copy frame from root to global."""
        for ids in ids_names:
            self.module_run(self._copy_command(ids, backend), hide=hide)

    def rsync(self):
        """Syncronize local IMAS database with SDCC public."""
        public = f'/home/ITER/{self.username}/public/imasdb/iter/'
        local = f'/home/{self.username}/imas/shared/imasdb/iter/'
        command = f'rsync -aP {self.cluster}:{public} {local}'
        subprocess.run(command.split())

    def subframe(self, index):
        """Return subframe."""
        if isinstance(index, int):
            index = slice(index, index)
        return self.frame.loc[index, :]


if __name__ == '__main__':

    connect = Connect()
    #connect.unique('workflow')
    connect.load_frame('workflow', 'ASTRA')
    #for workflow in ['CORSICA', 'DINA-IMAS']:
    #    connect.load_frame('workflow', workflow)
    connect.copy_frame('equilibrium', 'pf_active', 'pf_passive')
    connect.rsync()
