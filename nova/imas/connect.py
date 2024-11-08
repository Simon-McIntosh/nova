"""Query and copy / rsync IMAS ids from SDCC to local filesytem."""

from dataclasses import dataclass, field
import io
import pathlib
import re
import subprocess
from typing import ClassVar

import fabric
import numpy as np
import pandas
from tqdm import tqdm

from nova.imas.database import Datafile


@dataclass
class Connect:
    """Copy and rsync IMAS ids."""

    command: str
    machine: str
    user: str = "public"
    cluster: str = "sdcc-login04.iter.org"
    backend: str = "HDF5"
    modules: tuple[str] = ("IMAS/3.41.0-2024.07-foss-2023b",)
    columns: list[str] = field(default_factory=lambda: [])
    frame: pandas.DataFrame = field(
        init=False, repr=False, default_factory=pandas.DataFrame
    )

    _space_columns: ClassVar[list[str]] = []

    def __post_init__(self):
        """Connect to cluster."""
        self.ssh = fabric.Connection(self.cluster)
        self.username = self.ssh.run("whoami", hide=True).stdout.strip()
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    @property
    def _module_load_string(self):
        """Return module load command."""
        return " && ".join([f"ml load {module}" for module in self.modules])

    def module_run(self, command: str, hide=True):
        """Run command and return stdout."""
        command = f"{self._module_load_string} && {command}"
        return self.ssh.run(command, hide=hide).stdout

    def read_summary(self, columns: str, select: str | int | float) -> str:
        """Return scenario summary."""
        return self.module_run(f"{self.command} -c {columns} -s {select}")

    def unique(self, column: str):
        """Return unique column values."""
        text = self.read_summary(column, select="''")
        frame = self.to_dataframe(text, delimiter=r"\s\s+")
        return frame.iloc[:, 0].unique()

    def to_dataframe(self, summary_string, delimiter=r"\s+"):
        """Convert summary string to pandas dataframe."""
        for skip in [" NOTE"]:
            summary_string = re.sub(
                rf"^{skip}.*\n?", "", summary_string, flags=re.MULTILINE
            )
        frame = pandas.read_csv(
            io.StringIO(summary_string),
            delimiter=delimiter,
            skiprows=[0, 2],
            skipfooter=1,
            engine="python",
        )
        columns = {
            col: col.lower() if "[" not in col else col.split("[")[0].strip()
            for col in frame
        }
        return frame.rename(columns=columns)

    def load_frame(self, key: str, value: str | int | float):
        """Load scenario summary to dataframe, filtered by key value pair."""
        columns = [col for col in self.columns if col not in self._space_columns]
        if key not in columns:
            columns.append(key)
        space_columns = [col for col in self._space_columns if col in self.columns]
        frame = self.to_dataframe(self.read_summary(",".join(columns), value))
        if len(space_columns) > 0:
            space_frame = self.to_dataframe(
                self.read_summary(",".join(space_columns), value), delimiter=r"\s\s+"
            )
            for i, col in enumerate(space_frame):
                frame[col] = space_frame.iloc[:, i]
        frame = frame.loc[frame[key] == value, :]
        frame = frame.astype(dict(pulse=int, run=int), errors="ignore")
        self.frame = pandas.concat([self.frame, frame], ignore_index=True)
        return self.frame

    def copy_command(self, frame: pandas.DataFrame, ids: str = ""):
        """Return ids copy string."""
        command = [
            f"idscp {ids} -a -u {self.user} "
            f"-si {pulse} -ri {run} -d {self.machine} -b HDF5 "
            f"-so {pulse} -ro {run} -do {self.machine.lower()} "
            f"-bo {self.backend}"
            for pulse, run in zip(frame.pulse, frame.run)
        ]
        return " && ".join(command)

    def copy_frame(self, *ids_names: str, hide=False):
        """Copy frame from shared to public on remote."""
        for ids in ids_names:
            self.module_run(self.copy_command(self.frame, ids), hide=hide)

    def rsync(self):
        """Syncronize SDCC remote (user)/public with local IMAS database."""
        public = f"/home/ITER/{self.username}/public/imasdb/{self.machine.lower()}/"
        local = pathlib.Path.home() / pathlib.Path(
            f"imas/shared/imasdb/{self.machine.lower()}/"
        )
        pathlib.Path(local).mkdir(parents=True, exist_ok=True)
        command = f"rsync -aP {self.cluster}:{public} {local}"
        subprocess.run(command.split())
        command = f"rsync -aP {local} {self.cluster}:{public}"
        subprocess.run(command.split())

    def subframe(self, index):
        """Return subframe."""
        if isinstance(index, int):
            index = slice(index, index)
        return self.frame.loc[index, :]


@dataclass
class ScenarioDatabase(Datafile, Connect):
    """Manage public and local scenario data."""

    command: str = "scenario_summary"
    machine: str = "iter"
    columns: list[str] = field(
        default_factory=lambda: [
            "shot",
            "run",
            "database",
            "ip",
            "b0",
            "fuelling",
            "workflow",
            "ref_name",
            "confinement",
        ]
    )
    dirname: str = field(default=".nova.imas")
    workflow: list[str] = field(
        default_factory=lambda: ["CORSICA", "ASTRA", "DINA-IMAS"]
    )

    _space_columns: ClassVar[list[str]] = ["ref_name", "confinement", "workflow"]

    def __post_init__(self):
        """Set file attributes."""
        self.filename = self.command
        self.workflow.sort()
        self.group = ",".join(self.workflow)
        super().__post_init__()

    def store(self):
        """Store frame to netCDF file."""
        self.data = self.frame.to_xarray()
        super().store()

    def load(self):
        """Load frame from netCDF file."""
        super().load()
        if len(self.data) > 0:
            self.frame = self.data.to_dataframe()

    def build(self):
        """Load scenario workflows into frame."""
        for workflow in tqdm(self.workflow, f"loading workflows {self.workflow}"):
            self.load_frame("workflow", workflow)
        self.store()

    def sync_workflow(self):
        """Sync scenario workflows with local repo."""
        self.module_run(self.copy_command(self.frame), hide=True)
        self.rsync()

    def sync_shot(self, *shot: str):
        """Sync scenario shots input as pulse/run string."""
        self.frame = pandas.DataFrame(
            np.array([s.split("/") for s in shot], int), columns=["pulse", "run"]
        )
        self.module_run(self.copy_command(self.frame), hide=False)
        self.rsync()


@dataclass
class MachineDatabase(Connect):
    """Manage public and local machine data."""

    command: str = "md_summary"
    machine: str = "iter_md"
    columns: list[str] = field(default_factory=lambda: ["pbs", "ids"])

    _space_columns: ClassVar[list[str]] = []

    def load_ids(self, *ids_names: str):
        """Load multiple ids to frame."""
        for ids in ids_names:
            self.load_frame("ids", ids)

    def to_dataframe(self, summary_string, delimiter=r"\s+"):
        """Extend Connect.to_dataframe to seperate shot/run column."""
        frame = super().to_dataframe(summary_string, delimiter)
        if "shot/run" not in frame:
            return frame
        frame["pulse"] = [value.split("/")[0] for value in frame["shot/run"]]
        frame["run"] = [value.split("/")[1] for value in frame["shot/run"]]
        frame.drop(columns=["shot/run"], inplace=True)
        return frame

    def copy_ids(self, hide=False):
        """Copy machine description ids to public on remore."""
        for ids in self.frame.ids.unique():
            index = self.frame.ids == ids
            self.module_run(self.copy_command(self.frame[index], ids), hide=hide)

    def sync_ids(self, *ids_names):
        """Sync ids names with local repo."""
        if len(ids_names) == 0:
            ids_names = [
                "pf_active",
                "tf",
                "pf_passive",
                "magnetics",
                "wall",
                "pulse_schedule",
            ]
        self.load_ids(*ids_names)
        self.copy_ids()
        self.rsync()
        return self

    def sync_shot(self, *shot: str):
        """Sync scenario shots input as pulse/run string."""
        self.frame = pandas.DataFrame(
            np.array([s.split("/") for s in shot], int), columns=["pulse", "run"]
        )
        self.module_run(self.copy_command(self.frame), hide=True)
        self.rsync()


if __name__ == "__main__":
    pass
    # machine = MachineDatabase().sync_ids()
    # machine.load_ids('pf_active')
    # print(machine.frame)
    # MachineDatabase(modules=("IMAS/3.37.0-4.11.0-2020b",)).sync_shot("111003/1")
    # MachineDatabase().sync_shot("111001/103")
    # MachineDatabase().rsync()

    # iter_md = MachineDatabase(machine="ITER_MD")
    # iter_md.sync_shot("111003/2")
    # ScenarioDatabase().sync_shot("111003/1")

    # ScenarioDatabase().sync_shot("111001/103")
    ScenarioDatabase(machine="aug", workflow=[]).rsync()
    # ScenarioDatabase(machine="west", workflow=[]).rsync()
    # ScenarioDatabase(user='tribolp').sync_shot('135011/21')
    # ScenarioDatabase(user='dubrovm').sync_shot('105028/1')
    # ScenarioDatabase(user='dubrovm').sync_shot('105027/1')

    # ScenarioDatabase().sync_shot("134173/106")
    # scenario = ScenarioDatabase()
    # scenario.sync_shot("115001/1")

    # ScenarioDatabase(workflow=[], machine="mast_u").rsync()

    # scenario = ScenarioDatabase()
    # scenario.load_frame("workflow", "DINA-IMAS")

    # scenario.sync_workflow()
    # ScenarioDatabase().sync_shot('135011/7')
